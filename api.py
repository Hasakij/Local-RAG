import os
import re
import shutil
import chromadb
import warnings
import traceback
import fitz
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import GPT4All
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv

# Environment and logging setup
warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

# Configuration for LangSmith and API observability
os.environ["USER_AGENT"] = "Local-RAG-Bot"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG_eval"

app = FastAPI(title="LocalRagAPI")
store = {}
rag_chain = None

# Short-term session memory for chatbot
def get_session_history(session_id: str) :
	if session_id not in store:
		store[session_id] = ChatMessageHistory()
	# Last 2 messages
	if len(store[session_id].messages) > 2:
		store[session_id].messages = store[session_id].messages[-2:]
	return store[session_id]

# Detect structural elements in PDF to prevent skip_mode errors
def is_header_line(text: str) -> bool:
	chapter_patterns = [
		r"chapter\s+\d+", r"section\s+\d+", r"introduction", r"conclusion",
		r"appendix", r"summary", r"discussion", r"part\s+\d+"
	]
	text_lower = text.strip().lower()
	return any(re.search(pattern, text_lower) for pattern in chapter_patterns)

# Identifying the start of non-informative sections
def is_references_line(line: str) -> bool:
	ref_patterns = [
		r"^\s*references\s*$",
		r"^\s*bibliography\s*$",
		r"^\s*works\s+cited\s*$",
		r"^\s*further\s+reading\s*$",
		r"^\s*cited\s+references\s*$"
	]
	line_clean = re.sub(r"[:\.\-\—\–\*]*$", "", line.strip().lower())
	return any(re.match(pattern, line_clean) for pattern in ref_patterns)

# Strip reference sections
def filter_out_references_sections(pages: list[Document]) -> list[Document]:
	result_docs = []
	skip_mode = False

	for page in pages:
		# Extract and clean lines from the current page
		lines = [line.strip() for line in page.page_content.split('\n') if line.strip()]
		if not lines:
			if not skip_mode:
				result_docs.append(page)
			continue

		kept_lines = []
		page_in_skip = False

		for line in lines:
			if skip_mode:
				# Look for a new seaction header to resume indexing
				if is_header_line(line):
					skip_mode = False
					kept_lines.append(line)
			else:
				# Check if the current line marks the start of a reference section
				if is_references_line(line):
					skip_mode = True
					page_in_skip = True
				else:
					kept_lines.append(line)

		# Reconstruct page content if it wasn't completely skipped
		if kept_lines and not page_in_skip:
			filtered_content = "\n".join(kept_lines)
			result_docs.append(
				Document(page_content=filtered_content, metadata=page.metadata.copy())
			)
		elif not skip_mode and not page_in_skip:
			result_docs.append(page)

	return result_docs

# Initializing AI Models: local LLM and Vector DB
@app.on_event("startup")
def load_ai_components():
	global embeddings, db, llm, prompt
	print("Initializing components...")
	try:
		# HF for local embeddings generation
		embeddings = HuggingFaceEmbeddings(
			model_name="sentence-transformers/all-mpnet-base-v2",
			model_kwargs={'device': 'cpu'}
		)

		# Connecting to ChromaDB container
		chroma_client = chromadb.HttpClient(host="vectordb", port=8000)
		db = Chroma(
			client=chroma_client,
			collection_name="documents",
			embedding_function=embeddings
		)

		# Loading Phi-3-mini
		llm = GPT4All(
			model="/app/Phi-3-mini-4k-instruct-q4.gguf",
			device='cuda',
			n_threads=6,
			max_tokens=1024,
			temp=0.0,
			repeat_penalty=1.1,
			stop=["<|end|>", "<|user|>", "Question:", "\nQuestion", "Written by", "Answer:", "AI Answer", "\n\n", "Subject:", "Dear"]
		)

		# System instructions and RAG constraints
		template = """<|system|>
		You are a precise document assistant. Answer the question using the context.
		Rules:
		1. Do not use outside knowledge. 
		2. If the context contains the answer, write a clear and direct response.
		3. If the context only partially answers the question, provide only that partial information and stop. Do not guess the rest.
		4. Do not add conversational filler.
		5. Never sign your response.
		<|end|>
		<|user|>
		CONTEXT:
		{context}
		
		QUESTION: {input}
		<|end|>
		<|assistant|>
		"""
		
		prompt = PromptTemplate(input_variables=["context", "input"], template=template)

		print("Server ready")
	except Exception as e:
		print(f"Initialization failed {e}")
		print(traceback.format_exc())

# Data models for API requests and responses
class QuestionRequest(BaseModel):
	session_id: str
	question: str

class AnswerResponse(BaseModel):
	answer: str
	sources: list[str]
	context_texts: list[str]

# Main inference endpoint
@app.post("/ask", response_model=AnswerResponse)
async def ask_bot(request: QuestionRequest):
	global llm, prompt, db
	if llm is None or db is None:
		raise HTTPException(status_code=503, detail="AI Models or Database not initialized")

	try:
		status = await get_db_status()
		files = status.get("files_in_database", [])
		if not files:
			raise HTTPException(status_code=404, detail="Empty database")

		# MMR to balance relevance and diversity
		retriever = db.as_retriever(
			search_type="mmr",
			search_kwargs={
				"k": 5, # top k documents
				"fetch_k":30, # initial pool of documents 
				"lambda_mult": 0.7 # diversity-relevance balance
			}
		)

		# Build and invoke the RAG chain
		document_chain = create_stuff_documents_chain(llm, prompt)
		rag_chain = create_retrieval_chain(retriever, document_chain)
		response = rag_chain.invoke({"input": request.question})

		# Post-process LLM output to remove common artifacts and tokens
		raw_answer = response.get("answer", "")
		clean_answer = re.split(r'<\|end\|>|===|<\|assistant\|>|Question:', raw_answer)[0].replace("support:", "").strip()
		
		# Prepare context data and formatted source metadata for the response
		raw_contexts = [doc.page_content for doc in response.get("context", [])]
		formatted_sources = [
			f"{os.path.basename(doc.metadata.get('source', 'unknown'))} (Page: {doc.metadata.get('page', '?')})"
			for doc in response.get("context", [])
		]
		
		return AnswerResponse(
			answer=clean_answer,
			sources=list(set(formatted_sources)), # deduplicate source list
			context_texts=raw_contexts
		)
	except Exception as e:
		print("Error during /ask")
		print(traceback.format_exc())
		raise HTTPException(status_code=500, detail=str(e))

# Detect and cut off text starting from common headers
def remove_reference_sections(text):
	patterns = [
		r"\breferences\b",
		r"\bfurther reading\b",
		r"\bchapter references\b"
	]
	lower = text.lower()
	for pattern in patterns:
		match = re.search(pattern, lower)
		if match:
			return text[:match.start()].strip()
	return text

# PDF preprocessing: footer stripping, reference removal and stuttering cleanup
def load_clean_pdf(pdf_path, footer_margin=50):
	filename = os.path.basename(pdf_path)
	doc = fitz.open(pdf_path)
	clean_docs = []
	for i,page in enumerate(doc):
		# Avoiding page numbers and headers/footers
		rect = page.rect
		text_area = fitz.Rect(0, 0, rect.width, rect.height - footer_margin)
		raw_text = page.get_text("text", clip=text_area)
		text = remove_reference_sections(raw_text)

		# Deduplicate lines and words to fix PDF parsing artifacts (stuttering)
		lines = text.split('\n')
		unique_lines = list(dict.fromkeys([line.strip() for line in lines if line.strip()]))
		final_lines = []
		for line in unique_lines:
			words = line.split()
			clean_line = " ".join(list(dict.fromkeys(words)))
			final_lines.append(clean_line)
		final_text = "\n".join(final_lines)		

		if text.strip():
			clean_docs.append(
				Document(
					page_content=final_text,
					metadata={
						"source": filename,
						"page": i + 1
						}
					)
				)
	return clean_docs

# Endpoint for uploading and indexing endpoint
@app.post("/upload")
async def upload_pdf(file:UploadFile = File(...)):
	try:
		existing_docs = db.get(where={"source": file.filename})
		if existing_docs and existing_docs['ids']:
			return {
				"message": f"File: {file.filename} already in database"
			}
		file_location = f"/app/{file.filename}"
		with open(file_location, "wb") as buffer:
			shutil.copyfileobj(file.file, buffer)
		print(f"Processing file: {file.filename}...")

		# Initial cleaning of the document
		all_pages = load_clean_pdf(file_location, footer_margin=60)
		start_keywords = [r"\b1\.?\s+introduction\b", r"\bchapter\s+1\b", r"\babstract\b"]
		skip_keywords = ["contents", "table of contents", "brief table of contents"]
		stop_keywords = [r"\bbibliography\b", r"\breferences\b"]
		start_index = 0
		stop_index = len(all_pages)

		# Determine starting point
		for i,page in enumerate(all_pages):
			lines = [line.strip() for line in page.page_content.lower().split('\n') if line.strip()]
			if not lines:
				continue
			header = " ".join(lines[:8])
			header = re.sub(r"\s+", " ", header)
			if any(skip in header for skip in skip_keywords):
				continue
			if any(re.search(key,header) for key in start_keywords):
				start_index = i
				break
		# Determine ending point
		for i in range(start_index, len(all_pages)):
			lines = [line.strip() for line in all_pages[i].page_content.lower().split('\n') if line.strip()]

			if not lines:
				continue
			header = " ".join(lines[:8])
			header = re.sub(r"\s+", " ", header)
			if any(skip in header for skip in skip_keywords):
				continue
			if any(re.search(key, header) for key in stop_keywords):
				if i > start_index + 5:
					stop_index = i
					break
		docs = all_pages[start_index:stop_index]
		print(f"Retrieved text from pages: {start_index + 1}-{stop_index}")

		# Split docs into semantic chunks for vectorization
		splitter = RecursiveCharacterTextSplitter(
			chunk_size=900,
			chunk_overlap=150
		)
		raw_split_docs = splitter.split_documents(docs)
		split_docs = []

		# Remove noisy chunks
		for chunk in raw_split_docs:
			text = chunk.page_content.lower()
			years = len(re.findall(r'\b(19|20)\d{2}\b', text))
			brackets = len(re.findall(r'\[\d+\]', text))
			dots_table = len(re.findall(r'\.\.\.\.', text))
			words = sum(1 for w in ['journal', 'arxiv', 'proceedings', 'pp.', 'vol.', 'press', 'doi:'] if w in text)

			# If a chunk looks like a reference list or table, skip it
			if years >= 3 or brackets >= 3 or words >= 2 or dots_table >= 2:
				continue
			split_docs.append(chunk)
		print(f"Removed {len(raw_split_docs) - len(split_docs)} chunks from text")
		db.add_documents(split_docs)
		return {
				"message": f"Processed file: {file.filename} (Pages {start_index+1}-{stop_index})",
				"vector_chunks_added": len(split_docs)
		}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"{str(e)}")

# Status endpoint
@app.get("/status")
async def get_db_status():
	try:
		existing_data = db.get()
		total_chunks = len(existing_data['ids'])
		sources = []
		for meta in existing_data['metadatas']:
			if meta is not None:
				sources.append(meta.get('source'))
		unique_files = list(set(sources))

		return {
		"total_vector_chunks": total_chunks,
		"files_in_database": unique_files
		}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

# Remove vector chunks
@app.delete("/delete/{filename}")
async def delete_file(filename: str):
	try:
		existing_docs = db.get(where={"source": filename})
		if existing_docs and existing_docs['ids']:
			db.delete(ids=existing_docs['ids'])
			return {"message": f"Deleted {filename} from database"}
		else:
			raise HTTPException(status_code=404, detail=f"File {filename} not found in database")

	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))