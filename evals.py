import matplotlib.pyplot as plt
import os
import requests
import asyncio
import time
import pandas as pd
import re
import fitz
from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings

# Clean console output
warnings.filterwarnings("ignore", category=DeprecationWarning)
from dotenv import load_dotenv


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "RAG_eval"

# LLM judge for Ragas evaluation and testset generation
llm_model = ChatOpenAI(
	model="gpt-4o-mini",
	temperature=0.0
)
lc_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

# Ragas metrics
metrics = [
	faithfulness,		# measures if the answer is derived strictly from context
	answer_relevancy,	# measures if the answer directly addresses the user question
	context_precision,	# measures if the retrieved chunks are relevant to the ground truth
	context_recall		# measures if the retrieved context covers all info needed for the answer
]

# Cuts off the text content 
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

# Identify content boundaries
def is_header_line(text: str) -> bool:
	chapter_patterns = [
		r"chapter\s+\d+", r"section\s+\d+", r"introduction", r"conclusion",
		r"appendix", r"summary", r"discussion", r"part\s+\d+"
	]
	text_lower = text.strip().lower()
	return any(re.search(pattern, text_lower) for pattern in chapter_patterns)

# Identify start of bibliography sections
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

# Remove blocks of reference
def filter_out_references_sections(pages: list[Document]) -> list[Document]:
	result_docs = []
	skip_mode = False

	for page in pages:
		lines = [line.strip() for line in page.page_content.split('\n') if line.strip()]
		if not lines:
			if not skip_mode:
				result_docs.append(page)
			continue

		kept_lines = []
		page_in_skip = False

		for line in lines:
			if skip_mode:
				if is_header_line(line):
					skip_mode = False
					kept_lines.append(line)
			else:
				if is_references_line(line):
					skip_mode = True
					page_in_skip = True
				else:
					kept_lines.append(line)

		if kept_lines and not page_in_skip:
			filtered_content = "\n".join(kept_lines)
			result_docs.append(
				Document(page_content=filtered_content, metadata=page.metadata.copy())
			)
		elif not skip_mode and not page_in_skip:
			result_docs.append(page)

	return result_docs

# Pipeline: API health check, testset generation, inference, ragas evaluation
async def main_eval():
	max_retries = 30
	print("Waiting for API...")

	# Health check for local RAG API 
	for i in range(max_retries):
		try:
			check = requests.get("http://api:8000/status", timeout=2)
			if check.status_code == 200:
				print("API ready.")
				break
		except:
			if i % 5 == 0:
				print(f"Waiting for API ({i}/{max_retries})")
			time.sleep(5)
			continue
	else:
		print("API not reachable. Exiting.")
		return

	# Load testset or generate new from document 
	try:
		test_df = pd.read_csv("my_testset.csv")
		print("Loaded existing testset")
	except FileNotFoundError:
		print("File does not exist. Creating new testset")
		all_pages = load_clean_pdf(".pdf",footer_margin=60)

		# Isolate main content
		start_keywords = [r"\b1\.?\s+introduction\b", r"\bchapter\s+1\b", r"\babstract\b"]
		skip_keywords = ["contents", "table of contents", "brief table of contents"]
		stop_keywords = [r"\bbibliography\b", r"\breferences\b"]
		start_index = 0
		stop_index = len(all_pages)

		# Slice PDF pages
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
		for i in range(start_index, len(all_pages)):
			lines = [line.strip() for line in all_pages[i].page_content.lower().split('\n') if line.strip()]

			if not lines:
				continue
			header = " ".join(lines[:8])
			header = re.sub(r"\s+", " ", header)

			if any(skip in header for skip in skip_keywords):
				continue
			found_stop = False
			for key in stop_keywords:
				if re.search(key, header):
					if i > start_index + 5: # safety margin to prevent premature cutting
						print(f"\nCut book on page {i+1}")
						print(f"Found word: '{key}'")
						print(f"Read header: '{header}'\n")
						stop_index = i
						found_stop = True
						break
			if found_stop:
				break

			if any(re.search(key, header) for key in stop_keywords):
				if i > start_index + 5:
					stop_index = i
					break

		docs = all_pages[start_index:stop_index]
		print(f"Retrieved text from pages: {start_index + 1}-{stop_index}")
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

		# Testset generation using Ragas and GPT-4o-mini
		generator = TestsetGenerator.from_langchain(
			llm=llm_model,
			embedding_model=embeddings
		)

		testset = generator.generate_with_langchain_docs(
			split_docs,
			testset_size=50,
			transforms = [],
			raise_exceptions=True
		)

		test_df = testset.to_pandas()
		test_df.to_csv("my_testset.csv", index=False)
		print("Testset generated and saved")

	# Inference loop: send generated questions to RAG API
	results = []
	print(f"Testing {len(test_df)} generated questions")
	for _, row in test_df.iterrows():
		question = row['user_input']
		reference = row['reference']
		try:
			start_time = time.time()
			response = requests.post(
				"http://api:8000/ask",
				json={"session_id": "auto_eval", "question": row['user_input']},
				timeout=240
			)
			latency = time.time() - start_time
			if response.status_code == 200:
				data = response.json()
				print(f"Received response {len(results) + 1}/{len(test_df)}")
				results.append({
					"user_input": question,
					"response": data["answer"],
					"retrieved_contexts": data["context_texts"],
					"reference": row['reference'],
					"latency_sec": latency
					})
			else:
				print(f"API error {response.status_code}: {response.text}")
				results.append({"task_completion": 0, "error": response.text})
		except Exception as e:
			print(f"Request failed: {str(e)}")
			results.append({
				"user_input": question,
				"response": "Error: API connection failed",
				"retrieved_contexts": [],
				"reference": reference,
				"latency_sec": 0
			})
	if not results:
		print("No results to evaluate.")
		return

	# Ragas metrics
	print("Starting Ragas evaluation")
	dataset = EvaluationDataset.from_list(results)
	result = evaluate(
		dataset=dataset,
		metrics=metrics,
		llm=llm_model
	)
	eval_df = (result.to_pandas())
	eval_df.to_csv("results_eval.csv", index=False)
	
	pd.set_option('display.max_columns', None)
	print("\n"+"="*50)
	print("\nEvaluation")
	print("="*50)
	for index, row in eval_df.iterrows():
		print(f"\nQuestion {index + 1}")
		print(f"Question: {row['user_input']}")
		print(f"API Answer: {row['response']}")
		print("\n" + "-"*50)
		print(f"Metrics:")
		print(f"Faithfulness: {row['faithfulness']:.2f}")
		print(f"Answer Relevancy: {row['answer_relevancy']:.2f}")
		print(f"Context Precision: {row['context_precision']:.2f}")
		print(f"Context Recall: {row['context_recall']:.2f}")
		print("="*50)
	print("\n"+"-"*50)

	# Summarize results
	print(f"Average results:")
	mean_metrics = eval_df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean()
	print(mean_metrics.round(2).to_string())
	print("="*50)
if __name__ == "__main__":
	asyncio.run(main_eval())