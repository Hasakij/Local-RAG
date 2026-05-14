import streamlit as st
import requests
import time

st.set_page_config(page_title="Local RAG", layout="centered")
st.title("Local RAG")

API_URL = "http://api:8000"

# Initializing session state for chat and uploader
if "uploader_key" not in st.session_state:
	st.session_state.uploader_key = 0

if "messages" not in st.session_state:
	st.session_state.messages = []

if "session_id" not in st.session_state:
	st.session_state.session_id = "session"

with st.sidebar:
	st.header("Documents")
	uploaded_files = st.file_uploader(
		"Upload pdf file",
		type="pdf",
		accept_multiple_files=True,
		key=f"uploader_{st.session_state.uploader_key}"
	)

	if st.button("Send to database"):
		if uploaded_files:
			with st.spinner('Processing file...'):
				count = 0
				all_success = True
				for uploaded_file in uploaded_files:
					files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
					response = requests.post(f"{API_URL}/upload", files=files)
					if response.status_code == 200:
						count += 1
					else:
						st.error(f"Error for {uploaded_file.name}: {response.text}")
						all_success = False
				if count > 0:
					st.success(f"Added {count} file(s) to database")
					st.session_state.uploader_key += 1 # reset uploader
					time.sleep(2)
					st.rerun()
				
	st.divider()
	st.subheader("Files in database")

	# Retrieving list of files in DB
	try:
		status_res = requests.get(f"{API_URL}/status")
		if status_res.status_code == 200:
			files_in_db = status_res.json().get("files_in_database", [])
			if not files_in_db:
				st.info("Empty database")
			else:
				with st.container():
					for file_path in files_in_db:
						file_name = file_path.split("/")[-1]

						c1, c2 = st.columns([3, 1])
						c1.text(f"{file_name}")

						if c2.button("Del", key=f"del_{file_name}"):
							with st.spinner(f"Deleting {file_name}..."):
								delete_res = requests.delete(f"{API_URL}/delete/{file_name}")
								if delete_res.status_code == 200:
									time.sleep(1.0)
									st.success(f"Removed {file_name}")
									st.rerun()
								else:
									st.error("Error deleting")
						st.write("")
		else:
			st.sidebar.warning("Busy database...")
	except Exception as e:
		st.write("")
		st.caption("Syncing with database")

# Chat history				
for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.markdown(message["content"])

# User query
if prompt := st.chat_input("Ask question: "):
	st.session_state.messages.append({"role": "user", "content": prompt})
	with st.chat_message("user"):
		st.markdown(prompt)

	with st.chat_message("assistant"):
		with st.spinner("Thinking..."):
			payload = {
			"session_id": st.session_state.session_id,
			"question": prompt
			}
			res = requests.post(f"{API_URL}/ask", json=payload)
			if res.status_code == 200:
				response_data = res.json()
				final_text = response_data.get("answer", "No answer")
				sources = response_data.get("sources", [])
				st.markdown(final_text)
				if sources:
					st.caption(f"Sources: {', '.join(sources)}")

				st.session_state.messages.append({"role": "assistant", "content":final_text})
			else:
				st.error("Error connecting with model")