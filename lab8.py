"""
Simple Streamlit chatbot that answers questions grounded in a fixed PDF using
FAISS for vector storage and LangChain's RetrievalQA with a Hugging Face model.

Requires packages (install via pip):
	streamlit langchain langchain-community langchain-openai langchain-text-splitters
	sentence-transformers faiss-cpu pypdf huggingface-hub

Run the app:
	streamlit run lab8.py
"""

import os
from typing import Any, List, Optional

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.llms import LLM
from huggingface_hub import InferenceClient


def require_hf_token() -> None:
	"""Show a warning if the Hugging Face API token is missing."""
	if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
		st.warning(
			"Set HUGGINGFACEHUB_API_TOKEN to call the Hugging Face Inference API.",
			icon="⚠️",
		)


@st.cache_resource(show_spinner=False)
def build_vectorstore(pdf_path: str):
	"""Create and cache the FAISS vector store for the given PDF path."""
	loader = PyPDFLoader(pdf_path)
	docs = loader.load()

	splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
	chunks = splitter.split_documents(docs)

	embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
	vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
	return vectorstore


@st.cache_resource(show_spinner=False)
def build_qa_chain(pdf_path: str):
	"""Create and cache the RetrievalQA chain for the given PDF path."""
	vectorstore = build_vectorstore(pdf_path)
	retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

	# HF Inference API LLM (uses HUGGINGFACEHUB_API_TOKEN)
	hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
	client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)
	llm = HFClientLLM(client=client, temperature=0.1, max_new_tokens=512)

	qa_chain = RetrievalQA.from_chain_type(
		llm=llm,
		chain_type="stuff",
		retriever=retriever,
		return_source_documents=True,
	)
	return qa_chain


def render_app():
	st.set_page_config(page_title="RetrievalQA Chatbot", page_icon="💬", layout="centered")
	st.title("RetrievalQA Chatbot")
	st.write(
		"Ask questions grounded in a fixed PDF. The app embeds the PDF into FAISS and "
		"uses LangChain's RetrievalQA to answer with context."
	)

	require_hf_token()

	uploaded = st.file_uploader("Upload a PDF", type=["pdf"], help="Optional: upload instead of providing a path")
	pdf_path = st.text_input(
		"PDF path",
		value="D:/Lab8/Checklist.pdf",
		help="Absolute or relative path to the PDF to load (ignored if you upload)",
	)
	question = st.text_input("Your question", placeholder="e.g., What does section 2 say?")

	if st.button("Get answer"):
		# If a file was uploaded, save it to a temp path for processing.
		if uploaded is not None:
			tmp_path = os.path.join(st.experimental_get_query_params().get("tmpdir", ["."])[0], "uploaded.pdf")
			with open(tmp_path, "wb") as f:
				f.write(uploaded.read())
			pdf_to_use = tmp_path
		else:
			if not pdf_path.strip():
				st.info("Please provide a PDF path or upload a file.")
				return
			if not os.path.isfile(pdf_path):
				st.error(f"PDF not found at: {pdf_path}")
				return
			pdf_to_use = pdf_path
		if not question.strip():
			st.info("Please enter a question first.")
			return
		try:
			qa_chain = build_qa_chain(pdf_to_use)
		except Exception as exc:  # keep broad here to show errors to user
			st.error(f"Failed to load or embed PDF: {exc}")
			return

		with st.spinner("Thinking..."):
			result = qa_chain.invoke({"query": question})

		st.subheader("Answer")
		st.write(result.get("result", "No answer produced."))

		source_docs = result.get("source_documents") or []
		if source_docs:
			st.subheader("Source snippets")
			for idx, doc in enumerate(source_docs, start=1):
				snippet = doc.page_content.replace("\n", " ")
				st.caption(f"{idx}. {snippet[:250]}…")


class HFClientLLM(LLM):
	client: InferenceClient
	max_new_tokens: int = 512
	temperature: float = 0.1

	@property
	def _llm_type(self) -> str:  # pragma: no cover
		return "huggingface_hub_inference_client_chat"

	def _call(
		self,
		prompt: str,
		stop: Optional[List[str]] = None,
		run_manager: Optional[Any] = None,
		**kwargs: Any,
	) -> str:
		# Use chat completion; the instruct model supports conversational.
		messages = [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": prompt},
		]
		resp = self.client.chat_completion(
			messages,
			max_tokens=self.max_new_tokens,
			temperature=self.temperature,
			stop=stop,
		)
		if not resp or not getattr(resp, "choices", None):
			return ""
		content = resp.choices[0].message.content if hasattr(resp.choices[0].message, "content") else resp.choices[0].message.get("content")
		if isinstance(content, list):
			# content may be a list of parts; join text components
			content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
		return str(content) if content is not None else ""


if __name__ == "__main__":
	render_app()
