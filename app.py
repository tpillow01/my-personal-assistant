import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Set Streamlit page config
st.set_page_config(page_title="My Personal Assistant", layout="wide")

# Title
st.title("📁 My Personal Assistant")

# Session state for storing the AI chain
if "chain" not in st.session_state:
    st.session_state.chain = None

# Upload file
uploaded_file = st.file_uploader("Upload a document (PDF, Word, Excel)", type=["pdf", "docx", "doc", "xlsx"])

def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    file_ext = uploaded_file.name.split(".")[-1].lower()

    if file_ext == "pdf":
        loader = PyPDFLoader(tmp_file_path)
    elif file_ext in ["doc", "docx"]:
        loader = UnstructuredWordDocumentLoader(tmp_file_path)
    elif file_ext == "xlsx":
        loader = UnstructuredExcelLoader(tmp_file_path)
    else:
        st.error("Unsupported file type")
        return None

    return loader.load()

@st.cache_resource
def create_chain(_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(_docs)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    llm = ChatOpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")

    return (retriever, chain)

# Process file and build chain
if uploaded_file:
    with st.spinner("Processing document..."):
        docs = process_uploaded_file(uploaded_file)
        if docs:
            retriever, chain = create_chain(docs)
            st.session_state.chain = chain
            st.session_state.retriever = retriever
            st.success("Document processed! You can now ask questions.")

# Ask questions
if st.session_state.chain:
    user_question = st.text_input("Ask me something about your document:")
    if user_question:
        relevant_docs = st.session_state.retriever.get_relevant_documents(user_question)
        answer = st.session_state.chain.run(input_documents=relevant_docs, question=user_question)
        st.write("🤖", answer)
else:
    st.info("Upload a document to get started.")

