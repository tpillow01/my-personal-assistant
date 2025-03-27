import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
import tempfile
import os

load_dotenv()

st.set_page_config(page_title="My Personal Assistant", layout="centered")

st.markdown("""
    <style>
    .stApp {background-color:#000;color:#FFF;}
    h1,h2,h3,h4,h5,h6 {color:#FFF;}
    .stButton>button {background-color:#FFF;color:#000;}
    .stTextInput input {background-color:#222;color:#FFF;}
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 My Personal Assistant")
st.markdown("Ask general questions, or upload a document for specialized assistance!")

@st.cache_resource
def create_chain(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    llm = ChatOpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")

    return (retriever, chain)

def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = None
    if uploaded_file.type == "application/pdf":
        loader = PyMuPDFLoader(tmp_file_path)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        loader = Docx2txtLoader(tmp_file_path)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        loader = UnstructuredExcelLoader(tmp_file_path)
    else:
        st.error("Unsupported file type!")
        return []

    docs = loader.load()
    os.unlink(tmp_file_path)
    return docs

if 'chain' not in st.session_state:
    st.session_state.chain = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload document (optional):", type=["pdf", "docx", "xlsx"])

if uploaded_file:
    docs = process_uploaded_file(uploaded_file)
    if docs:
        st.session_state.chain = create_chain(docs)
        st.success("✅ Document uploaded! Now you can ask document-specific questions.")

user_question = st.text_input("Ask me anything:", key="user_input")

if st.button("Send") and user_question:
    if st.session_state.chain:
        response = st.session_state.chain({"question": user_question, "chat_history": st.session_state.chat_history})
        answer = response['answer']
    else:
        # General AI chat (no document context)
        llm = ChatOpenAI(temperature=0.5, model='gpt-3.5-turbo')
        answer = llm.predict(user_question)

    st.session_state.chat_history.append((user_question, answer))
    st.markdown(f"**🧑 You:** {user_question}")
    st.markdown(f"**🤖 Assistant:** {answer}")

if st.session_state.chain and st.button("Summarize Document"):
    summary = st.session_state.chain({"question": "Summarize this document concisely.", "chat_history": []})['answer']
    st.markdown(f"**📌 Summary:** {summary}")


