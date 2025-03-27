import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
import PyPDF2
import docx2txt
import openpyxl
import pandas as pd
from datetime import datetime

from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="My Personal Assistant", layout="wide")
st.title("📁 My Personal Assistant")

if "chain" not in st.session_state:
    st.session_state.chain = None

uploaded_file = st.file_uploader("Upload a document (PDF, Word, Excel)", type=["pdf", "docx", "doc", "xlsx"])

# ────────────────────────────────────────────────
# 🔍 Document Scanner (PDF, Word, Excel)
# ────────────────────────────────────────────────
def scan_document_for_keyword(file, keyword, file_ext):
    results = []

    if file_ext == "pdf":
        reader = PyPDF2.PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if keyword.lower() in text.lower():
                    results.append((f"Page {page_num+1}", text[:500]))
            except:
                continue

    elif file_ext in ["doc", "docx"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        paragraphs = docx2txt.process(tmp_path).split("\n")
        for idx, para in enumerate(paragraphs):
            if keyword.lower() in para.lower():
                results.append((f"Paragraph {idx+1}", para.strip()[:500]))

    elif file_ext == "xlsx":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        wb = openpyxl.load_workbook(tmp_path, data_only=True)
        for sheet in wb.worksheets:
            for row in sheet.iter_rows(values_only=True):
                row_text = " ".join([str(cell) for cell in row if cell])
                if keyword.lower() in row_text.lower():
                    results.append((f"Sheet: {sheet.title}", row_text.strip()[:500]))

    return results

# ────────────────────────────────────────────────
# 🤖 Langchain AI Q&A Mode
# ────────────────────────────────────────────────
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

    return retriever, chain

# ────────────────────────────────────────────────
# 📅 Schedule Conflict Checker
# ────────────────────────────────────────────────
def detect_schedule_conflicts(file):
    df = pd.read_excel(file)
    st.write("### Schedule Preview:")
    st.dataframe(df)

    if not {'Name', 'Start Time', 'End Time'}.issubset(df.columns):
        st.error("Excel must include columns: Name, Start Time, End Time")
        return

    df['Start Time'] = pd.to_datetime(df['Start Time'])
    df['End Time'] = pd.to_datetime(df['End Time'])

    conflicts = []
    grouped = df.sort_values('Start Time').groupby('Name')

    for name, group in grouped:
        for i in range(len(group)-1):
            current = group.iloc[i]
            next_row = group.iloc[i+1]
            if current['End Time'] > next_row['Start Time']:
                conflicts.append({
                    'Name': name,
                    'Conflict 1': f"{current['Start Time']} - {current['End Time']}",
                    'Conflict 2': f"{next_row['Start Time']} - {next_row['End Time']}"
                })

    if conflicts:
        st.warning("Conflicts detected:")
        st.dataframe(pd.DataFrame(conflicts))
    else:
        st.success("✅ No conflicts found in the schedule!")

# ────────────────────────────────────────────────
# UI Mode Toggle + Logic
# ────────────────────────────────────────────────
mode = st.radio("Choose mode:", ["🔍 Document Scanner", "🤖 AI Chat", "📅 Schedule Conflict Checker"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()

    if mode == "🔍 Document Scanner":
        search_term = st.text_input("Enter a name or keyword to search for:")
        if search_term:
            with st.spinner("Scanning document..."):
                results = scan_document_for_keyword(uploaded_file, search_term, file_ext)
                if results:
                    st.success(f"Found {len(results)} match(es) for '{search_term}':")
                    for label, snippet in results:
                        with st.expander(f"🔎 {label}"):
                            st.write(snippet)
                else:
                    st.warning("No matches found.")

    elif mode == "🤖 AI Chat":
        if "last_uploaded_filename" not in st.session_state or uploaded_file.name != st.session_state.last_uploaded_filename:
            with st.spinner("Processing document for AI..."):
                docs = process_uploaded_file(uploaded_file)
                if docs:
                    retriever, chain = create_chain(docs)
                    st.session_state.chain = chain
                    st.session_state.retriever = retriever
                    st.session_state.last_uploaded_filename = uploaded_file.name
                    st.success("Document processed! Ask your question below:")

        if st.session_state.chain:
            user_question = st.text_input("Ask a question about your document:")
            if user_question:
                relevant_docs = st.session_state.retriever.get_relevant_documents(user_question)
                answer = st.session_state.chain.run(input_documents=relevant_docs, question=user_question)
                st.write("🤖", answer)

    elif mode == "📅 Schedule Conflict Checker":
        if file_ext != "xlsx":
            st.error("Please upload a valid Excel (.xlsx) schedule file.")
        else:
            detect_schedule_conflicts(uploaded_file)
else:
    st.info("📤 Upload a document to get started.")

