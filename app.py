import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from groq import Groq

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="RAG Chatbot (Groq)", layout="wide")

# ✅ STREAMLIT SECRETS (REPLACES .env)
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

USER_ID = "user_123"
DB_PATH = f"./db/{USER_ID}"
os.makedirs(DB_PATH, exist_ok=True)

# ----------------------------
# EMBEDDINGS
# ----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------
# SESSION STATE
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ----------------------------
# PROCESS PDF FILES
# ----------------------------
def process_files(uploaded_files):
    all_docs = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            temp_path = tmp.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        all_docs.extend(docs)

        os.remove(temp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.split_documents(all_docs)

# ----------------------------
# CREATE VECTOR DB
# ----------------------------
def create_vectorstore(docs):
    if not docs:
        raise ValueError("No text found in uploaded PDFs")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )

    vectorstore.persist()
    return vectorstore

# ----------------------------
# LOAD VECTOR DB
# ----------------------------
def load_vectorstore():
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )

# ----------------------------
# RAG + GROQ RESPONSE
# ----------------------------
def generate_answer_stream(query):

    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    docs = retriever.invoke(query)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.
If answer is not in context, say "I don't know based on the document".

Context:
{context}

Question:
{query}

Answer clearly:
"""

    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# ----------------------------
# SIDEBAR - UPLOAD
# ----------------------------
st.sidebar.title("📂 Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Upload documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        docs = process_files(uploaded_files)
        st.session_state.vectorstore = create_vectorstore(docs)

    st.sidebar.success("Documents indexed successfully!")

elif os.path.exists(DB_PATH):
    st.session_state.vectorstore = load_vectorstore()

# ----------------------------
# CHAT UI
# ----------------------------
st.title("📚 RAG Chatbot (Groq + Chroma)")

st.markdown("Ask questions from your uploaded PDFs")

# show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask something...")

if query:

    if st.session_state.vectorstore is None:
        st.warning("Please upload PDFs first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        for chunk in generate_answer_stream(query):
            full_response += chunk
            response_container.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )