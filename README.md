# 📚 MultiDoc RAG Chatbot (Groq Powered)

An advanced **Retrieval-Augmented Generation (RAG) chatbot** that allows users to upload multiple PDFs and ask questions based on their content.

Powered by **Groq LLM + ChromaDB + LangChain**, this chatbot provides fast and accurate answers using document context.

---

## 🚀 Live Demo

🔗 https://huggingface.co/spaces/ShreyaAnde/multidoc-rag-chatbot-groq

---

## 🖼️ Screenshots



![Screenshot 1](Screenshot%20(400).png)
![Screenshot 2](Screenshot%20(402).png)
![Screenshot 3](Screenshot%20(403).png)


---

## ✨ Features

- 📂 Upload multiple PDF documents
- 🔍 Context-based semantic search using embeddings
- ⚡ Fast responses using Groq (LLaMA 3)
- 🧠 RAG pipeline (Retriever + LLM)
- 💬 Chat-style interface
- 🔄 Streaming responses
- 📦 Persistent vector database (ChromaDB)

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **LLM:** Groq (LLaMA 3.1)  
- **Embeddings:** Sentence Transformers  
- **Vector DB:** ChromaDB  
- **Framework:** LangChain  

---

## ⚙️ Installation (Local Setup)

```bash
# Clone repo
git clone https://github.com/ShreyaAnde/MultiDoc-RAG-Chatbot-Groq-Powered-.git

cd MultiDoc-RAG-Chatbot-Groq-Powered-

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
