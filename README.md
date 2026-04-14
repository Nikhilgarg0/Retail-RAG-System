# 🚚 Logistics RAG System

A Gemini-powered Retrieval-Augmented Generation (RAG) system for querying logistics documents.

---

## Features

- **Logistics-only validation** — PDFs are classified by Gemini before ingestion; non-logistics documents are rejected with a clear reason
- **Multi-PDF support** — Upload and query multiple PDFs at once; each is added incrementally (no full rebuilds)
- **Streamlit UI** — Clean, minimal chat interface with dark/light mode toggle
- **Source citations** — Every answer includes the source filename and page number

---

## Project Structure

```
logistics-rag/
├── app.py               # FastAPI backend (RAG + PDF validation)
├── streamlit_app.py     # Streamlit frontend
├── src/
│   └── main.py          # CLI version of the RAG system
├── data/
│   └── raw/             # Uploaded PDFs stored here
├── chroma_db/           # Auto-created vector store
├── static/              # (Legacy) HTML frontend assets
├── requirements-core.txt
└── .env                 # GOOGLE_API_KEY goes here
```

---

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements-core.txt
   ```

2. **Set your API key** — create a `.env` file:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

3. **Start the FastAPI backend**
   ```bash
   uvicorn app:app --reload
   ```

4. **Start the Streamlit frontend** (in a new terminal)
   ```bash
   streamlit run streamlit_app.py
   ```

5. Open `http://localhost:8501` in your browser.

---

## How it Works

1. Upload one or more PDFs via the sidebar
2. Each PDF is classified by Gemini — non-logistics documents are rejected
3. Accepted PDFs are chunked and embedded into ChromaDB
4. Ask questions in the chat; answers are grounded in your documents only
5. Source citations (filename + page) are shown below each answer

---

## Backend API (FastAPI)

| Endpoint | Method | Description |
|---|---|---|
| `/upload` | POST | Upload one or multiple PDFs (validated) |
| `/chat` | POST | Ask a question |
| `/documents` | GET | List all uploaded documents |
| `/documents/{name}` | DELETE | Remove a document |
| `/health` | GET | Backend health check |