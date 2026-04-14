"""
app.py - FastAPI backend for Logistics RAG system
Run: uvicorn app:app --reload
"""
import os
import sys
import shutil
import json
import re
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
load_dotenv()

RAW_DATA_DIR  = PROJECT_ROOT / "data" / "raw"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
STATIC_DIR    = PROJECT_ROOT / "static"
COLLECTION    = "logistics_docs"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
TOP_K         = 5
GEMINI_MODEL  = "models/gemini-2.5-flash"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

LOGISTICS_KEYWORDS = [
    "shipment", "freight", "cargo", "transport", "delivery", "logistics",
    "warehouse", "inventory", "supply chain", "shipping", "dispatch",
    "consignment", "bill of lading", "customs", "import", "export",
    "carrier", "route", "fleet", "tracking", "order fulfillment",
    "distribution", "container", "pallets", "last mile", "3pl", "forwarder"
]

# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(title="Logistics RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
vectorstore = None
embeddings  = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def safe_delete_chromadb(path: Path):
    if path.exists():
        try:
            shutil.rmtree(path)
        except Exception as e:
            print(f"Warning: Could not delete {path}: {e}")


def is_logistics_document(pdf_path: Path):
    """
    Classify whether a PDF is logistics/transport related.
    Returns (is_logistics: bool, reason: str).
    """
    try:
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()

        sample_text = "\n\n".join(
            p.page_content for p in pages[:3] if p.page_content.strip()
        )[:3000]

        if not sample_text.strip():
            return False, "The PDF appears to be empty or unreadable."

        lowered = sample_text.lower()
        keyword_hits = [kw for kw in LOGISTICS_KEYWORDS if kw in lowered]

        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0,
            convert_system_message_to_human=True
        )

        classification_prompt = f"""You are a document classifier. Analyze the following text from a PDF and determine if it is related to logistics, transportation, supply chain, shipping, freight, or related domains.

Text sample:
\"\"\"
{sample_text}
\"\"\"

Keyword hints found: {keyword_hits if keyword_hits else 'none'}

Respond with ONLY a JSON object in this exact format (no markdown, no explanation):
{{"is_logistics": true, "confidence": "high", "reason": "one sentence explanation"}}

Be strict: only return true if the document is genuinely about logistics/transport/supply chain operations."""

        response = llm.invoke([("human", classification_prompt)])
        content  = response.content.strip()

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result.get("is_logistics", False), result.get("reason", "Classification complete.")

        return len(keyword_hits) >= 2, f"Keyword-based detection: {keyword_hits}"

    except Exception as e:
        print(f"Classification error: {e}")
        return True, "Classification service unavailable, document accepted."


def load_pdf_chunks(pdf_path: Path):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    loader = PyPDFLoader(str(pdf_path))
    pages  = loader.load()
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["source"] = pdf_path.name
    print(f"  ✓ {pdf_path.name}: {len(pages)} pages -> {len(chunks)} chunks")
    return chunks


def rebuild_vectorstore():
    global vectorstore
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise ValueError("No PDFs found in data/raw")

    print("\nRebuilding vector store from all PDFs...")
    safe_delete_chromadb(CHROMA_DB_DIR)

    all_chunks = []
    for pdf_path in pdf_files:
        all_chunks.extend(load_pdf_chunks(pdf_path))

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DB_DIR)
    )
    print(f"Vector store built with {len(all_chunks)} chunks from {len(pdf_files)} file(s)")
    return len(all_chunks)


def add_pdf_to_vectorstore(pdf_path: Path):
    global vectorstore
    chunks = load_pdf_chunks(pdf_path)

    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION,
            persist_directory=str(CHROMA_DB_DIR)
        )
    else:
        vectorstore.add_documents(chunks)

    print(f"Added {len(chunks)} chunks from {pdf_path.name}")
    return len(chunks)


def get_answer(question: str, include_sources: bool = True):
    global vectorstore

    if vectorstore is None:
        if CHROMA_DB_DIR.exists():
            vectorstore = Chroma(
                collection_name=COLLECTION,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DB_DIR)
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="No documents uploaded yet. Please upload a logistics PDF first."
            )

    docs = vectorstore.similarity_search(question, k=TOP_K)

    if not docs:
        return {
            "answer": "I couldn't find any relevant information in the uploaded documents.",
            "sources": []
        }

    context_parts = []
    sources = []
    for i, doc in enumerate(docs, 1):
        src  = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[Excerpt {i} | {src} | page {page}]\n{doc.page_content}")
        if include_sources:
            sources.append({
                "filename": src,
                "page": page,
                "content": doc.page_content[:200] + "..."
            })

    context = "\n\n".join(context_parts)

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        convert_system_message_to_human=True
    )

    prompt_template = """You are a helpful logistics assistant. Use ONLY the document excerpts below to answer the question.

Rules:
- Answer using ONLY the information in the excerpts.
- Quote exact numbers, names, codes, and dates where possible.
- If the excerpts do not contain the answer, say exactly: "I could not find that information in the provided documents."
- Be concise and clear.

Document excerpts:
----------------
{context}
----------------

Question: {question}"""

    prompt   = ChatPromptTemplate.from_messages([("human", prompt_template)])
    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": sources if include_sources else []
    }


# ─────────────────────────────────────────────
# API MODELS
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    include_sources: bool = True


class ChatResponse(BaseModel):
    answer: str
    sources: list


class UploadResponse(BaseModel):
    message: str
    files_processed: list
    files_rejected: list
    total_chunks: int


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload one or multiple PDFs. Each is validated as logistics content before ingestion."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    accepted     = []
    rejected     = []
    total_chunks = 0

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            rejected.append({"filename": file.filename, "reason": "Only PDF files are allowed."})
            continue

        file_path = RAW_DATA_DIR / file.filename
        content   = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        print(f"\nClassifying: {file.filename}")
        is_logistics, reason = is_logistics_document(file_path)

        if not is_logistics:
            print(f"  Rejected: {reason}")
            file_path.unlink(missing_ok=True)
            rejected.append({
                "filename": file.filename,
                "reason": f"Not a logistics document: {reason}"
            })
            continue

        print(f"  Accepted: {reason}")
        try:
            chunks = add_pdf_to_vectorstore(file_path)
            total_chunks += chunks
            accepted.append({"filename": file.filename, "chunks": chunks})
        except Exception as e:
            file_path.unlink(missing_ok=True)
            rejected.append({"filename": file.filename, "reason": f"Processing error: {str(e)}"})

    if not accepted and rejected:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "All uploaded files were rejected.",
                "rejected": rejected
            }
        )

    return UploadResponse(
        message=f"Processed {len(files)} file(s): {len(accepted)} accepted, {len(rejected)} rejected.",
        files_processed=accepted,
        files_rejected=rejected,
        total_chunks=total_chunks
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = get_answer(request.question, request.include_sources)
        return ChatResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/documents")
async def list_documents():
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
    return {
        "documents": [
            {"filename": f.name, "size_kb": round(f.stat().st_size / 1024, 2)}
            for f in pdf_files
        ]
    }


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    file_path = RAW_DATA_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    file_path.unlink()
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))

    if pdf_files:
        chunks_created = rebuild_vectorstore()
        return {
            "message": f"Deleted {filename} and rebuilt vector store.",
            "remaining_documents": len(pdf_files),
            "chunks_created": chunks_created
        }
    else:
        safe_delete_chromadb(CHROMA_DB_DIR)
        global vectorstore
        vectorstore = None
        return {
            "message": f"Deleted {filename}. No documents remaining.",
            "remaining_documents": 0,
            "chunks_created": 0
        }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "vectorstore_initialized": vectorstore is not None
    }


@app.on_event("startup")
async def startup_event():
    global vectorstore
    if CHROMA_DB_DIR.exists():
        try:
            vectorstore = Chroma(
                collection_name=COLLECTION,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DB_DIR)
            )
            count = vectorstore._collection.count()
            print(f"Loaded vector store with {count} vectors")
        except Exception as e:
            print(f"Warning: Could not load vector store: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)