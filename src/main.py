"""
main.py - Gemini-powered RAG system (FIXED)
Run from project root: python src/main.py
"""
import os
import sys
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv

# Fix imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RAW_DATA_DIR   = PROJECT_ROOT / "data" / "raw"
CHROMA_DB_DIR  = PROJECT_ROOT / "chroma_db"
COLLECTION     = "logistics_docs"
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200
TOP_K          = 5
GEMINI_MODEL   = "gemini-2.5-flash"  # Change to gemini-1.5-pro if needed


# ─────────────────────────────────────────────
# HELPER: Safe delete ChromaDB
# ─────────────────────────────────────────────
def safe_delete_chromadb(path: Path, max_retries: int = 3):
    """Delete ChromaDB directory with retry logic for Windows file locks"""
    if not path.exists():
        return
    
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            print(f"  ✓ Deleted old vector store")
            return
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"  ⚠️  File locked, retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(2)
            else:
                print(f"  ❌ Could not delete {path}")
                print(f"     Error: {e}")
                print(f"     Solution: Close any programs using the database, or manually delete:")
                print(f"     rmdir /s /q {path}")
                sys.exit(1)


# ─────────────────────────────────────────────
# 1. LOAD & CHUNK DOCUMENTS
# ─────────────────────────────────────────────
def load_documents():
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {RAW_DATA_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()
        chunks = splitter.split_documents(pages)

        # Stamp metadata on every chunk
        for chunk in chunks:
            chunk.metadata["source"] = pdf_path.name

        all_chunks.extend(chunks)
        print(f"  ✓ {pdf_path.name}: {len(pages)} pages → {len(chunks)} chunks")

    print(f"\n  📚 Total: {len(all_chunks)} chunks from {len(pdf_files)} file(s)")
    return all_chunks


# ─────────────────────────────────────────────
# 2. GET EMBEDDINGS (with fallback options)
# ─────────────────────────────────────────────
def get_embeddings():
    """Get Gemini embeddings"""
    print("  Using embedding model: models/gemini-embedding-001")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return embeddings


# ─────────────────────────────────────────────
# 3. BUILD OR LOAD VECTOR STORE
# ─────────────────────────────────────────────
def get_vectorstore(force_rebuild: bool = False):
    embeddings = get_embeddings()
    
    db_exists = CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir())

    if db_exists and not force_rebuild:
        print("📂 Loading existing vector store...")
        try:
            vs = Chroma(
                collection_name=COLLECTION,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DB_DIR)
            )
            count = vs._collection.count()
            print(f"  ✓ Loaded {count} vectors")

            # If the store is empty, rebuild automatically
            if count == 0:
                print("  ⚠️  Store is empty – rebuilding...")
                return get_vectorstore(force_rebuild=True)

            return vs
        except Exception as e:
            print(f"  ⚠️  Error loading vector store: {e}")
            print("  ⚠️  Will rebuild from scratch...")
            return get_vectorstore(force_rebuild=True)

    # Rebuild
    print("🔨 Building vector store from scratch...")
    safe_delete_chromadb(CHROMA_DB_DIR)
    
    chunks = load_documents()

    print(f"  🔄 Creating embeddings (this may take 1-2 minutes)...")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DB_DIR)
    )
    print(f"  ✓ Saved {len(chunks)} vectors to {CHROMA_DB_DIR}")
    return vs


# ─────────────────────────────────────────────
# 4. RAG CHAIN
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a helpful logistics assistant. Use the document excerpts below to answer the question.

Rules:
- Answer using ONLY the information in the excerpts.
- Quote exact numbers, names, and codes where possible.
- If the excerpts do not contain the answer, say exactly:
  "I could not find that information in the provided documents."
- Always end your answer with a "Sources:" line listing the filenames and pages used.

Document excerpts:
----------------
{context}
----------------
"""

def build_answer(vectorstore, question: str, verbose: bool = False) -> str:
    # Retrieve top-k relevant chunks
    docs = vectorstore.similarity_search(question, k=TOP_K)

    if not docs:
        return "⚠️  No relevant documents found. Is your vector store populated?"

    # Build context string
    context_parts = []
    for i, doc in enumerate(docs, 1):
        src  = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[Excerpt {i} | {src} | page {page}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    if verbose:
        print("\n" + "="*60)
        print("📄 RETRIEVED CHUNKS:")
        print("="*60)
        for i, doc in enumerate(docs, 1):
            print(f"\n[Chunk {i}] {doc.metadata.get('source')} | Page {doc.metadata.get('page')}")
            print("-" * 60)
            print(doc.page_content[:300] + "...")
        print("="*60 + "\n")

    # Call Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        convert_system_message_to_human=True
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("human", SYSTEM_PROMPT + "\n\nQuestion: {question}")
    ])
    
    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    return response.content


# ─────────────────────────────────────────────
# 5. INTERACTIVE CHAT LOOP
# ─────────────────────────────────────────────
def chat(vectorstore):
    print("\n" + "="*60)
    print("🤖  Logistics RAG Assistant (Powered by Gemini)")
    print("="*60)
    print("Commands:")
    print("  • Type a question to get an answer")
    print("  • Add '!verbose' to see retrieved document chunks")
    print("  • Type 'quit' to exit")
    print("="*60 + "\n")

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        verbose = "!verbose" in raw
        question = raw.replace("!verbose", "").strip()

        print("\n🤖 Thinking...\n")
        try:
            answer = build_answer(vectorstore, question, verbose=verbose)
            print(f"Assistant:\n{answer}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
        
        print("-" * 60 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not set in .env")
        print("   Add this to your .env file:")
        print("   GOOGLE_API_KEY=your_gemini_api_key_here")
        sys.exit(1)

    print("\n" + "="*60)
    print("🚀  LOGISTICS RAG SYSTEM (Gemini Edition)")
    print("="*60 + "\n")

    # Get vector store (will rebuild if needed)
    vectorstore = get_vectorstore(force_rebuild=False)
    
    # Start chat
    chat(vectorstore)


if __name__ == "__main__":
    main()