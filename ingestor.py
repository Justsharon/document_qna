import os
import pypdf
import chromadb
from pathlib import Path

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF"""
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks 

def ingest_document(pdf_path: str, collection: chromadb.Collection) -> dict:
    """
    Full ingestion pipeline for one PDF:
    1. Extract text
    2. Chunk text
    3. Store chunks in ChromaDB with metadata
    Returns summary dict
    """
    filename = Path(pdf_path).stem  # filename without extension

    # Step 1 — extract
    text = extract_text_from_pdf(pdf_path)

    # Step 2 — chunk
    chunks = chunk_text(text)

    # Step 3 — store in ChromaDB
    # Each chunk needs:
    # - unique id: f"{filename}_chunk_{i}"
    # - document: the chunk text
    # - metadata: {"source": filename, "chunk_index": i, "total_chunks": len(chunks)}
    # YOUR CODE HERE
    collection.add(
        ids=[f"{filename}_chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[
            {"source": filename, "chunk_index": i, "total_chunks": len(chunks)}
            for i in range(len(chunks))
        ]
    )

    return {
        "filename": filename,
        "total_chunks": len(chunks),
        "total_words": len(text.split())
    }

def build_document_collection(docs_folder: str) -> chromadb.Collection:
    """
    Ingest all PDFs in a folder into one ChromaDB collection.
    """
    client = chromadb.Client()
    collection = client.create_collection(name="documents")

    pdf_files = list(Path(docs_folder).glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in documents folder")
        return collection

    for pdf_path in pdf_files:
        print(f"Ingesting: {pdf_path.name}")
        summary = ingest_document(str(pdf_path), collection)
        print(f"  → {summary['total_chunks']} chunks from {summary['total_words']} words")

    return collection

if __name__ == "__main__":
    collection = build_document_collection("documents")
    print(f"\nCollection ready: {collection.count()} total chunks")