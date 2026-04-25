## Document Q&A Assistant


### Project Summary
I built a Document Q&A Assistant that ingests a PDF library, chunks documents with overlap to preserve semantic continuity, stores chunks in ChromaDB with source metadata, and answers natural language questions using a RAG pipeline with Groq.
During development I discovered real production failure modes — pure semantic search misses structural queries like 'Phase 1', retrieval quality depends more on chunking strategy than embedding model choice, and token limits force context prioritisation. I implemented a hybrid retrieval approach combining semantic search with keyword fallback to fix the structural query problem, and I learned the importance of verifying retrieval output separately from AI answer quality.
The system includes source attribution for every answer, confidence thresholds to prevent hallucinations, and a Streamlit UI. It's deployed-ready with proper environment variable handling for API keys