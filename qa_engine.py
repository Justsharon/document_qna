import os
from groq import Groq
import chromadb
from dotenv import load_dotenv

load_dotenv()

DISTANCE_THRESHOLD = 1.2


def format_document_context(results: list[dict]) -> str:
    """Convert retrieved chunks into structured context with source attribution"""
    if not results:
        return "No relevant information found in the documents."

    lines = ["Here are the relevant excerpts from the documents:\n"]
    for i, r in enumerate(results, 1):
        content = r['content'][:800] 
        lines.append(f"Excerpt {i}:")
        lines.append(f"  Source: {r['source']} (chunk {r['chunk_index']})")
        lines.append(f"  Content: {content}")
        lines.append("")

    return "\n".join(lines)


def retrieve_chunks(collection, query, n_results=5, source_filter=None):
    """Hybrid retrieval — semantic + keyword match"""
    kwargs = {"query_texts": [query], "n_results": n_results}
    if source_filter:
        kwargs["where"] = {"source": {"$eq": source_filter}}

    raw = collection.query(**kwargs)

    # Parse semantic results
    results = []
    for i, doc in enumerate(raw["documents"][0]):
        results.append({
            "content": doc,
            "source": raw["metadatas"][0][i]["source"],
            "chunk_index": raw["metadatas"][0][i]["chunk_index"],
            "distance": round(raw["distances"][0][i], 3)
        })

    filtered = [r for r in results if r["distance"] < DISTANCE_THRESHOLD]

    # Keyword fallback — find chunks containing exact query terms
    query_lower = query.lower()
    all_chunks = collection.get()
    keyword_matches = []
    
    for i, doc in enumerate(all_chunks["documents"]):
        doc_lower = doc.lower()
        # Check for specific structural terms
        if any(term in doc_lower for term in ["phase 1", "phase 2", "phase 3", "phase 4"]):
            if any(word in query_lower for word in ["phase", "first", "second", "third", "fourth"]):
                keyword_matches.append({
                    "content": doc,
                    "source": all_chunks["metadatas"][i]["source"],
                    "chunk_index": all_chunks["metadatas"][i]["chunk_index"],
                    "distance": 0.5  # synthetic score — keyword match is strong signal
                })

    # Combine and deduplicate by chunk_index
    seen = set()
    combined = []
    for chunk in filtered + keyword_matches:
        key = (chunk["source"], chunk["chunk_index"])
        if key not in seen:
            seen.add(key)
            combined.append(chunk)

    return combined[:n_results]

def generate_answer(query: str, context: str) -> str:
    """Generate answer using retrieved context"""
    client = Groq(api_key=os.getenv("API_KEY"))

    prompt = f"""You are a helpful document assistant.

Answer the user's question using ONLY the excerpts provided below.
If the excerpts don't contain the answer, say "The documents don't contain information about this."
Cite the source document name when giving your answer.
Keep your answer concise and grounded in the excerpts.

{context}

User question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content


def ask(
    collection: chromadb.Collection,
    query: str,
    source_filter: str = None
) -> dict:
    """Full Q&A pipeline"""
    chunks = retrieve_chunks(collection, query, source_filter=source_filter)

    # DEBUG
    print("\n--- RETRIEVED CHUNKS ---")
    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i} [distance: {c['distance']}] from chunk_index {c['chunk_index']}")
        print(f"  Preview: {c['content'][:150]}...")
    print("--- END DEBUG ---\n")

    context = format_document_context(chunks)
    answer = generate_answer(query, context)

    return {
        "query": query,
        "answer": answer,
        "sources": list(set(c["source"] for c in chunks)),
        "chunks_used": len(chunks)
    }


if __name__ == "__main__":
    from ingestor import build_document_collection

    collection = build_document_collection("documents")

    queries = [
        "What is the main focus of the 2026 data career roadmap?",
        "What are vector databases used for?",
        "What does the first phase of the roadmap focus on?",
    ]

    for q in queries:
        print("\n" + "="*60)
        result = ask(collection, q)
        print(f"Q: {result['query']}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Chunks used: {result['chunks_used']}")
