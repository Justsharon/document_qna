import streamlit as st
from ingestor import build_document_collection
from qa_engine import ask

st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide"
)

# Poppins font and styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

.stMarkdown, h1, h2, h3, p, label {
    font-family: 'Poppins', sans-serif !important;
}

.stApp {
    background-color: #0f0f1a;
}

.source-badge {
    display: inline-block;
    background-color: #1e3a5f;
    color: #7dd3fc;
    padding: 4px 12px;
    border-radius: 8px;
    font-size: 0.85rem;
    margin: 4px 4px 4px 0;
    font-family: Poppins, sans-serif;
}

.answer-box {
    background-color: #1e1e2e;
    border-left: 4px solid #22c55e;
    padding: 20px;
    border-radius: 8px;
    margin: 12px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("Document Q&A Assistant")
st.caption("Ask questions about your document library — answers grounded in source material with attribution")

# Build collection (cached)
@st.cache_resource
def load_collection():
    return build_document_collection("documents")

with st.spinner("Loading document library..."):
    collection = load_collection()

st.success(f"Knowledge base ready — {collection.count()} chunks indexed")

st.divider()

# Query input
query = st.text_input(
    "Ask a question about your documents",
    placeholder="What does Phase 1 of the roadmap focus on?"
)

if st.button("Ask", type="primary") and query:
    with st.spinner("Searching documents and generating answer..."):
        result = ask(collection, query)

    # Answer
    st.markdown("### Answer")
    st.markdown(
        f'<div class="answer-box">{result["answer"]}</div>',
        unsafe_allow_html=True
    )

    # Metrics
    col1, col2 = st.columns(2)
    col1.metric("Sources consulted", len(result["sources"]))
    col2.metric("Chunks used", result["chunks_used"])

    # Sources
    if result["sources"]:
        st.markdown("### Sources")
        sources_html = "".join([
            f'<span class="source-badge"> {s}</span>'
            for s in result["sources"]
        ])
        st.markdown(sources_html, unsafe_allow_html=True)
    else:
        st.warning("No sources were confident enough to answer this question.")

st.divider()

# Example queries
st.markdown("### Try these example questions")
examples = [
    "What is the main focus of the 2026 data career roadmap?",
    "What does Phase 1 focus on?",
    "Why is effect size important alongside p-value?",
    "What are the main AI integration tools mentioned?"
]

for ex in examples:
    if st.button(ex, key=ex):
        st.session_state["query"] = ex
        st.rerun()