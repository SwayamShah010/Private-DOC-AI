"""
PrivateDocs AI - Streamlit UI.

This file is intentionally "thin": every screen calls into the modules
built in Phases 2-7 (indexing_service, semantic_search, rag_pipeline,
summarizer) rather than containing pipeline logic itself. If you need
to change how indexing or answering works, change it in those modules -
not here.

Run with: streamlit run app/ui/streamlit_app.py
(or `python run.py`, which does the same thing - see run.py)
"""
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config.settings import settings
from app.embeddings.embedder import Embedder
from app.generation.ollama_client import OllamaClient
from app.generation.rag_pipeline import RAGPipeline
from app.generation.summarizer import Summarizer
from app.retrieval.keyword_search import KeywordSearch
from app.retrieval.reranker import Reranker
from app.retrieval.semantic_search import SemanticSearch
from app.storage.indexing_service import index_pdf_file
from app.storage.vector_store import VectorStore

st.set_page_config(page_title="PrivateDocs AI", page_icon="\U0001F4C4", layout="wide")


# --------------------------------------------------------------------------
# Cached resources - heavy objects (embedding model, vector DB connection)
# load once per session instead of on every rerun/interaction.
# --------------------------------------------------------------------------
@st.cache_resource
def get_vector_store() -> VectorStore:
    return VectorStore()


@st.cache_resource
def get_embedder() -> Embedder:
    return Embedder()


@st.cache_resource
def get_ollama_client() -> OllamaClient:
    return OllamaClient()


@st.cache_resource
def get_search():
    """Semantic-only or hybrid (semantic + BM25 keyword), per
    settings.enable_hybrid_search - same decision app/retrieval/factory.py
    makes, but built here so it reuses the cached Embedder/VectorStore
    instead of constructing fresh ones on every question (which the
    factory's defaults would do, since it doesn't know about Streamlit's
    caching)."""
    vector_store = get_vector_store()
    semantic = SemanticSearch(embedder=get_embedder(), vector_store=vector_store)
    if not settings.enable_hybrid_search:
        return semantic
    from app.retrieval.hybrid_search import HybridSearch
    return HybridSearch(semantic_search=semantic, keyword_search=KeywordSearch(vector_store=vector_store))


@st.cache_resource
def get_reranker() -> Reranker | None:
    if not settings.enable_reranking:
        return None
    return Reranker()


def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline(search=get_search(), llm=get_ollama_client(), reranker=get_reranker())


def get_summarizer() -> Summarizer:
    return Summarizer(vector_store=get_vector_store(), llm=get_ollama_client())


# --------------------------------------------------------------------------
# Sidebar - collection selector, document list, privacy status, settings
# --------------------------------------------------------------------------
def render_sidebar() -> str:
    st.sidebar.title("\U0001F4C4 PrivateDocs AI")
    st.sidebar.caption("Local-first document intelligence")

    # Privacy / model status indicator - PRD Section 13
    st.sidebar.divider()
    st.sidebar.subheader("Status")
    st.sidebar.success(f"\U0001F512 Privacy mode: **{settings.privacy_mode}**")
    ollama_ok = get_ollama_client().is_available()
    if ollama_ok:
        st.sidebar.success(f"\u2713 Ollama connected ({settings.ollama_model})")
    else:
        st.sidebar.error(
            f"\u2717 Can't reach Ollama at {settings.ollama_host}. "
            f"Run `ollama serve` and `ollama pull {settings.ollama_model}`."
        )

    # Collection selector
    st.sidebar.divider()
    st.sidebar.subheader("Collection")
    vector_store = get_vector_store()
    existing = [c for c in vector_store.list_collections()]
    options = existing if existing else ["default"]
    collection_id = st.sidebar.selectbox("Active collection", options=options, index=0)
    new_collection = st.sidebar.text_input("Or create a new collection", placeholder="e.g. research_papers")
    if new_collection:
        collection_id = new_collection.strip()

    # Document list for the active collection
    st.sidebar.divider()
    st.sidebar.subheader("Documents in this collection")
    docs = vector_store.list_documents(collection_id=collection_id)
    if not docs:
        st.sidebar.caption("No documents indexed yet.")
    else:
        for d in docs:
            col1, col2 = st.sidebar.columns([4, 1])
            col1.write(f"\U0001F4C4 {d['filename']} ({d['page_count']} pg)")
            if col2.button("\U0001F5D1\uFE0F", key=f"del_{d['document_id']}", help="Delete this document"):
                vector_store.delete_document(d["document_id"], collection_id=collection_id)
                st.rerun()

    # Advanced settings - PRD Section 13
    st.sidebar.divider()
    with st.sidebar.expander("\u2699\uFE0F Advanced settings"):
        st.caption(f"Chunk size: {settings.chunk_size} | Overlap: {settings.chunk_overlap}")
        st.caption(f"Top-k retrieval: {settings.top_k}")
        st.caption(f"Embedding model: {settings.embedding_model}")
        retrieval_mode = "Hybrid (semantic + keyword)" if settings.enable_hybrid_search else "Semantic only"
        st.caption(f"Retrieval mode: {retrieval_mode}")
        st.caption(f"Reranking: {'on' if settings.enable_reranking else 'off'}")
        st.caption(f"OCR fallback: {'on' if settings.enable_ocr else 'off'}")
        st.caption("Edit .env to change these values.")

    return collection_id


# --------------------------------------------------------------------------
# Tab: Upload & Indexing
# --------------------------------------------------------------------------
def render_upload_tab(collection_id: str):
    st.header("Upload & Index Documents")
    st.caption(f"Uploading into collection: **{collection_id}**")

    uploaded_files = st.file_uploader(
        "Drag and drop PDF files here", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Index uploaded files", type="primary"):
        embedder = get_embedder()
        vector_store = get_vector_store()
        progress = st.progress(0.0, text="Starting...")

        for i, uploaded_file in enumerate(uploaded_files):
            progress.progress(i / len(uploaded_files), text=f"Indexing {uploaded_file.name}...")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            result = index_pdf_file(
                tmp_path,
                filename=uploaded_file.name,
                collection_id=collection_id,
                embedder=embedder,
                vector_store=vector_store,
            )
            Path(tmp_path).unlink(missing_ok=True)  # clean up temp file (PRD Section 12)

            if result.success:
                if result.ocr_pages_used:
                    st.success(
                        f"\u2713 {result.filename}: indexed {result.chunk_count} chunks "
                        f"({result.ocr_pages_used} page(s) recovered via OCR)."
                    )
                else:
                    st.success(f"\u2713 {result.filename}: indexed {result.chunk_count} chunks.")
            else:
                st.error(f"\u2717 {result.filename}: {result.error}")

        progress.progress(1.0, text="Done.")
        st.rerun()


# --------------------------------------------------------------------------
# Tab: Ask Documents (RAG chat)
# --------------------------------------------------------------------------
def render_chat_tab(collection_id: str):
    st.header("Ask Your Documents")

    history_key = f"chat_history_{collection_id}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    for turn in st.session_state[history_key]:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn["citations"]:
                with st.expander(f"\U0001F4CE Sources ({len(turn['citations'])})"):
                    for c in turn["citations"]:
                        st.markdown(f"**{c.filename}, p. {c.page_number}**")
                        st.caption(c.chunk_text)

    query = st.chat_input("Ask a question about your documents...")
    if query:
        st.session_state[history_key].append({"query": query, "answer": "...", "citations": []})
        with st.spinner("Searching and generating answer..."):
            try:
                pipeline = get_rag_pipeline()
                result = pipeline.answer(query, collection_id=collection_id)
                st.session_state[history_key][-1] = {
                    "query": query,
                    "answer": result.answer_text,
                    "citations": result.citations,
                }
            except Exception as exc:  # noqa: BLE001
                st.session_state[history_key][-1] = {
                    "query": query,
                    "answer": f"Something went wrong: {exc}",
                    "citations": [],
                }
        st.rerun()


# --------------------------------------------------------------------------
# Tab: Summarize
# --------------------------------------------------------------------------
def render_summary_tab(collection_id: str):
    st.header("Summarize")
    vector_store = get_vector_store()
    docs = vector_store.list_documents(collection_id=collection_id)

    if not docs:
        st.info("No documents indexed in this collection yet.")
        return

    mode = st.radio("Summarize", ["A single document", "The whole collection"], horizontal=True)

    if mode == "A single document":
        filenames = {d["filename"]: d["document_id"] for d in docs}
        choice = st.selectbox("Choose a document", options=list(filenames.keys()))
        if st.button("Generate summary"):
            with st.spinner(f"Summarizing {choice}..."):
                summarizer = get_summarizer()
                result = summarizer.summarize_document(filenames[choice], collection_id=collection_id)
            st.markdown(result.summary_text)
            if result.truncated:
                st.warning(
                    "This document was long enough that only part of it was summarized. "
                    "Full-length map-reduce summarization is planned as a future improvement."
                )
    else:
        if st.button("Generate collection summary"):
            with st.spinner(f"Summarizing all {len(docs)} documents..."):
                summarizer = get_summarizer()
                result = summarizer.summarize_collection(collection_id=collection_id)
            st.markdown(result.summary_text)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    collection_id = render_sidebar()

    tab_chat, tab_upload, tab_summary = st.tabs(["\U0001F4AC Ask Documents", "\U0001F4E4 Upload & Index", "\U0001F4DD Summarize"])
    with tab_chat:
        render_chat_tab(collection_id)
    with tab_upload:
        render_upload_tab(collection_id)
    with tab_summary:
        render_summary_tab(collection_id)


if __name__ == "__main__":
    main()
