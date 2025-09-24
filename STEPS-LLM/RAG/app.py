import os
import io
import json
import time
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# =========================
# Configuration / Constants
# =========================
SYSTEM_PROMPT = """You are a precise assistant. Answer the user's question using ONLY the provided context.
If the answer is not in the context, say you do not have enough information.
Cite sources in square brackets like [1], [2] that match the provided context items.
Be concise and factual."""
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# ==============
# Misc Utilities
# ==============
def read_pdf(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    text = []
    for page in reader.pages:
        try:
            text.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(text)

def read_txt(file: io.BytesIO) -> str:
    return file.read().decode(errors="ignore")

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = " ".join(text.split())  # normalize whitespace
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text): break
        start = max(0, end - overlap)
    return chunks

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def build_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product; with L2-normalized vectors ~ cosine sim
    index.add(embeddings.astype(np.float32))
    return index

def embed_chunks(model, chunks: List[str], batch: int = 64) -> np.ndarray:
    vecs = []
    for i in range(0, len(chunks), batch):
        vecs.extend(model.encode(chunks[i:i+batch], normalize_embeddings=False))
    return l2_normalize(np.array(vecs, dtype=np.float32))

def embed_query(model, q: str) -> np.ndarray:
    v = model.encode([q], normalize_embeddings=False)
    return l2_normalize(np.array(v, dtype=np.float32))

def format_context(hits: List[Dict[str, Any]], char_budget: int = 2400) -> Tuple[str, List[Dict[str, Any]]]:
    chosen, total = [], 0
    for h in hits:
        t = h["text"]
        if total + len(t) + 100 > char_budget:
            break
        chosen.append(h); total += len(t)
    ctx = []
    for i, h in enumerate(chosen, 1):
        ctx.append(f"[{i}] Source: {h['source']} (chunk {h['chunk_id']})\n{h['text']}")
    return "\n\n".join(ctx), chosen

def make_prompt(context: str, question: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    # keep a bounded recent history to control token growth
    recent = history[-12:]  # up to 6 user+assistant turns
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(recent)
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nAnswer the last user question using ONLY the context above."
    })
    return messages


def call_openai(messages: list[dict], model_name: str) -> str:
    # Reads OPENAI_API_KEY from environment automatically
    client = OpenAI()
    resp = client.chat.completions.create( model=model_name, messages=messages, temperature=0.0)
    return resp.choices[0].message.content.strip()

# =================
# Streamlit UI App
# =================
st.set_page_config(page_title="RAG with Streamlit (OpenAI)", page_icon="📚", layout="wide")
st.title("📚 RAG Demo (Streamlit, OpenAI)")

# -------------
# Sidebar
# -------------
with st.sidebar:
    st.header("Ingest documents")
    uploaded_files = st.file_uploader( "Upload PDFs / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True )

    st.divider()
    st.subheader("Chunking & Retrieval")
    chunk_size = st.slider("Chunk size (chars)", 300, 1200, 800, step=50)
    overlap = st.slider("Chunk overlap (chars)", 0, 300, 120, step=10)
    top_k = st.slider("Top-K retrieval", 1, 10, 4, step=1)

    st.divider()
    st.subheader("LLM")
    openai_model = st.text_input("OpenAI model", value=DEFAULT_OPENAI_MODEL)
    st.caption("Set environment variable OPENAI_API_KEY before running.")

    st.divider()
    persist = st.checkbox("Persist index on disk", value=False, help="Saves/loads FAISS index and docs.json")
    build = st.button("Build / Rebuild Index")

# ---------------------
# Session State Set-Up
# ---------------------
if "embedder" not in st.session_state:
    st.session_state.embedder = SentenceTransformer("all-MiniLM-L6-v2")  # local embeddings
if "index" not in st.session_state:
    st.session_state.index = None
if "doc_vectors" not in st.session_state:
    st.session_state.doc_vectors = None
if "docs" not in st.session_state:
    st.session_state.docs = []  # [{source, chunk_id, text}]
if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history for this session/tab

# -----------------
# Load persistence
# -----------------
if persist and st.session_state.index is None:
    try:
        if os.path.exists("index.faiss") and os.path.exists("docs.json"):
            st.session_state.index = faiss.read_index("index.faiss")
            with open("docs.json", "r") as f:
                st.session_state.docs = json.load(f)
            # The vector store is not needed once index is saved/loaded
            st.success(f"Loaded persisted index with {len(st.session_state.docs)} chunks.")
    except Exception as e:
        st.warning(f"Could not load persisted index: {e}")

# ------------
# Build index
# ------------
if build:
    texts, sources = [], []
    for f in uploaded_files or []:
        name = f.name
        if name.lower().endswith(".pdf"):
            content = read_pdf(f)
        else:
            content = read_txt(f)
        chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
        for i, c in enumerate(chunks):
            texts.append(c)
            sources.append({"source": name, "chunk_id": i, "text": c})

    if not texts:
        st.warning("Please upload at least one file.")
    else:
        with st.spinner("Embedding & indexing..."):
            embs = embed_chunks(st.session_state.embedder, texts)
            index = build_index(embs)
            st.session_state.index = index
            st.session_state.docs = sources
            st.session_state.doc_vectors = embs

            if persist:
                try:
                    faiss.write_index(index, "index.faiss")
                    with open("docs.json", "w") as f:
                        json.dump(st.session_state.docs, f)
                except Exception as e:
                    st.warning(f"Could not persist index: {e}")

        st.success(f"Indexed {len(texts)} chunks from {len(uploaded_files)} files.")

# ---------------
# Chat interface
# ---------------
st.subheader("Ask questions (Chat)")
st.caption("This chat remembers messages in this browser session/tab.")

# Render previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input("Ask about your documents…")
if user_msg:
    # Save user turn
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # Guard: ensure index is ready
    if st.session_state.index is None or st.session_state.docs == []:
        with st.chat_message("assistant"):
            st.warning("Please upload files and click **Build / Rebuild Index** first.")
        st.stop()

    # Retrieve
    with st.spinner("Retrieving…"):
        qv = embed_query(st.session_state.embedder, user_msg)
        scores, idxs = st.session_state.index.search( qv.astype(np.float32), k=min(top_k, len(st.session_state.docs)) )
        idxs = [i for i in idxs[0] if i != -1]
        hits = []
        for j, i in enumerate(idxs):
            meta = st.session_state.docs[i].copy()
            meta["score"] = float(scores[0][j])
            hits.append(meta)
        context_str, chosen = format_context(hits)

    # Construct LLM messages (history-aware + retrieved context)
    with st.spinner("Generating…"):
        messages_for_llm = make_prompt(context_str, user_msg, st.session_state.messages)
        answer = call_openai(messages_for_llm, model_name=openai_model)

    # Show assistant response + sources
    with st.chat_message("assistant"):
        st.markdown(answer)
        if chosen:
            st.markdown("**Sources**")
            for i, h in enumerate(chosen, 1):
                with st.expander(f"[{i}] {h['source']} (chunk {h['chunk_id']})"):
                    st.write(h["text"])
        else:
            st.info("No sources retrieved.")

    # Save assistant turn
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Optional: persist chat (overwrite single file)
    if persist:
        try:
            with open("last_chat.json", "w") as f:
                json.dump(st.session_state.messages, f)
        except Exception:
            pass