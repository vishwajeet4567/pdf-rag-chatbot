"""
PDF RAG Chatbot — powered by Endee Vector Database + Google Gemini
Author: Vishwajeet Siranje
Description:
    Upload any PDF → the app chunks it, embeds it with sentence-transformers,
    stores the vectors in Endee (local), then answers your questions using
    retrieved context passed to Google Gemini.
"""

import os
import uuid
import hashlib
import streamlit as st
import pdfplumber
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
ENDEE_URL       = os.getenv("ENDEE_URL", "http://localhost:8080/api/v1")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
INDEX_NAME      = "pdf_rag_index"
EMBED_MODEL     = "all-MiniLM-L6-v2"   # 384-dim, fast & accurate
EMBED_DIM       = 384
CHUNK_SIZE      = 400   # characters per chunk
CHUNK_OVERLAP   = 80    # overlap to preserve context across boundaries
TOP_K           = 5     # number of chunks to retrieve per query

# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Chatbot — Endee",
    page_icon="📄",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }

    .hero-title {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .hero-sub {
        text-align: center; color: #94a3b8; font-size: 1rem; margin-bottom: 2rem;
    }
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em;
        background: rgba(167,139,250,0.15); color: #a78bfa;
        border: 1px solid rgba(167,139,250,0.3); margin: 2px;
    }
    .source-box {
        background: rgba(255,255,255,0.05); border-left: 3px solid #60a5fa;
        border-radius: 6px; padding: 10px 14px; margin: 6px 0;
        font-size: 0.82rem; color: #cbd5e1;
    }
    .status-ok  { color: #34d399; font-weight: 600; }
    .status-err { color: #f87171; font-weight: 600; }
    .info-card {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 16px; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">📄 PDF RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Powered by '
    '<span class="badge">Endee Vector DB</span>'
    '<span class="badge">Sentence Transformers</span>'
    '<span class="badge">Google Gemini</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
if "embed_model" not in st.session_state:
    st.session_state.embed_model = None
if "endee_index" not in st.session_state:
    st.session_state.endee_index = None

# ──────────────────────────────────────────────────────────────
# Helpers — Embedding
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL)

def embed_texts(model, texts: list[str]) -> list[list[float]]:
    """Return normalized float32 embeddings as plain Python lists."""
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vecs.tolist()

# ──────────────────────────────────────────────────────────────
# Helpers — Endee
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Endee…")
def get_endee_index():
    """Connect to Endee server, create index if absent, return Index object."""
    client = Endee()
    client.set_base_url(ENDEE_URL)

    # Create index only if it doesn't exist yet
    existing = [idx.name for idx in client.list_indexes()]
    if INDEX_NAME not in existing:
        client.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            space_type="cosine",
            precision=Precision.INT8,   # INT8 quantisation for speed
        )
    return client.get_index(name=INDEX_NAME)

# ──────────────────────────────────────────────────────────────
# Helpers — PDF Processing
# ──────────────────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file) -> list[dict]:
    """Extract text page-by-page from a PDF, return list of {page, text}."""
    pages = []
    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i + 1, "text": text})
    return pages

def chunk_text(text: str, source: str, page: int) -> list[dict]:
    """Split text into overlapping chunks; return list of chunk dicts."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({
                "id": hashlib.md5(f"{source}_{page}_{start}".encode()).hexdigest(),
                "text": chunk,
                "source": source,
                "page": page,
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# ──────────────────────────────────────────────────────────────
# Helpers — Gemini
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_gemini_model(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def generate_answer(model, question: str, context_chunks: list[dict]) -> str:
    """Build a RAG prompt and call Gemini to generate an answer."""
    context = "\n\n".join(
        f"[Source: {c['meta'].get('source','?')}, Page {c['meta'].get('page','?')}]\n{c['meta'].get('text','')}"
        for c in context_chunks
    )
    prompt = f"""You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I couldn't find that in the uploaded document."

Context:
{context}

Question: {question}

Answer:"""
    response = model.generate_content(prompt)
    return response.text.strip()

# ──────────────────────────────────────────────────────────────
# Sidebar — Configuration & PDF Upload
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input(
        "Gemini API Key",
        value=GEMINI_API_KEY,
        type="password",
        help="Get a free key at ai.google.dev",
        placeholder="AIza…",
    )

    endee_url = st.text_input(
        "Endee Server URL",
        value=ENDEE_URL,
        help="Default: http://localhost:8080/api/v1",
    )

    st.divider()
    st.markdown("## 📂 Upload PDF")
    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("📥 Index Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            embed_model = load_embed_model()
            try:
                client = Endee()
                client.set_base_url(endee_url)

                # Create or retrieve index
                existing = [idx.name for idx in client.list_indexes()]
                if INDEX_NAME not in existing:
                    client.create_index(
                        name=INDEX_NAME,
                        dimension=EMBED_DIM,
                        space_type="cosine",
                        precision=Precision.INT8,
                    )
                index = client.get_index(name=INDEX_NAME)

                for uf in uploaded_files:
                    file_key = uf.name
                    if file_key in st.session_state.indexed_files:
                        st.info(f"✅ `{uf.name}` already indexed — skipping.")
                        continue

                    with st.spinner(f"Indexing `{uf.name}`…"):
                        pages = extract_text_from_pdf(uf)
                        if not pages:
                            st.error(f"No extractable text in `{uf.name}`.")
                            continue

                        all_chunks = []
                        for p in pages:
                            all_chunks.extend(chunk_text(p["text"], uf.name, p["page"]))

                        # Batch upsert into Endee
                        BATCH = 64
                        for i in range(0, len(all_chunks), BATCH):
                            batch = all_chunks[i:i + BATCH]
                            texts  = [c["text"] for c in batch]
                            vectors = embed_texts(embed_model, texts)
                            items = [
                                {
                                    "id": c["id"],
                                    "vector": vectors[j],
                                    "meta": {
                                        "text":   c["text"],
                                        "source": c["source"],
                                        "page":   c["page"],
                                    },
                                }
                                for j, c in enumerate(batch)
                            ]
                            index.upsert(items)

                        st.session_state.indexed_files.add(file_key)
                        st.success(f"✅ Indexed `{uf.name}` — {len(all_chunks)} chunks stored in Endee.")

                # Cache index reference
                st.session_state.endee_index = index
                st.session_state.endee_url   = endee_url

            except Exception as e:
                st.error(f"❌ Endee error: {e}\n\nMake sure Endee is running (`docker compose up -d`).")

    st.divider()

    # Status indicators
    st.markdown("### 📊 Status")
    n_docs = len(st.session_state.indexed_files)
    if n_docs:
        st.markdown(f'<p class="status-ok">● {n_docs} document(s) indexed</p>', unsafe_allow_html=True)
        for f in st.session_state.indexed_files:
            st.markdown(f"&nbsp;&nbsp;📄 `{f}`")
    else:
        st.markdown('<p class="status-err">● No documents indexed yet</p>', unsafe_allow_html=True)

    if api_key:
        st.markdown('<p class="status-ok">● Gemini API key set</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-err">● No Gemini API key</p>', unsafe_allow_html=True)

    st.divider()
    st.markdown(
        '<div style="font-size:0.75rem;color:#64748b;">'
        'Built with <b>Endee</b> vector database<br>'
        'For Endee.io — Tap Academy Hiring Challenge'
        '</div>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────
# Main Chat Area
# ──────────────────────────────────────────────────────────────
col_left, col_main, col_right = st.columns([1, 6, 1])

with col_main:
    if not st.session_state.indexed_files:
        st.markdown("""
        <div class="info-card" style="text-align:center; padding:40px;">
            <div style="font-size:3rem;">📤</div>
            <h3 style="color:#e2e8f0;">Get Started</h3>
            <p style="color:#94a3b8;">
                1. Add your <b>Gemini API key</b> in the sidebar<br>
                2. Upload one or more <b>PDF files</b><br>
                3. Click <b>Index Documents</b><br>
                4. Start asking questions below!
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📎 Source Chunks", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="source-box">'
                            f'<b>📄 {src["source"]}</b> — Page {src["page"]}<br>'
                            f'<em>{src["text"][:220]}…</em>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # Chat input
    if question := st.chat_input("Ask anything about your PDF…", disabled=not st.session_state.indexed_files):
        if not api_key:
            st.error("⚠️ Please enter your Gemini API key in the sidebar.")
            st.stop()

        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching Endee & generating answer…"):
                try:
                    embed_model = load_embed_model()

                    # Connect to Endee and retrieve index
                    client = Endee()
                    client.set_base_url(endee_url if endee_url else ENDEE_URL)
                    index = client.get_index(name=INDEX_NAME)

                    # Embed query and search
                    q_vec = embed_texts(embed_model, [question])[0]
                    results = index.query(vector=q_vec, top_k=TOP_K)

                    # Build context from results
                    context_chunks = []
                    for r in results:
                        context_chunks.append({
                            "id": r.id,
                            "similarity": r.similarity,
                            "meta": r.meta or {},
                        })

                    # Generate answer with Gemini
                    gemini = get_gemini_model(api_key)
                    answer = generate_answer(gemini, question, context_chunks)

                    st.markdown(answer)

                    # Save to history
                    sources = [
                        {
                            "source": c["meta"].get("source", "Unknown"),
                            "page":   c["meta"].get("page", "?"),
                            "text":   c["meta"].get("text", ""),
                        }
                        for c in context_chunks
                    ]
                    st.session_state.chat_history.append({
                        "role":    "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                    # Show source citations inline
                    with st.expander("📎 Source Chunks", expanded=False):
                        for src in sources:
                            st.markdown(
                                f'<div class="source-box">'
                                f'<b>📄 {src["source"]}</b> — Page {src["page"]}<br>'
                                f'<em>{src["text"][:220]}…</em>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                except Exception as e:
                    err_msg = f"❌ Error: {e}"
                    st.error(err_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
