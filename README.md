<p align="center">
  <img src="docs/assets/logo-dark.svg" height="80" alt="Endee" />
</p>

<h1 align="center">PDF RAG Chatbot — powered by Endee Vector Database</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Vector_DB-Endee-a78bfa?style=flat-square" />
  <img src="https://img.shields.io/badge/Embeddings-sentence--transformers-60a5fa?style=flat-square" />
  <img src="https://img.shields.io/badge/LLM-Google_Gemini-34d399?style=flat-square" />
  <img src="https://img.shields.io/badge/UI-Streamlit-ff4b4b?style=flat-square" />
  <img src="https://img.shields.io/badge/License-Apache_2.0-yellow?style=flat-square" />
</p>

---

## 📌 Project Overview

This project is a **Retrieval-Augmented Generation (RAG) chatbot** that lets you upload any PDF document and ask natural-language questions about it. It uses:

| Component | Technology |
|-----------|-----------|
| Vector Database | **Endee** (local, via Docker) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim, INT8 quantised) |
| LLM / Answer Generation | **Google Gemini 1.5 Flash** (free tier) |
| UI | **Streamlit** |

**Key highlights:**
- Uses the official **Endee Python SDK** (`pip install endee`) for `create_index`, `upsert`, and `query`
- INT8 quantisation via `Precision.INT8` for fast, memory-efficient retrieval
- Cosine similarity search with `top_k=5` chunks per query
- Source citations shown for every answer (file name + page number)
- No OpenAI dependency — runs fully on free APIs

---

## 🏗️ System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit Frontend                          │
│   Upload PDF │  Ask Question  │  View Answer + Source Citations  │
└──────┬───────────────┬────────────────────────────┬─────────────┘
       │               │                            │
  [Indexing Flow]  [Query Flow]              [Answer Display]
       │               │                            │
       ▼               ▼                            │
 pdfplumber        SentenceTransformer         Google Gemini
 (extract text)    (embed question)            (generate answer)
       │               │                            ▲
       ▼               ▼                            │
 Chunker           ┌───────────────────────┐        │
 (400 chars,       │   Endee Vector DB     │────────┘
  80 overlap)      │  (Docker, port 8080)  │  top-5 chunks + metadata
       │           │                       │
       │ upsert    │  Index: pdf_rag_index │
       └──────────▶│  Space:  cosine       │
  vectors +        │  Dim:    384          │
  metadata         │  Prec:   INT8         │
  {text, source,   └───────────────────────┘
   page}
```

### Data Flow — Indexing
1. User uploads PDF → `pdfplumber` extracts text page-by-page
2. Text is split into 400-char chunks (80-char overlap)
3. Each chunk is embedded with `all-MiniLM-L6-v2`
4. Vectors + metadata (`text`, `source`, `page`) are upserted into Endee in batches of 64

### Data Flow — Querying
1. User asks a question → embedded with the same model
2. `index.query(vector=q_vec, top_k=5)` retrieves the 5 most similar chunks from Endee
3. Chunks are injected into a RAG prompt sent to **Google Gemini 1.5 Flash**
4. Answer is displayed in the chat UI with expandable source citations

---

## ⚙️ How Endee is Used

```python
from endee import Endee, Precision

# Connect to local Endee server
client = Endee()
client.set_base_url("http://localhost:8080/api/v1")

# Create a cosine-similarity index with INT8 quantisation
client.create_index(
    name="pdf_rag_index",
    dimension=384,
    space_type="cosine",
    precision=Precision.INT8,       # INT8 for fast, memory-efficient retrieval
)

# Get the index handle
index = client.get_index(name="pdf_rag_index")

# Upsert document chunks with metadata
index.upsert([
    {
        "id": "chunk_001",
        "vector": [...],            # 384-dim embedding
        "meta": {"text": "...", "source": "doc.pdf", "page": 1}
    }
])

# Retrieve top-5 similar chunks for a query
results = index.query(vector=[...], top_k=5)
for r in results:
    print(r.id, r.similarity, r.meta)
```

---

## 🚀 Setup Instructions

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Python 3.10+
- Free Gemini API key — get one at [ai.google.dev](https://ai.google.dev)

### Step 1 — Start Endee Locally

```bash
# From the repo root
docker compose up -d
```

Verify Endee is running:
```bash
curl http://localhost:8080/health
# Expected: {"status":"ok"}
```

### Step 2 — Install Python Dependencies

```bash
pip install -r app/requirements.txt
```

### Step 3 — Set Your Gemini API Key

```bash
export GEMINI_API_KEY="AIza..."
```

Or enter it directly in the sidebar when the app opens.

### Step 4 — Run the App

```bash
streamlit run app/rag_chatbot.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Step 5 — Use the Chatbot

1. Paste your Gemini API key in the sidebar (if not set via env)
2. Upload one or more PDF files
3. Click **Index Documents** — chunks are embedded and stored in Endee
4. Type a question in the chat box
5. The app retrieves the top-5 relevant chunks from Endee and generates an answer with Gemini

---

## 📁 Project Structure

```
pdf-rag-chatbot/
├── app/
│   ├── rag_chatbot.py      ← Main Streamlit app (RAG logic + UI)
│   └── requirements.txt    ← Python dependencies
├── docker-compose.yml      ← Runs Endee locally (port 8080)
├── docs/                   ← Endee documentation
└── README.md               ← This file
```

---

## 🌟 Features

- ✅ **Multi-PDF support** — index multiple PDFs and ask across all of them
- ✅ **Source citations** — every answer shows the source file and page number
- ✅ **INT8 quantisation** — uses Endee's `Precision.INT8` for efficient storage
- ✅ **Overlapping chunks** — 80-char overlap preserves context at chunk boundaries
- ✅ **Persistent index** — Endee persists data across restarts via Docker volumes
- ✅ **No OpenAI needed** — uses free Google Gemini API

---

## 📝 License

Apache License 2.0 — see [LICENSE](./LICENSE)

---

