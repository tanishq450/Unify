# 🏦 Unify — Intelligent Finance RAG

> A multi-strategy Retrieval-Augmented Generation system for financial documents with built-in hallucination prevention.

Unify routes financial queries to the optimal retrieval engine — hybrid vector search, knowledge graphs, or table extraction — then verifies every claim in the generated answer before returning it to the user.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **🧠 Intent-Based Routing** | Classifies queries into 6 categories and routes to the optimal RAG strategy |
| **🔍 Hybrid Retrieval** | BM25 keyword + dense semantic search with Qdrant RRF fusion |
| **📊 Table-Aware RAG** | Extracts and retrieves structured tables from financial PDFs |
| **🕸️ Graph RAG** | Neo4j knowledge graph for entity relationships and multi-hop reasoning |
| **🔄 Cross-Encoder Reranking** | Precision reranking with `bge-reranker-v2-m3` |
| **🎯 MMR Diversity** | Maximum Marginal Relevance prevents redundant results |
| **✅ Hallucination Verification** | FinGround-inspired atomic claim decomposition and verification |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌──────────────────────────┐
│   Intent Classifier       │  6 intents → 3 strategies
│   (Rule-based + Adaptive) │
└──────────┬───────────────┘
           │
    ┌──────┼──────────┐
    ▼      ▼          ▼
┌───────┐ ┌────────┐ ┌──────────┐
│Simple │ │ Graph  │ │  Table   │
│  RAG  │ │  RAG   │ │   RAG    │
│       │ │        │ │          │
│Qdrant │ │ Neo4j  │ │pdfplumber│
│Hybrid │ │ Cypher │ │LlamaParse│
│Search │ │   QA   │ │ GPT-4V   │
└───┬───┘ └───┬────┘ └────┬─────┘
    └─────────┴────────────┘
              │
              ▼
    ┌──────────────────┐
    │  LLM Generation   │  Draft answer from context
    └────────┬─────────┘
             ▼
    ┌──────────────────┐
    │  FinGround        │  Decompose → Verify → Regenerate
    │  Verifier         │  (strips unverified claims)
    └────────┬─────────┘
             ▼
      Verified Answer
      + Confidence Score
      + Claim Audit Trail
```

### Query Routing Strategy

| Intent | Example Query | RAG Strategy |
|---|---|---|
| `GENERAL_KNOWLEDGE` | "What does Apple do?" | Simple RAG |
| `NUMERICAL_TABLE` | "Show iPhone revenue by quarter" | Multimodal RAG |
| `COMPARISON` | "Apple vs Microsoft market cap" | Multimodal RAG |
| `TREND` | "Revenue growth trend 2024" | Multimodal RAG |
| `RELATIONSHIP` | "Who are Apple's suppliers?" | Graph RAG |
| `ENTITY` | "Who is Apple's CEO?" | Graph RAG |

---

## 📁 Project Structure

```
Hackathon_Project/
├── main.py                          # CLI entry point & orchestrator
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
│
├── Model_loader/
│   ├── llm.py                       # LLM & embedding model loader (MeshAPI)
│   └── embedding_model.py           # Raw OpenAI client for verifier
│
├── implementations/
│   ├── intent_classifier.py         # 6-category query router
│   ├── Rag.py                       # Hybrid RAG pipeline (ingest + query)
│   ├── hybrid_retriever.py          # Search engine: reranking + MMR
│   ├── Graph_rag.py                 # Neo4j knowledge graph RAG
│   ├── hallucination_verifier.py    # FinGround claim verification
│   └── multimodal_table_extractor.py # PDF table extraction (3 methods)
│
├── qdrant/
│   └── qdrant.py                    # Async Qdrant client wrapper
│
├── utils/
│   └── Data_ingestion.py            # PDF loading, chunking, unified ingest
│
└── docs/
    ├── ARCHITECTURE.md              # Deep-dive architecture guide
    ├── UNIFIED_ARCHITECTURE.md      # Unified system design
    └── mainarchitecture.md          # Orchestrator flow diagram
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Qdrant](https://qdrant.tech/documentation/quick-start/) running on `localhost:6333`
- [Neo4j](https://neo4j.com/download/) running on `localhost:7687` (optional, for Graph RAG)

### 1. Clone & Install

```bash
git clone https://github.com/tanishq450/Unify.git
cd Unify

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# LLM API (OpenAI-compatible endpoint)
MESH_API_BASE=https://api.meshapi.ai/v1
MESH_API_KEY=your-api-key-here

# Qdrant vector database
QDRANT_URL=http://localhost:6333

# Neo4j (optional — Graph RAG)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
```

### 3. Start Services

```bash
# Start Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant

# Start Neo4j (Docker, optional)
docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### 4. Run

```bash
# Ingest a PDF
python3 main.py ingest path/to/financial_report.pdf my_collection

# Query
python3 main.py query "What was Apple's revenue in 2024?" my_collection

# Interactive REPL
python3 main.py interactive my_collection
```

---

## 💡 Usage Examples

### Interactive Mode

```
🏦 Finance RAG — Interactive Mode
Type 'quit' to exit, 'ingest <path>' to add a PDF.

You > What was the gross margin breakdown by segment?

======================================================================
  Intent:    numerical_table
  Strategy:  multimodal_rag
  Reasoning: Numerical/table query detected (gross margin, breakdown, segment)
======================================================================

📝 Draft Answer:
Based on the financial data, gross margin was 44.1% for FY2024...

----------------------------------------------------------------------
✅ Verified Answer (confidence: 0.92):
Based on the financial data, gross margin was 44.1% for FY2024...

----------------------------------------------------------------------
🔍 Claim Verification:
  ✓ [numerical] Gross margin was 44.1%
    └─ Exact match: 44.1 in context
  ✓ [temporal] In fiscal year 2024
    └─ Date verified in context
======================================================================
```

---

## 🔧 Component Details

### Hybrid Retriever

The search engine combines two retrieval methods for maximum recall and precision:

```
Query → Dense Embedding (text-embedding-3-small)
      → Sparse Embedding (BM25 via fastembed)
      → Qdrant RRF Fusion (k=20 candidates)
      → Cross-Encoder Reranking (bge-reranker-v2-m3 → top 5)
      → MMR Diversity Filtering (λ=0.7)
      → Final Results
```

### Hallucination Verifier

Every LLM-generated answer goes through a 3-stage verification pipeline:

1. **Decompose** — Break answer into atomic claims (numerical, temporal, comparative, etc.)
2. **Verify** — Route each claim to a type-specific verifier:
   - **Numerical**: exact/fuzzy number matching against context (±1% exact, ±5% fuzzy)
   - **Comparative**: recalculate growth percentages from base values
   - **Temporal**: validate dates/periods exist in source documents
   - **Computational**: recompute formulas (e.g., gross margin = (rev - cogs) / rev)
3. **Regenerate** — Produce final answer using *only* verified claims

### Table Extraction (3 methods, cascading fallback)

| Method | Accuracy | Cost | Use Case |
|---|---|---|---|
| LlamaParse | ~90% | $0.003/page | Production (best quality) |
| pdfplumber | ~75% | Free | Local/offline |
| GPT-4V / Claude | ~85% | $0.01-0.03/page | Complex/scanned tables |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | GPT-4o-mini via MeshAPI (OpenAI-compatible) |
| **Embeddings** | `text-embedding-3-small` (dense) + `Qdrant/bm25` (sparse) |
| **Vector DB** | Qdrant (hybrid search with RRF fusion) |
| **Graph DB** | Neo4j (knowledge graph + Cypher QA) |
| **Reranker** | `BAAI/bge-reranker-v2-m3` (cross-encoder) |
| **Chunking** | Chonkie (token-based, 1000 tokens, 200 overlap) |
| **PDF Parsing** | PyMuPDF + pdfplumber + LlamaParse |
| **Orchestration** | LlamaIndex (core RAG) + LangChain (Graph RAG) |
| **Logging** | Loguru |

---

## 📄 License

