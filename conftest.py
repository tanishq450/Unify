"""
conftest.py — top-level pytest configuration

Stubs out heavy / unavailable third-party packages so that the modules under
test (intent_classifier, hallucination_verifier, evaluation) can be imported
without requiring a full Ollama installation, a live Qdrant/Neo4j instance,
or GPU drivers.

This file is loaded by pytest automatically before any test collection occurs.
"""

import sys
import types
from unittest.mock import MagicMock

# ── Helper ────────────────────────────────────────────────────────────────────

def _stub(name: str, attrs: dict = None):
    """Register a fake module (and its parent chain) in sys.modules.

    Does NOT override modules that already exist in sys.modules (e.g. real
    packages discovered on disk) unless `attrs` is supplied for the leaf.
    """
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        mod_name = ".".join(parts[:i])
        if mod_name not in sys.modules:
            mod = types.ModuleType(mod_name)
            # Mark every node as a package so sub-module imports work
            mod.__path__ = []
            mod.__package__ = mod_name
            sys.modules[mod_name] = mod

    leaf = sys.modules[name]
    if attrs:
        for k, v in attrs.items():
            setattr(leaf, k, v)
    return leaf


# ── llama_index stubs ─────────────────────────────────────────────────────────

for _mod in [
    "llama_index",
    "llama_index.embeddings",
    "llama_index.embeddings.ollama",
    "llama_index.embeddings.openai",
    "llama_index.llms",
    "llama_index.llms.openai",
    "llama_index.core",
    "llama_index.core.settings",
    "llama_index.readers",
    "llama_index.readers.file",
    "llama_index.instrumentation",
]:
    _stub(_mod)

sys.modules["llama_index.embeddings.ollama"].OllamaEmbedding = MagicMock()
sys.modules["llama_index.embeddings.openai"].OpenAIEmbedding = MagicMock()
sys.modules["llama_index.llms.openai"].OpenAI = MagicMock()
sys.modules["llama_index.core"].Settings = MagicMock()
sys.modules["llama_index.core"].VectorStoreIndex = MagicMock()
sys.modules["llama_index.core"].SimpleDirectoryReader = MagicMock()
sys.modules["llama_index.core"].StorageContext = MagicMock()
sys.modules["llama_index.core"].Document = MagicMock()
sys.modules["llama_index.core"].load_index_from_storage = MagicMock()
sys.modules["llama_index.readers.file"].PDFReader = MagicMock()

# ── Other heavy / optional packages ──────────────────────────────────────────

for _mod in [
    "qdrant",
    "qdrant.qdrant",
    "qdrant_client",
    "qdrant_client.models",
    "fastembed",
    "fitz",
    "chonkie",
    "langchain_neo4j",
    "langchain_experimental",
    "langchain_experimental.graph_transformers",
    "langchain_openai",
    "langchain_core",
    "langchain_core.language_models",
    "neo4j",
]:
    _stub(_mod)

sys.modules["qdrant_client.models"].SparseVector = MagicMock()
sys.modules["fastembed"].SparseTextEmbedding = MagicMock(
    return_value=MagicMock(embed=MagicMock(return_value=[]))
)
sys.modules["fitz"].open = MagicMock()

# ── Model_loader stub (depends on unavailable Ollama) ────────────────────────

_stub("Model_loader")
_stub("Model_loader.llm")
_stub("Model_loader.embedding_model")


class _FakeModelLoader:
    def __init__(self): pass
    def load_models(self): pass
    def set_settings(self): pass
    @staticmethod
    def get_langchain_llm(): return MagicMock()


sys.modules["Model_loader.llm"].ModelLoader = _FakeModelLoader
sys.modules["Model_loader.embedding_model"].RawOpenAIClient = MagicMock
sys.modules["Model_loader"].ModelLoader = _FakeModelLoader

# ── implementations: stub only the heavy sub-modules ─────────────────────────
# The real implementations/ package lives on disk; we must NOT replace it.
# We only pre-populate sys.modules for sub-modules that have dependencies we
# cannot satisfy (Rag.py → Model_loader → Ollama; Graph_rag.py → Neo4j; etc.)
# By inserting them before Python tries to import them from disk, the import
# machinery will use our stubs instead.

class _FakeRagPipeline:
    async def query(self, **kwargs): return {}


class _FakeGraphRAG:
    def load_llm(self, llm): pass
    def query(self, q): return {}


_rag_mod = types.ModuleType("implementations.Rag")
_rag_mod.Rag_pipeline = _FakeRagPipeline
sys.modules["implementations.Rag"] = _rag_mod

_graph_mod = types.ModuleType("implementations.Graph_rag")
_graph_mod.GRAPH_RAG = _FakeGraphRAG
sys.modules["implementations.Graph_rag"] = _graph_mod

_hybrid_mod = types.ModuleType("implementations.hybrid_retriever")
_hybrid_mod.HybridRetriever = MagicMock()
_hybrid_mod.FinanceHybridRetriever = MagicMock()
sys.modules["implementations.hybrid_retriever"] = _hybrid_mod

_mmte_mod = types.ModuleType("implementations.multimodal_table_extractor")
_mmte_mod.UnifiedTableExtractor = MagicMock()
_mmte_mod.TableAwareRAG = MagicMock()
sys.modules["implementations.multimodal_table_extractor"] = _mmte_mod

# ── utils: stub only the heavy sub-module ────────────────────────────────────

_utils_di_mod = types.ModuleType("utils.Data_ingestion")
_utils_di_mod.Docloader = MagicMock()
_utils_di_mod.unified_ingest = MagicMock()
_utils_di_mod.chunking = MagicMock()
sys.modules["utils.Data_ingestion"] = _utils_di_mod
