from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import os

# ── Defaults (override via environment variables) ──────────────────────
MESH_API_BASE = os.getenv("MESH_API_BASE", "https://api.meshapi.ai/v1")
MESH_API_KEY = os.getenv("MESH_API_KEY", "rsk_01KQMA836XVPYT6HX34QDX8KPG")


class ModelLoader:
    def __init__(self):
        self.llm = None
        self.embed_model = None

    def load_models(self):
        """Load LlamaIndex LLM and embedding model (OpenAI-compatible via MeshAPI)."""
        self.llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_base=MESH_API_BASE,
            api_key=MESH_API_KEY,
        )
        self.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_base=MESH_API_BASE,
            api_key=MESH_API_KEY,
        )

    def set_settings(self):
        """Push loaded models into LlamaIndex global Settings."""
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

    # ── LangChain compatibility ────────────────────────────────────────
    @staticmethod
    def get_langchain_llm():
        """
        Return a LangChain-compatible ChatOpenAI instance.

        Required by:
          - Graph RAG (LLMGraphTransformer, GraphCypherQAChain)
          - Any LangChain chain/agent
        """
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_base=MESH_API_BASE,
            openai_api_key=MESH_API_KEY,
        )

