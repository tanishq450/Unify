import os
from openai import OpenAI
import ollama

# ── Defaults (override via environment variables) ──────────────────────
MESH_API_BASE = os.getenv("MESH_API_BASE", "https://api.meshapi.ai/v1")
MESH_API_KEY  = os.getenv("MESH_API_KEY",  "rsk_01KQMA836XVPYT6HX34QDX8KPG")
DEFAULT_MODEL = "anthropic/claude-opus-4.1"


class ModelLoader:
    def __init__(self):
        self.client: OpenAI | None = None
        self.embed_model_name: str = "qwen3-embedding:4b"

    def load_models(self):
        """Create an OpenAI-compatible client pointed at MeshAPI and init embedding model."""
        self.client = OpenAI(
            api_key=MESH_API_KEY,
            base_url=MESH_API_BASE,
        )
        self.load_embedding_model()

    def chat(self, messages: list[dict], model: str = DEFAULT_MODEL, temperature: float = 0.1) -> str:
        """Send a chat request and return the reply text."""
        response = self.client.chat.completions.create(
            model=model, messages=messages, temperature=temperature
        )
        return response.choices[0].message.content
    

    def load_embedding_model(self, model: str = "qwen3-embedding:4b"):
        """Store the Ollama embedding model name to use for embed()."""
        self.embed_model_name = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return dense embeddings using a local Ollama model."""
        response = ollama.embed(model=self.embed_model_name, input=texts)
        return response.embeddings

    def encode(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Alias for embed() to match sentence-transformers API."""
        return self.embed(texts)

    def generate(self, prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.1) -> str:
        """Single-turn completion (matches RawOpenAIClient API)."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, model=model, temperature=temperature)
