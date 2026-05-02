"""
Raw OpenAI client for direct chat completions (non-LlamaIndex / non-LangChain).

Use this when you need the vanilla OpenAI SDK, e.g. for the
hallucination verifier's LLM calls or quick ad-hoc completions.
"""

from openai import OpenAI
import os
import loguru


# ── Defaults (same env vars as llm.py) ─────────────────────────────────
MESH_API_BASE = os.getenv("MESH_API_BASE", "https://api.meshapi.ai/v1")
MESH_API_KEY = os.getenv("MESH_API_KEY", "rsk_01KQMA836XVPYT6HX34QDX8KPG")


class RawOpenAIClient:
    """
    Thin wrapper around the vanilla OpenAI Python SDK.

    Usage:
        client = RawOpenAIClient()
        answer = client.generate("Summarise this text: ...")
    """

    def __init__(self):
        self.log = loguru.logger
        self.client = OpenAI(
            base_url=MESH_API_BASE,
            api_key=MESH_API_KEY,
        )
        self.log.info("RawOpenAIClient initialised (MeshAPI)")

    def generate(self, prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.1) -> str:
        """Single-turn completion — returns the assistant message text."""
        response = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content