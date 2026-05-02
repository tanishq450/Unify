"""
Tests for api.py (FastAPI endpoints)

Uses FastAPI's TestClient to exercise the HTTP layer without starting a real
server or touching live databases / LLMs.  Heavy dependencies (orchestrator,
evaluation module) are mocked with pytest monkeypatch / unittest.mock.

Covered endpoints:
  GET  /        — health check
  POST /ingest  — file type validation, orchestrator path
  POST /query   — empty-query validation, orchestrator path
  POST /evaluate — component validation, evaluation dispatch
"""

import io
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


# ─────────────────────────────────────────────────────────────
# Fixtures — patch the heavy imports before importing `api`
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Build a TestClient with FinanceRAGOrchestrator fully mocked so that the
    module-level import in api.py does not attempt to load real models.
    """
    mock_orchestrator_cls = MagicMock()
    mock_orchestrator_instance = MagicMock()
    mock_orchestrator_instance.ingest = AsyncMock(return_value=None)
    mock_orchestrator_instance.query = AsyncMock(return_value={
        "answer": "Apple's revenue was $383B.",
        "verified_answer": "Apple's revenue was $383B.",
        "confidence": 0.95,
        "intent": "numerical_table",
        "strategy": "multimodal_rag",
        "routing_reasoning": "Numerical query detected.",
        "claims": [],
    })
    mock_orchestrator_cls.return_value = mock_orchestrator_instance

    with patch("main.FinanceRAGOrchestrator", mock_orchestrator_cls):
        import api  # import after patching
        api.orchestrators.clear()   # reset any cached instances
        yield TestClient(api.app)


# ─────────────────────────────────────────────────────────────
# 1. GET /  — health check
# ─────────────────────────────────────────────────────────────

def test_root_returns_ok(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "message" in body


# ─────────────────────────────────────────────────────────────
# 2. POST /ingest
# ─────────────────────────────────────────────────────────────

def test_ingest_rejects_non_pdf(client):
    """Only PDF files are accepted."""
    response = client.post(
        "/ingest",
        files={"file": ("report.docx", b"fake content", "application/octet-stream")},
        data={"collection_name": "test"},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_ingest_accepts_pdf(client):
    response = client.post(
        "/ingest",
        files={"file": ("report.pdf", b"%PDF-1.4 fake pdf content", "application/pdf")},
        data={"collection_name": "test"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "report.pdf" in body["message"]


def test_ingest_uses_default_collection_when_omitted(client):
    """collection_name defaults to 'default' if not supplied."""
    response = client.post(
        "/ingest",
        files={"file": ("annual.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert response.status_code == 200
    assert "default" in response.json()["message"]


def test_ingest_txt_file_rejected(client):
    response = client.post(
        "/ingest",
        files={"file": ("notes.txt", b"some text", "text/plain")},
        data={"collection_name": "test"},
    )
    assert response.status_code == 400


# ─────────────────────────────────────────────────────────────
# 3. POST /query
# ─────────────────────────────────────────────────────────────

def test_query_empty_string_rejected(client):
    response = client.post("/query", json={"query": "", "collection_name": "default"})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_query_whitespace_only_rejected(client):
    response = client.post("/query", json={"query": "   ", "collection_name": "default"})
    assert response.status_code == 400


def test_query_valid_returns_result(client):
    response = client.post(
        "/query",
        json={"query": "What was Apple's revenue in 2024?", "collection_name": "default"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "data" in body
    data = body["data"]
    assert "answer" in data
    assert "verified_answer" in data
    assert "confidence" in data
    assert "intent" in data
    assert "strategy" in data


def test_query_uses_default_collection(client):
    """collection_name has a default value of 'default'."""
    response = client.post("/query", json={"query": "Who is Apple's CEO?"})
    assert response.status_code == 200


def test_query_custom_collection(client):
    response = client.post(
        "/query",
        json={"query": "Apple revenue", "collection_name": "my_collection"},
    )
    assert response.status_code == 200


# ─────────────────────────────────────────────────────────────
# 4. POST /evaluate
# ─────────────────────────────────────────────────────────────

def _mock_evaluation_module():
    """Return a mock evaluation module."""
    mock_eval = MagicMock()
    mock_eval.evaluate_intent_classifier.return_value = {"accuracy": 0.97}
    mock_eval.evaluate_verifier.return_value = {"accuracy": 0.90}
    mock_eval.evaluate_e2e.return_value = {"faithfulness": 0.88}
    return mock_eval


def test_evaluate_invalid_component_rejected(client):
    response = client.post("/evaluate", json={"component": "unknown"})
    assert response.status_code == 400
    assert "intent" in response.json()["detail"]


@pytest.mark.parametrize("component,expected_key", [
    ("intent", "intent_classifier"),
    ("verifier", "hallucination_verifier"),
    ("e2e", "end_to_end"),
])
def test_evaluate_valid_component(client, component, expected_key):
    mock_eval = _mock_evaluation_module()
    with patch("api.evaluation", mock_eval, create=True):
        import importlib, api as api_mod
        api_mod_eval_backup = getattr(api_mod, "evaluation", None)

        # Directly patch the import inside the endpoint
        import sys
        sys.modules["evaluation"] = mock_eval

        response = client.post("/evaluate", json={"component": component})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["component"] == component
        assert expected_key in body["data"]


def test_evaluate_all_runs_all_components(client):
    mock_eval = _mock_evaluation_module()
    import sys
    sys.modules["evaluation"] = mock_eval

    response = client.post("/evaluate", json={"component": "all"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    data = body["data"]
    assert "intent_classifier" in data
    assert "hallucination_verifier" in data
    assert "end_to_end" in data
