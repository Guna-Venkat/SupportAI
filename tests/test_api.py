"""
test_api.py
===========
Integration tests for FastAPI REST API endpoints.
"""

import os
from unittest.mock import MagicMock

# Set testing environment variable BEFORE importing app
os.environ["TESTING"] = "true"

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_api_version(client):
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "SupportAI"
    assert "version" in data


def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "classifier_loaded" in data


def test_api_predict_validation_failure(client):
    # Text is too short (< 3 chars)
    response = client.post("/predict", json={"text": "hi"})
    assert response.status_code == 422


def test_api_predict_success_mocked(client):
    mock_engine = MagicMock()
    mock_engine.route_ticket.return_value = {
        "intent": "card_payment_fee_charged",
        "confidence": 0.95,
        "route": "classifier",
        "retrieved_docs": [],
        "llm_used": False,
        "reply": "Automated routing to category: card_payment_fee_charged",
    }

    original_engine = app.state.decision_engine
    app.state.decision_engine = mock_engine
    try:
        response = client.post(
            "/predict", json={"text": "I was charged an extra fee on my card payment."}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "card_payment_fee_charged"
        assert data["route"] == "classifier"
        assert data["llm_used"] is False
        mock_engine.route_ticket.assert_called_once_with(
            "I was charged an extra fee on my card payment."
        )
    finally:
        app.state.decision_engine = original_engine


def test_api_retrieve_success_mocked(client):
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"rank": 1, "index": 42, "score": 0.85, "text": "matched text"}
    ]

    original_retriever = app.state.retriever
    app.state.retriever = mock_retriever
    try:
        response = client.post("/retrieve", json={"query": "my account status", "top_k": 1})
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "my account status"
        assert len(data["results"]) == 1
        assert data["results"][0]["text"] == "matched text"
    finally:
        app.state.retriever = original_retriever


def test_api_explain_success_mocked(client):
    mock_explainer = MagicMock()
    mock_explainer.explain_ticket.return_value = {
        "predicted_class": "card_payment_fee_charged",
        "predicted_probability": 0.95,
        "attributions": [["charged", 0.45], ["fee", 0.35]],
        "explanation_html": "<p>Explanation</p>",
    }

    original_explainer = app.state.explainer
    app.state.explainer = mock_explainer
    try:
        response = client.post(
            "/explain", json={"text": "charged fee", "num_features": 2, "num_samples": 10}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_class"] == "card_payment_fee_charged"
        assert data["predicted_probability"] == 0.95
        assert len(data["attributions"]) == 2
        assert "Explanation" in data["explanation_html"]
    finally:
        app.state.explainer = original_explainer


def test_api_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "supportai_requests_total" in response.text


def test_api_observability_middleware_logs_traces(client):
    import json
    from pathlib import Path

    traces_file = Path("logs/traces.jsonl")
    if traces_file.exists():
        try:
            traces_file.unlink()
        except OSError:
            pass

    mock_engine = MagicMock()
    mock_engine.route_ticket.return_value = {
        "intent": "card_payment_fee_charged",
        "confidence": 0.95,
        "route": "classifier",
        "retrieved_docs": [],
        "llm_used": False,
        "reply": "Automated reply",
    }

    original_engine = app.state.decision_engine
    app.state.decision_engine = mock_engine
    try:
        response = client.post(
            "/predict", json={"text": "I was charged an extra fee on my card payment."}
        )
        assert response.status_code == 200

        # Check X-Request-ID response header
        assert "X-Request-ID" in response.headers
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) > 0

        # Check logs/traces.jsonl was created and contains a trace matching request_id
        assert traces_file.exists()
        with open(traces_file, encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) >= 1
        trace = json.loads(lines[-1])
        assert trace["request_id"] == request_id
        assert trace["endpoint"] == "/predict"
        assert trace["method"] == "POST"
        assert trace["intent"] == "card_payment_fee_charged"
        assert trace["confidence"] == 0.95
        assert trace["route"] == "classifier"
        assert "git_commit" in trace
        assert "experiment_id" in trace
    finally:
        app.state.decision_engine = original_engine


def test_api_demo_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "SupportAI Control Center" in response.text
