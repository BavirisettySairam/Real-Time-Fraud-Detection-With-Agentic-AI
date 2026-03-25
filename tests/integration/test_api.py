import pytest
from fastapi.testclient import TestClient

from services.api_gateway.main import app


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] in {"healthy", "degraded"}
    assert "components" in body
    assert "uptime_seconds" in body


def test_predict_endpoint_smoke(client):
    payload = {
        "TransactionAmt": 250.0,
        "ProductCD": "W",
        "card1": 12345,
        "hour": 14,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["decision"] in {"APPROVE", "REVIEW", "BLOCK"}
    assert "agent_scores" in body
    assert "processing_time_ms" in body


def test_batch_predict_endpoint(client):
    payload = {
        "transactions": [
            {"TransactionAmt": 100.0, "ProductCD": "W", "card1": 1},
            {"TransactionAmt": 7000.0, "ProductCD": "W", "card1": 1, "hour": 2},
        ]
    }
    response = client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["total_processed"] == 2
    assert len(body["predictions"]) == 2


def test_metrics_endpoint_available(client):
    response = client.get("/metrics")
    assert response.status_code == 200


def test_agents_endpoint_reports_langgraph_active(client):
    response = client.get("/api/v1/agents")
    assert response.status_code == 200

    body = response.json()
    assert "execution" in body
    assert body["execution"]["use_langgraph"] is True
    assert body["execution"]["langgraph_active"] is True


def test_workflow_endpoint_returns_mermaid_payload(client):
    response = client.get("/api/v1/workflow")
    assert response.status_code == 200

    body = response.json()
    assert body["workflow"] == "LangGraph Agent Pipeline"
    assert body["mode"] in {"parallel", "sequential"}
    assert isinstance(body["langgraph_active"], bool)
    assert isinstance(body["mermaid_diagram"], str)
    assert len(body["mermaid_diagram"].strip()) > 0
