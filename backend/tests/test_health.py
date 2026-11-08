"""Tests for the /health endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok_with_service_info():
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "Smart Knowledge Assistant"
    assert body["environment"] in {"development", "staging", "production"}


def test_health_returns_a_version():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["version"]


def test_openapi_document_describes_health_endpoint():
    openapi = client.get("/openapi.json").json()
    assert "/health" in openapi["paths"]
    assert "Smart Knowledge Assistant" in openapi["info"]["title"]
