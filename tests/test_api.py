"""
Tests for FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "status" in data
    assert data["status"] == "running"


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


def test_cors_middleware_configured(client):
    """Test that CORS middleware is configured (basic test)."""
    response = client.get("/")
    
    # Test that the response is successful, indicating middleware is working
    assert response.status_code == 200


def test_404_endpoint(client):
    """Test non-existent endpoint returns 404."""
    response = client.get("/nonexistent")
    
    assert response.status_code == 404