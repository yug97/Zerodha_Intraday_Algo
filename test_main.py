"""Unit tests for FastAPI application"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data


class TestIndexEndpoints:
    """Test index endpoints"""
    
    def test_index_page(self):
        """Test HTML index page"""
        response = client.get("/")
        assert response.status_code == 200
        assert "html" in response.text.lower()
        assert "zerodha" in response.text.lower()
    
    def test_api_index(self):
        """Test API index endpoint"""
        response = client.get("/api/index")
        assert response.status_code == 200
        data = response.json()
        assert "api_key" in data
        assert "redirect_url" in data
        assert "login_url" in data


class TestStocksEndpoint:
    """Test stocks endpoint"""
    
    def test_get_stocks(self):
        """Test get stocks endpoint"""
        response = client.get("/api/stocks")
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "stocks" in data
        assert data["count"] > 0
        assert "INFY" in data["stocks"]


class TestConfigEndpoint:
    """Test config endpoint"""
    
    def test_get_config(self):
        """Test get config endpoint"""
        response = client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert "capital" in data
        assert "api_key" in data
        assert "host" in data
        assert "port" in data


class TestWebhookEndpoint:
    """Test webhook endpoint"""
    
    def test_webhook_missing_auth(self):
        """Test webhook without authentication"""
        response = client.post("/webhook", json={
            "stock": "INFY",
            "position": "long"
        })
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]
    
    def test_webhook_invalid_stock(self):
        """Test webhook with invalid stock"""
        response = client.post("/webhook", json={
            "stock": "INVALID",
            "position": "long"
        })
        assert response.status_code == 401  # No auth token
    
    def test_webhook_invalid_position(self):
        """Test webhook with invalid position"""
        response = client.post("/webhook", json={
            "stock": "INFY",
            "position": "invalid_position"
        })
        # Pydantic validation should handle this
        assert response.status_code in [401, 422]


class TestNotFoundEndpoint:
    """Test 404 handling"""
    
    def test_nonexistent_endpoint(self):
        """Test nonexistent endpoint"""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
