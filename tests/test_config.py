"""
Tests for configuration settings.
"""
import pytest
from src.config.settings import Settings


def test_default_settings():
    """Test default configuration values."""
    settings = Settings()
    
    assert settings.app_name == "Kimi Knowledge Base"
    assert settings.app_version == "0.1.0"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.qdrant_host == "localhost"
    assert settings.qdrant_port == 6333


def test_settings_from_env(monkeypatch):
    """Test settings loaded from environment variables."""
    monkeypatch.setenv("APP_NAME", "Test App")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("DEBUG", "true")
    
    settings = Settings()
    
    assert settings.app_name == "Test App"
    assert settings.port == 9000
    assert settings.debug is True


def test_cache_settings():
    """Test cache-related settings."""
    settings = Settings()
    
    assert settings.cache_db_path == "./data/cache.db"
    assert settings.cache_ttl == 3600
    assert settings.cache_max_size == 1000


def test_embedding_settings():
    """Test embedding model settings."""
    settings = Settings()
    
    assert "sentence-transformers" in settings.embedding_model
    assert settings.embedding_dimension == 384
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50