"""
Configuration settings for the Kimi Knowledge Base system.
"""
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Application Configuration
    app_name: str = "Kimi Knowledge Base"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "knowledge_base"
    
    # Cache Configuration
    cache_db_path: str = "./data/cache.db"
    cache_ttl: int = 3600
    cache_max_size: int = 1000
    
    # File Storage
    upload_dir: str = "./data/documents"
    max_file_size: str = "50MB"
    
    # Kimi2 API Configuration
    moonshot_api_key: Optional[str] = None
    moonshot_base_url: str = "https://api.moonshot.cn/v1"
    
    # Security
    secret_key: str = "dev-secret-key"
    access_token_expire_minutes: int = 30
    
    # Embedding Model
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384
    
    # Text Processing
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_chunks_per_query: int = 10
    
    # Performance
    max_concurrent_requests: int = 10
    request_timeout: int = 30


# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """获取全局配置实例"""
    return settings