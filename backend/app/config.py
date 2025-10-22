"""
Configuration settings for the Financial Intelligence System
"""

import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        from pydantic import BaseSettings
    except ImportError:
        # Fallback for older pydantic versions
        class BaseSettings:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

from pydantic import Field


class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    
    # PostgreSQL settings
    db_host: str = Field(default="localhost", env="DB_HOST")
    db_port: int = Field(default=5432, env="DB_PORT")
    db_name: str = Field(default="financial_intelligence", env="DB_NAME")
    db_user: str = Field(default="postgres", env="DB_USER")
    db_password: str = Field(default="postgres", env="DB_PASSWORD")
    
    # Connection pool settings
    db_min_connections: int = Field(default=5, env="DB_MIN_CONNECTIONS")
    db_max_connections: int = Field(default=20, env="DB_MAX_CONNECTIONS")
    
    # Redis settings
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Vector database settings
    embedding_dimension: int = Field(default=768, env="EMBEDDING_DIMENSION")
    
    @property
    def database_url(self) -> str:
        """Get database connection URL"""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class APISettings(BaseSettings):
    """API configuration settings"""
    
    # FastAPI settings
    app_name: str = Field(default="Advanced Financial Intelligence System", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Security settings
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS settings
    allowed_origins: list = Field(default=["*"], env="ALLOWED_ORIGINS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class ExternalAPISettings(BaseSettings):
    """External API configuration settings"""
    
    # Yahoo Finance (no API key required)
    yahoo_finance_base_url: str = Field(
        default="https://query1.finance.yahoo.com/v8/finance/chart/",
        env="YAHOO_FINANCE_BASE_URL"
    )
    
    # Alpha Vantage
    alpha_vantage_api_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    alpha_vantage_base_url: str = Field(
        default="https://www.alphavantage.co/query",
        env="ALPHA_VANTAGE_BASE_URL"
    )
    
    # Quandl
    quandl_api_key: Optional[str] = Field(default=None, env="QUANDL_API_KEY")
    quandl_base_url: str = Field(
        default="https://www.quandl.com/api/v3/",
        env="QUANDL_BASE_URL"
    )
    
    # Rate limiting
    api_rate_limit_per_minute: int = Field(default=60, env="API_RATE_LIMIT_PER_MINUTE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class MLSettings(BaseSettings):
    """Machine Learning configuration settings"""
    
    # Model settings
    default_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="DEFAULT_EMBEDDING_MODEL"
    )
    
    sentiment_model: str = Field(
        default="ProsusAI/finbert",
        env="SENTIMENT_MODEL"
    )
    
    # Processing settings
    max_document_length: int = Field(default=10000, env="MAX_DOCUMENT_LENGTH")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    
    # Cache settings
    model_cache_dir: str = Field(default="./models", env="MODEL_CACHE_DIR")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class CelerySettings(BaseSettings):
    """Celery configuration settings"""
    
    # Broker settings (using Redis)
    broker_url: str = Field(default="redis://localhost:6379/1", env="CELERY_BROKER_URL")
    result_backend: str = Field(default="redis://localhost:6379/2", env="CELERY_RESULT_BACKEND")
    
    # Task settings
    task_serializer: str = Field(default="json", env="CELERY_TASK_SERIALIZER")
    result_serializer: str = Field(default="json", env="CELERY_RESULT_SERIALIZER")
    accept_content: list = Field(default=["json"], env="CELERY_ACCEPT_CONTENT")
    timezone: str = Field(default="UTC", env="CELERY_TIMEZONE")
    enable_utc: bool = Field(default=True, env="CELERY_ENABLE_UTC")
    
    # Worker settings
    worker_prefetch_multiplier: int = Field(default=1, env="CELERY_WORKER_PREFETCH_MULTIPLIER")
    task_acks_late: bool = Field(default=True, env="CELERY_TASK_ACKS_LATE")
    worker_max_tasks_per_child: int = Field(default=1000, env="CELERY_WORKER_MAX_TASKS_PER_CHILD")
    
    # Monitoring
    flower_port: int = Field(default=5555, env="FLOWER_PORT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class Settings(BaseSettings):
    """Main application settings"""
    
    # Sub-settings
    database: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()
    external_apis: ExternalAPISettings = ExternalAPISettings()
    ml: MLSettings = MLSettings()
    celery: CelerySettings = CelerySettings()
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings