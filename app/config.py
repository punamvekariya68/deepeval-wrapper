import os
from typing import Optional, List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    app_name: str = "DeepEval REST API Wrapper"
    version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # JWT Configuration
    secret_key: str = os.getenv("SECRET_KEY", "change-this-secret-key-in-production")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Authentication
    admin_username: str = os.getenv("ADMIN_USERNAME", "admin")
    admin_password: str = os.getenv("ADMIN_PASSWORD", "changeme123")
    
    # API Key Authentication (comma-separated list)
    api_keys: str = os.getenv("API_KEYS", "deepeval-default-key")
    
    @property
    def api_keys_list(self) -> List[str]:
        """Get API keys as a list."""
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]
    
    # DeepEval Configuration
    deepeval_api_key: Optional[str] = os.getenv("DEEPEVAL_API_KEY")
    
    # LLM Provider API Keys
    # openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    # anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    # cohere_api_key: Optional[str] = os.getenv("COHERE_API_KEY")
    
    # Redis Configuration (optional)
    use_redis: bool = os.getenv("USE_REDIS", "false").lower() == "true"
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Celery Configuration (optional)
    celery_broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    celery_result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    
    # Evaluation Configuration
    default_max_concurrent: int = 10
    default_timeout: int = 300  # 5 minutes
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS
    cors_origins: List[str] = ["*"]  # Configure for production
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
