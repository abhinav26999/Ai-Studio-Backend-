import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    PORT: int = 8000
    
    # Firebase settings
    FIREBASE_CREDENTIALS_PATH: str = "ai-studio-637ab-firebase-adminsdk-fbsvc-d51de0aebf.json"
    FIREBASE_PROJECT_ID: str = "ai-studio-637ab"
    FIREBASE_STORAGE_BUCKET: str = "ai-studio-637ab.firebasestorage.app"
    
    # Hugging Face Settings
    HF_TOKEN: str = ""
    HF_ROUTER_URL: str = "https://router.huggingface.co/v1"
    QWEN_MODEL_ID: str = "Qwen/Qwen2.5-7B-Instruct"
    BG_REMOVAL_MODEL_ID: str = "briaai/RMBG-2.0"  # Options: "briaai/RMBG-2.0" or "briaai/RMBG-1.4"
    
    # Redis & Celery Settings
    REDIS_URL: str = "redis://localhost:6379/0"
    ENABLE_CELERY: bool = True

    # Credit Costs Contract
    CREDIT_COST_IMAGE_GEN: int = 3
    CREDIT_COST_THEME_CHANGE: int = 3
    CREDIT_COST_BG_REMOVAL: int = 3
    CREDIT_COST_MESH_GEN: int = 5
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
