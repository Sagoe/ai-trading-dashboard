"""Central config — reads from .env automatically."""

from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Union


class Settings(BaseSettings):
    APP_ENV: str = "development"
    DEBUG: bool = True
    NEWS_API_KEY: str = ""
    CORS_ORIGINS: Union[List[str], str] = ["https://sagoe.github.io", "http://localhost:5173", "http://localhost:3000"]
    MODEL_DIR: str = "./ml/saved_models"
    DATA_CACHE_DIR: str = "./data/cache"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            # Handle both JSON array and comma-separated formats
            v = v.strip()
            if v.startswith("["):
                import json
                return json.loads(v)
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
