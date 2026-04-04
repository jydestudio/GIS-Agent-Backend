"""
Application settings loaded from environment variables.

Uses pydantic-settings for type-safe configuration with validation.
All secrets are loaded from .env and never hardcoded.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the GeoAI Agent backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    app_name: str = "GeoAI Agent"
    app_version: str = "0.1.0"
    debug: bool = False

    # ── OpenRouter LLM ───────────────────────────
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "google/gemini-2.0-flash-001"

    # ── Neon PostgreSQL ──────────────────────────
    database_url: str

    # ── CORS ─────────────────────────────────────
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    # ── Static Files ─────────────────────────────
    static_maps_dir: str = "output_maps"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings singleton.

    Using lru_cache ensures we only parse env vars once per process.
    """
    return Settings()  # type: ignore[call-arg]
