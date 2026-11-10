"""Application configuration, loaded from environment variables.

All tunables (LLM provider/model, file limits, ML defaults) are read here so the
rest of the codebase never touches ``os.environ`` directly. API keys are never
hard-coded — they come from the environment / ``.env`` file only.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository backend root (…/backend). Used to resolve default data/artifact dirs.
BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed application settings.

    Values are read (case-insensitively) from environment variables, falling back
    to a local ``.env`` file. Unknown variables are ignored so a shared ``.env``
    with unrelated keys does not break startup.
    """

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Application ----------------------------------------------------
    app_name: str = "Agentic AutoML"
    environment: str = "development"
    log_level: str = "INFO"
    cors_origins: list[str] = Field(
        default=[
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
        ]
    )

    # ---- Storage --------------------------------------------------------
    data_dir: Path = BACKEND_ROOT / "data"
    artifacts_dir: Path = BACKEND_ROOT / "artifacts"

    # ---- Upload / dataset limits ---------------------------------------
    max_upload_mb: int = 50
    max_rows: int = 200_000
    max_columns: int = 2_000

    # ---- ML defaults ----------------------------------------------------
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    max_training_rows: int = 50_000  # subsample very large datasets for speed

    # ---- LLM ------------------------------------------------------------
    # Provider abstraction — the agents call ``get_llm()`` which reads these.
    llm_provider: str = "groq"  # groq | openai
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.1
    llm_max_tokens: int | None = None
    llm_timeout: int = 60

    groq_api_key: str | None = None
    openai_api_key: str | None = None
    llm_api_key: str | None = None  # generic override (takes precedence)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, value: object) -> object:
        """Allow ``CORS_ORIGINS`` to be a comma-separated string or JSON list."""
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                return value  # let pydantic parse JSON
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]
        return value

    @property
    def active_api_key(self) -> str | None:
        """Return the API key for the configured provider, if any."""
        if self.llm_api_key:
            return self.llm_api_key
        if self.llm_provider == "openai":
            return self.openai_api_key
        return self.groq_api_key

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        """Create storage directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance."""
    return Settings()
