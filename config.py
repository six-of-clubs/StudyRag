"""
Centralised configurations:

Reads from .env file (or real environment variables) and exposes a single
`settings` object that every other module imports.  Values are validated at
startup so mis-configuration fails fast.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (no-op if file is missing)
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


# ---------------------------------------------------------------------------
# Model presets — each mode maps to an Ollama model name.
# Override in .env if you want different models.
# ---------------------------------------------------------------------------

MODEL_PRESETS = {
    "fast": _env("MODEL_FAST", "mistral:7b"),
    "thinking": _env("MODEL_THINKING", "deepseek-r1:8b"),
    "math": _env("MODEL_MATH", "phi4-mini"),
}

DEFAULT_MODE = "fast"


@dataclass(frozen=True)
class Settings:
    """Immutable application settings, populated once at import time."""

    # LLM
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "mistral:7b"))

    # Embedding
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

    # Chunking
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 512))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 64))

    # Retrieval
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    similarity_threshold: float = field(default_factory=lambda: _env_float("SIMILARITY_THRESHOLD", 0.3))

    # Vector store
    chroma_persist_dir: str = field(default_factory=lambda: _env("CHROMA_PERSIST_DIR", "./data/chroma"))

    # Server
    api_host: str = field(default_factory=lambda: _env("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 8000))

    def __post_init__(self):
        # Validate critical constraints
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be less than "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"SIMILARITY_THRESHOLD must be between 0 and 1, "
                f"got {self.similarity_threshold}"
            )


# Singleton — import this everywhere
settings = Settings()
