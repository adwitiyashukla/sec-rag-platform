"""Typed application settings.

Every tunable lives here, is validated at import time, and is reachable from an
environment variable prefixed with SECRAG_. Nothing in the codebase reads
os.environ directly, which means a misconfiguration fails loudly at startup
rather than quietly halfway through a request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from secrag.core.errors import ConfigurationError

ProviderName = Literal["groq", "gemini", "echo"]

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Validated configuration for the whole platform."""

    model_config = SettingsConfigDict(
        env_prefix="SECRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Paths -----------------------------------------------------------
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = _PROJECT_ROOT / "data"

    # --- LLM providers ---------------------------------------------------
    groq_api_key: str = ""
    gemini_api_key: str = ""
    llm_providers: str = Field(
        default="groq,gemini",
        description="Comma separated fallback chain, tried left to right",
    )
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-2.5-flash-lite"

    request_timeout_s: float = Field(default=45.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=6)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=64, le=8192)

    # --- SEC EDGAR -------------------------------------------------------
    # EDGAR blocks requests without a descriptive User-Agent carrying a contact
    # address. This is a documented requirement, not an optional courtesy.
    edgar_user_agent: str = "sec-rag-platform/0.1.0 (contact@example.com)"
    edgar_rate_limit_per_s: float = Field(
        default=8.0, gt=0, le=10, description="EDGAR permits at most 10 requests per second"
    )

    # --- Models ----------------------------------------------------------
    # ONNX rather than PyTorch. Same weights, roughly a sixth of the image size
    # and an order of magnitude faster to cold start, which is what makes free
    # tier hosting viable.
    dense_model: str = "BAAI/bge-small-en-v1.5"
    dense_dim: int = 384
    sparse_model: str = "prithivida/Splade_PP_en_v1"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    embed_batch_size: int = Field(default=64, ge=1, le=512)
    # fastembed defaults to the system temp directory, which is periodically
    # cleared and forces a re-download. Pinning it under the project keeps cold
    # starts predictable and lets the Docker build pre-bake the weights.
    model_cache_dir: Path | None = None

    # BGE was trained with an asymmetric prefix on the query side only.
    # Omitting it measurably degrades recall, so it is applied centrally.
    dense_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # --- Chunking --------------------------------------------------------
    chunk_target_tokens: int = Field(default=380, ge=64, le=1024)
    chunk_overlap_tokens: int = Field(default=64, ge=0, le=256)
    max_table_tokens: int = Field(default=900, ge=128, le=2048)

    # --- Retrieval -------------------------------------------------------
    dense_top_k: int = Field(default=30, ge=1, le=200)
    sparse_top_k: int = Field(default=30, ge=1, le=200)
    splade_top_k: int = Field(default=30, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1, le=200)
    rerank_top_n: int = Field(default=6, ge=1, le=50)
    rerank_candidates: int = Field(default=40, ge=1, le=200)
    enable_splade: bool = True

    # --- Router ----------------------------------------------------------
    router_confidence_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    router_model_path: Path | None = None

    # --- Guardrails ------------------------------------------------------
    min_groundedness: float = Field(default=0.45, ge=0.0, le=1.0)
    enable_pii_redaction: bool = True
    enable_injection_detection: bool = True

    # --- Semantic cache --------------------------------------------------
    cache_enabled: bool = True
    cache_similarity_threshold: float = Field(default=0.96, ge=0.5, le=1.0)
    cache_max_entries: int = Field(default=512, ge=1)
    cache_ttl_s: int = Field(default=3600, ge=0)

    # --- Evaluation ------------------------------------------------------
    eval_k: int = Field(default=6, ge=1, le=50)

    # --- Service ---------------------------------------------------------
    host: str = "0.0.0.0"  # noqa: S104 - containerised service must bind all interfaces
    port: int = Field(default=7860, ge=1, le=65535)
    log_level: str = "INFO"
    log_json: bool = False
    cors_origins: str = "*"

    # --- Derived ---------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def provider_chain(self) -> list[str]:
        """Provider names in fallback order, whitespace tolerant."""
        return [p.strip().lower() for p in self.llm_providers.split(",") if p.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def api_key_for(self, provider: str) -> str:
        return {"groq": self.groq_api_key, "gemini": self.gemini_api_key, "echo": "n/a"}.get(
            provider, ""
        )

    def model_for(self, provider: str) -> str:
        return {"groq": self.groq_model, "gemini": self.gemini_model, "echo": "echo-1"}.get(
            provider, ""
        )

    # --- Validation ------------------------------------------------------

    @model_validator(mode="after")
    def _validate(self) -> Self:
        known = {"groq", "gemini", "echo"}
        chain = self.provider_chain
        if not chain:
            msg = "SECRAG_LLM_PROVIDERS must name at least one provider"
            raise ConfigurationError(msg)
        if unknown := set(chain) - known:
            msg = f"Unknown provider(s) {sorted(unknown)}. Valid options: {sorted(known)}"
            raise ConfigurationError(msg)
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            msg = "chunk_overlap_tokens must be smaller than chunk_target_tokens"
            raise ConfigurationError(msg)
        if self.rerank_top_n > self.rerank_candidates:
            msg = "rerank_top_n cannot exceed rerank_candidates"
            raise ConfigurationError(msg)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def models_dir(self) -> Path:
        return self.model_cache_dir or (self.data_dir / "models")

    def ensure_dirs(self) -> None:
        """Create the on-disk layout. Safe to call repeatedly."""
        for path in (self.data_dir, self.raw_dir, self.index_dir, self.cache_dir, self.models_dir):
            path.mkdir(parents=True, exist_ok=True)

    def configured_providers(self) -> list[str]:
        """Providers in the chain that actually have credentials.

        The echo provider never needs a key, which is what lets the whole test
        suite and CI run with no secrets configured at all.
        """
        return [p for p in self.provider_chain if p == "echo" or self.api_key_for(p)]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
