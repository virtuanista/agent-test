import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


_load_env()

# Hosts that almost never carry useful text for research (social feeds, doc dumps).
_DEFAULT_BLOCK = "pinterest.com,instagram.com,facebook.com,tiktok.com,scribd.com,slideshare.net"


# Free OpenRouter models that answered a forced tool call correctly (probed 2026-10-11). Fallback pool only.
# Left out on purpose: gemma-4-* (upstream 429s), nemotron-3-ultra (ignores tool_choice=required),
# inkling (403: only on "agentic harnesses").
_OPENROUTER_DEFAULT = (
    "poolside/laguna-s-2.1:free,poolside/laguna-xs-2.1:free,"
    "nvidia/nemotron-3.5-lightning:free,apodex/apodex-1.1-mini:free,cohere/north-mini-code:free"
)
PROVIDER_SEP = "::"  # pool entry "openrouter::vendor/model" = that model on the OpenRouter endpoint


def _pool() -> tuple[str, ...]:
    base = os.getenv("LLM_MODEL_POOL", "openai/gpt-oss-20b,qwen/qwen3.8-27b,openai/gpt-oss-120b").split(",")
    extra: list[str] = []
    if os.getenv("OPENROUTER_API_KEY"):  # no key, no fallback: the pool stays exactly as before
        models = [m.strip() for m in os.getenv("OPENROUTER_MODELS", _OPENROUTER_DEFAULT).split(",")]
        extra = [f"openrouter{PROVIDER_SEP}{m}" for m in models if m]  # an empty entry must not become "openrouter::"
    return tuple(m.strip() for m in base + extra if m.strip())


def today() -> str:
    return date.today().isoformat()


@dataclass(frozen=True)
class Settings:
    # Any OpenAI-compatible endpoint: Groq, LiteLLM, OmniRoute, Ollama...
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")
    model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    strong_model: str = os.getenv("LLM_MODEL_STRONG", "openai/gpt-oss-120b")
    # Each model has its own free-tier quota (Groq: 8000 tokens/min per model); a 429 moves to the next one.
    model_pool: tuple[str, ...] = _pool()
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    max_loops: int = int(os.getenv("MAX_LOOPS", "25"))
    read_pages: int = int(os.getenv("READ_PAGES", "2"))
    page_chars: int = int(os.getenv("PAGE_CHARS", "1000"))
    block_domains: tuple[str, ...] = tuple(
        d.strip().lower() for d in os.getenv("BLOCK_DOMAINS", _DEFAULT_BLOCK).split(",") if d.strip()
    )
    search_cache_hours: float = float(os.getenv("SEARCH_CACHE_HOURS", "12"))
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    db_path: str = os.getenv("DB_PATH", str(ROOT / "agent.db"))


settings = Settings()


def available_targets() -> list[str]:
    """Pool entries whose provider has credentials configured, in priority order."""
    ready = ("default", "openrouter") if settings.openrouter_api_key else ("default",)
    return [m for m in dict.fromkeys(settings.model_pool) if split_target(m)[0] in ready]


def split_target(target: str) -> tuple[str, str]:
    """'openrouter::vendor/model' -> ('openrouter', 'vendor/model'); a bare name uses the default endpoint."""
    provider, sep, name = target.partition(PROVIDER_SEP)
    return (provider, name) if sep else ("default", target)
