"""Central settings, loaded from environment / .env."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    cloudflare_api_token: str = ""
    cloudflare_account_id: str = ""  # dash.cloudflare.com > Workers AI (or any zone's overview)
    llm_provider: str = "cloudflare"  # groq | openrouter | deepseek | cloudflare
    judge_provider: str = ""  # eval judge's provider; empty = same as llm_provider
    llm_model: str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"  # a model id valid for the chosen provider
    # Groq reports tokens but not cost. Set both to price runs for models without a built-in
    # price (USD per million tokens); otherwise cost is reported as 0 for those models.
    llm_price_input_per_m: float | None = None
    llm_price_output_per_m: float | None = None
    judge_model: str = ""  # eval LLM judge; empty = same as llm_model
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "jinaai/jina-reranker-v1-turbo-en"  # small + fast on CPU
    rerank_candidates: int = 30  # how many fused hits the re-ranker scores
    rerank_max_chars: int = 1000  # truncate each passage before scoring (latency)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_environment: str = "development"  # production | staging | development
    langfuse_release: str = "0.1.0"  # shown on every trace; bump (or set to a git sha) per release

    raw_dir: Path = ROOT / "data" / "raw"
    index_dir: Path = ROOT / "data" / "index"

    chunking: str = "heading"  # fixed | heading
    chunk_tokens: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    use_bm25: bool = True
    use_reranker: bool = False
    use_agent: bool = False
    # Refuse when the best vector cosine similarity is below this. Tune on the golden set.
    refusal_threshold: float = 0.55


settings = Settings()
