from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    llama_cloud_api_key: str = ""
    cohere_api_key: str = ""
    pinecone_api_key: str = ""

    pinecone_index_name: str = "health-insurance-policies"
    pinecone_environment: str = "us-east-1"

    eval_model: str = "claude-haiku-4-5-20251001"  # cheaper model for eval runs
    retrieval_mode: str = "hybrid_rerank"
    retrieval_top_k: int = 5
    retrieval_candidate_k: int = 20

    cors_origins: str = "http://localhost:3000"
    data_dir: str = "data"

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()
