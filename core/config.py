from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    LLM_PROVIDER: str = "openai"          # "openai" | "gemini"
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    FAISS_INDEX_PATH: str = "data/faiss_index"
    MEDICAL_DOCS_PATH: str = "data/medical_docs"
    TOP_K_RETRIEVAL: int = 5
    MAX_TOKENS: int = 1024
    TEMPERATURE: float = 0.2

    class Config:
        env_file = ".env"


settings = Settings()
