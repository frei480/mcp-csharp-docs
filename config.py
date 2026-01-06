from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigBase(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", 
    )
    GIT_REPO_URL: str
    DOCS_PATH: str
    AI_SERVER_URL: str
    RERANK_MODEL: str
    EMBEDDING_MODEL: str
    CHROMA_PERSIST_DIR: str
    CHUNK_SIZE : int
    CHUNK_OVERLAP: int
    COLLECTION_NAME: str
    REGISTRY_COLLECTION_NAME: str

cfg = ConfigBase()
