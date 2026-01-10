from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigBase(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", 
    )
    # Git и документация
    GIT_REPO_URL: str = ""
    DOCS_PATH: str = "./docs"
    
    # LMStudio / AI сервер
    AI_SERVER_URL: str = "http://localhost:1234/v1"
    
    # Модели LMStudio
    EMBEDDING_MODEL: str = "text-embedding-embeddinggemma-300m"
    ENTITY_EXTRACTION_MODEL: str = "nvidia/nemotron-3-nano"
    RELATION_EXTRACTION_MODEL: str = "nvidia/nemotron-3-nano"
    QUERY_MODEL: str = "qwen/qwen3-vl-8b"
    RERANK_MODEL: str = "nvidia/nemotron-3-nano"
    
    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 128
    COLLECTION_NAME: str = "csharp_docs"
    REGISTRY_COLLECTION_NAME: str = "docs_registry"
    
    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"
    
    # GraphRAG настройки
    MAX_ENTITIES_PER_CHUNK: int = 20
    MAX_RELATIONS_PER_CHUNK: int = 15
    GRAPHRAG_CONTEXT_DEPTH: int = 2  # Глубина обхода графа для контекста

cfg = ConfigBase()
