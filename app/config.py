from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    base_url: str = os.getenv("BASE_URL","")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    closeai_api_key: str = os.getenv("CLOSEAI_API_KEY", "")
    closeai_base_url: str = ''

    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    embeddings_model_name: str = os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-3-large")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./data/chroma")
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))
    collection_name: str = os.getenv("COLLECTION_NAME", "knowledge_base")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

settings = Settings()
