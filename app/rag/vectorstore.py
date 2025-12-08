from langchain_chroma import Chroma
from app.config import settings
from chromadb import HttpClient


def get_vectorstore(embeddings):
    client = HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port
    )
    return Chroma(
        client=client,
        collection_name=settings.collection_name,
        embedding_function=embeddings,
    )