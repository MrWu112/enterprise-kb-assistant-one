from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config import settings
from app.rag.vectorstore import get_vectorstore
from redis.asyncio import Redis, ConnectionPool
import asyncio

# 全局连接池(整个进程只创建一次)
_pool: ConnectionPool | None = None
_lock = asyncio.Lock()

async def get_redis() -> Redis:
    global _pool
    if _pool is None:
        async with _lock:
            if _pool is None:
                _pool = ConnectionPool.from_url(
                    settings.redis_url,
                    max_connections=20,
                    decode_responses=True, # 自动把bytes转成str
                )
    redis_client = Redis(connection_pool=_pool)
    return redis_client


def get__llm():
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.closeai_api_key,
        temperature=0.2,
        streaming=True,
        base_url=settings.closeai_base_url,

    )

def get_embeddings():
    return OpenAIEmbeddings(
        api_key=settings.closeai_api_key,
        base_url=settings.closeai_base_url,
        model=settings.embeddings_model_name,
    )

def get_vs():
    return get_vectorstore(get_embeddings())

if __name__ == '__main__':
    print(get__llm())
    print(get_embeddings())
    print(get_vs())