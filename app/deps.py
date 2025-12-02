from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config import settings
from app.rag.vectorstore import get_vectorstore


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