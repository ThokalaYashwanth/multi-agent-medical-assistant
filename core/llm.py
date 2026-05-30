from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from core.config import settings


def get_llm():
    if settings.LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=settings.TEMPERATURE,
        )
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
    )


def get_embeddings():
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )
