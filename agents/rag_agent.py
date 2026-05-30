from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from core.llm import get_llm
from core.vector_store import get_vector_store
from agents.state import AgentState

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical knowledge assistant. Use the retrieved medical literature below 
to answer the user's question accurately. If the context does not contain enough information, 
say so clearly — do NOT hallucinate medical facts.

Context:
{context}

Answer concisely and cite relevant parts of the context."""),
    ("human", "{query}"),
])


def rag_agent(state: AgentState) -> AgentState:
    """Retrieves relevant medical literature and generates a grounded answer."""
    llm = get_llm()
    store = get_vector_store()
    retriever = store.retriever()

    # Retrieve context
    docs = retriever.invoke(state["query"])
    context = "\n\n".join(d.page_content for d in docs)

    chain = RAG_PROMPT | llm
    result = chain.invoke({"query": state["query"], "context": context})

    return {
        **state,
        "retrieved_context": context,
        "final_response": result.content,
        "confidence": 0.87,
    }
