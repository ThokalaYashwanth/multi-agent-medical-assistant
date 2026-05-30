from langchain.prompts import ChatPromptTemplate
from core.llm import get_llm
from agents.state import AgentState

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a medical query router. Classify the user's query into exactly one intent:
- diagnosis: user describes symptoms and wants diagnosis/differential
- rag: user asks about medical literature, drugs, treatments, or clinical facts
- drug: user asks about drug interactions, dosages, or contraindications
- report: user wants a clinical note, SOAP note, or medical summary generated
- safety: query contains dangerous/harmful content that needs flagging

Respond with ONLY the intent word, nothing else."""),
    ("human", "{query}"),
])


def orchestrator_agent(state: AgentState) -> AgentState:
    """Routes the query to the appropriate specialist agent."""
    llm = get_llm()
    chain = INTENT_PROMPT | llm

    result = chain.invoke({"query": state["query"]})
    intent = result.content.strip().lower()

    if intent not in {"diagnosis", "rag", "drug", "report", "safety"}:
        intent = "rag"  # default fallback

    return {**state, "intent": intent}
