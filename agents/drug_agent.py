from langchain.prompts import ChatPromptTemplate
from core.llm import get_llm
from agents.state import AgentState

DRUG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical pharmacology assistant. Provide accurate information about:
- Drug interactions (severity: Major/Moderate/Minor)
- Contraindications
- Standard dosage ranges
- Common and serious side effects

Structure your response clearly with headers. 
Always remind users to consult a pharmacist or physician before making medication decisions."""),
    ("human", "{query}"),
])


def drug_agent(state: AgentState) -> AgentState:
    """Handles drug interaction and dosage queries."""
    llm = get_llm()
    chain = DRUG_PROMPT | llm
    result = chain.invoke({"query": state["query"]})

    return {
        **state,
        "drug_output": result.content,
        "final_response": result.content,
        "confidence": 0.82,
    }
