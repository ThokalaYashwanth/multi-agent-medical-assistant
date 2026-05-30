from langchain.prompts import ChatPromptTemplate
from core.llm import get_llm
from agents.state import AgentState

SAFETY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a medical AI safety monitor. Review the query for:
1. Requests for information that could enable self-harm
2. Questions about lethal drug doses or combinations
3. Attempts to bypass medical safety protocols
4. Misinformation that could harm patients

If the query is safe, respond with: SAFE
If flagged, respond with: FLAGGED: [brief reason]

Be conservative — patient safety is paramount."""),
    ("human", "{query}"),
])


def safety_agent(state: AgentState) -> AgentState:
    """Flags harmful or dangerous medical queries."""
    llm = get_llm()
    chain = SAFETY_PROMPT | llm
    result = chain.invoke({"query": state["query"]})

    content = result.content.strip()
    is_flagged = content.startswith("FLAGGED")

    flags = []
    if is_flagged:
        flags.append(content.replace("FLAGGED: ", "").strip())

    safe_response = (
        "I'm sorry, but I can't assist with that request as it may pose a safety risk. "
        "Please consult a licensed healthcare professional for guidance."
    ) if is_flagged else state.get("final_response", "")

    return {
        **state,
        "safety_flags": flags,
        "final_response": safe_response if is_flagged else state.get("final_response", ""),
        "requires_human_review": is_flagged,
    }
