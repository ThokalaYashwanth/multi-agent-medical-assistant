from langchain.prompts import ChatPromptTemplate
from core.llm import get_llm
from agents.state import AgentState

DIAGNOSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical diagnosis assistant. Analyze the described symptoms and provide:
1. Top 3 differential diagnoses with likelihood (High/Medium/Low)
2. Key distinguishing features for each
3. Recommended immediate steps (tests, referrals)
4. Red flag symptoms to watch for

IMPORTANT: Always include this disclaimer: 
"This is an AI-assisted analysis for informational purposes only. 
Consult a licensed physician for actual medical diagnosis and treatment."

Be concise and structured."""),
    ("human", "Patient symptoms: {query}"),
])


def diagnosis_agent(state: AgentState) -> AgentState:
    """Analyzes symptoms and generates differential diagnoses."""
    llm = get_llm()
    chain = DIAGNOSIS_PROMPT | llm
    result = chain.invoke({"query": state["query"]})

    return {
        **state,
        "diagnosis_output": result.content,
        "final_response": result.content,
        "confidence": 0.75,
        "requires_human_review": True,  # Diagnosis always needs human review
    }
