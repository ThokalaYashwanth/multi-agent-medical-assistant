import re
from langchain.prompts import ChatPromptTemplate
from core.llm import get_llm
from agents.state import AgentState

REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical documentation specialist. Generate a structured SOAP note from 
the provided clinical information.

Format:
**S (Subjective):** Patient-reported symptoms and history
**O (Objective):** Observable findings, vitals, test results
**A (Assessment):** Clinical impression and diagnoses (with ICD-10 codes where possible)
**P (Plan):** Treatment plan, medications, follow-up, referrals

Extract ICD-10 codes where applicable. Be precise and use clinical terminology."""),
    ("human", "Clinical notes: {query}"),
])


def report_agent(state: AgentState) -> AgentState:
    """Generates SOAP notes and clinical summaries."""
    llm = get_llm()
    chain = REPORT_PROMPT | llm
    result = chain.invoke({"query": state["query"]})

    # Extract ICD-10 codes from response
    icd_pattern = r'\b[A-Z]\d{2}(?:\.\d{1,4})?\b'
    icd_codes = re.findall(icd_pattern, result.content)

    response = result.content
    if icd_codes:
        response += f"\n\n**Extracted ICD-10 Codes:** {', '.join(set(icd_codes))}"

    return {
        **state,
        "report_output": response,
        "final_response": response,
        "confidence": 0.90,
    }
