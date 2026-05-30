from typing import TypedDict, List, Optional, Annotated
import operator


class AgentState(TypedDict):
    """Shared state passed between all agents in the graph."""
    query: str
    intent: Optional[str]                    # detected intent: diagnosis | rag | drug | report | safety
    retrieved_context: Optional[str]          # RAG agent output
    diagnosis_output: Optional[str]           # Diagnosis agent output
    drug_output: Optional[str]               # Drug agent output
    report_output: Optional[str]             # Report agent output
    safety_flags: Optional[List[str]]        # Safety agent warnings
    final_response: Optional[str]            # Final assembled response
    messages: Annotated[List[dict], operator.add]  # Full conversation history
    confidence: Optional[float]              # Confidence score 0-1
    requires_human_review: bool              # Human-in-the-loop flag
