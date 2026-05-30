from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.orchestrator import orchestrator_agent
from agents.rag_agent import rag_agent
from agents.diagnosis_agent import diagnosis_agent
from agents.drug_agent import drug_agent
from agents.report_agent import report_agent
from agents.safety_agent import safety_agent


def route_by_intent(state: AgentState) -> str:
    """Conditional edge: route to specialist based on detected intent."""
    intent = state.get("intent", "rag")
    routing = {
        "diagnosis": "diagnosis",
        "rag": "rag",
        "drug": "drug",
        "report": "report",
        "safety": "safety",
    }
    return routing.get(intent, "rag")


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("orchestrator", orchestrator_agent)
    graph.add_node("rag", rag_agent)
    graph.add_node("diagnosis", diagnosis_agent)
    graph.add_node("drug", drug_agent)
    graph.add_node("report", report_agent)
    graph.add_node("safety", safety_agent)

    # Entry point
    graph.set_entry_point("orchestrator")

    # Conditional routing after orchestrator
    graph.add_conditional_edges(
        "orchestrator",
        route_by_intent,
        {
            "rag": "rag",
            "diagnosis": "diagnosis",
            "drug": "drug",
            "report": "report",
            "safety": "safety",
        },
    )

    # All specialist agents pass through safety check before ending
    for node in ["rag", "diagnosis", "drug", "report"]:
        graph.add_edge(node, "safety")

    graph.add_edge("safety", END)

    return graph.compile()


# Compiled graph singleton
pipeline = build_graph()


def run_pipeline(query: str, history: list = None) -> AgentState:
    """Run the full multi-agent pipeline for a query."""
    initial_state: AgentState = {
        "query": query,
        "intent": None,
        "retrieved_context": None,
        "diagnosis_output": None,
        "drug_output": None,
        "report_output": None,
        "safety_flags": [],
        "final_response": None,
        "messages": history or [],
        "confidence": None,
        "requires_human_review": False,
    }
    return pipeline.invoke(initial_state)
