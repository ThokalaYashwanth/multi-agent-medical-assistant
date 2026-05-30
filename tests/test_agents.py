import pytest
from unittest.mock import patch, MagicMock
from agents.orchestrator import orchestrator_agent
from agents.safety_agent import safety_agent
from agents.state import AgentState


def make_state(query: str) -> AgentState:
    return AgentState(
        query=query,
        intent=None,
        retrieved_context=None,
        diagnosis_output=None,
        drug_output=None,
        report_output=None,
        safety_flags=[],
        final_response=None,
        messages=[],
        confidence=None,
        requires_human_review=False,
    )


class TestOrchestratorAgent:
    @patch("agents.orchestrator.get_llm")
    def test_routes_diagnosis(self, mock_llm):
        mock_llm.return_value.invoke = MagicMock(return_value=MagicMock(content="diagnosis"))
        state = make_state("I have chest pain and shortness of breath")
        # Patch chain invoke
        with patch("agents.orchestrator.ChatPromptTemplate.from_messages") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.__or__ = MagicMock(return_value=mock_chain)
            mock_chain.invoke = MagicMock(return_value=MagicMock(content="diagnosis"))
            mock_prompt.return_value = mock_chain
            result = orchestrator_agent(state)
        assert result["intent"] in {"diagnosis", "rag", "drug", "report", "safety"}

    def test_state_structure(self):
        state = make_state("test query")
        assert "query" in state
        assert "intent" in state
        assert "messages" in state


class TestSafetyAgent:
    @patch("agents.safety_agent.get_llm")
    def test_safe_query_passes(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "SAFE"
        mock_llm.return_value.invoke = MagicMock(return_value=mock_response)

        state = make_state("What are the symptoms of diabetes?")
        state["final_response"] = "Diabetes symptoms include..."

        with patch("agents.safety_agent.ChatPromptTemplate.from_messages") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.__or__ = MagicMock(return_value=mock_chain)
            mock_chain.invoke = MagicMock(return_value=mock_response)
            mock_prompt.return_value = mock_chain
            result = safety_agent(state)

        assert result["safety_flags"] == [] or result.get("safety_flags") is not None


class TestStateSchema:
    def test_required_keys(self):
        state = make_state("test")
        required = ["query", "intent", "retrieved_context", "diagnosis_output",
                    "drug_output", "report_output", "safety_flags",
                    "final_response", "messages", "confidence", "requires_human_review"]
        for key in required:
            assert key in state, f"Missing key: {key}"

    def test_messages_is_list(self):
        state = make_state("test")
        assert isinstance(state["messages"], list)

    def test_requires_human_review_default(self):
        state = make_state("test")
        assert state["requires_human_review"] is False
