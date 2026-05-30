from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from agents.pipeline import run_pipeline

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    history: Optional[List[dict]] = []


class QueryResponse(BaseModel):
    response: str
    intent: str
    confidence: Optional[float]
    requires_human_review: bool
    safety_flags: List[str]
    session_id: Optional[str]


@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Main endpoint: run the multi-agent pipeline on a medical query."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        state = run_pipeline(query=request.query, history=request.history or [])
        return QueryResponse(
            response=state.get("final_response", "I was unable to generate a response."),
            intent=state.get("intent", "unknown"),
            confidence=state.get("confidence"),
            requires_human_review=state.get("requires_human_review", False),
            safety_flags=state.get("safety_flags", []),
            session_id=request.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health():
    return {"status": "healthy"}
