from fastapi import APIRouter, Request
from api.api.models import RAGRequest, RAGResponse
from api.agends.retrieval_generation import rag_pipeline
import logging

logger = logging.getLogger(__name__)

api_router = APIRouter()

@api_router.post("/rag/", tags=["RAG"])
def rag(
    request: Request,
    payload: RAGRequest
) -> RAGResponse:
    
    answers = rag_pipeline(question=payload.query)

    return RAGResponse(
        request_id=request.state.request_id,
        answers=answers
    )    