from fastapi import APIRouter, Request
from api.api.models import RAGRequest, RAGResponse
from api.agends.retrieval_generation import rag_pipeline
from qdrant_client import QdrantClient
import logging

logger = logging.getLogger(__name__)

qdrant_client = QdrantClient(url="http://qdrant:6333")

api_router = APIRouter()

@api_router.post("/rag/", tags=["RAG"])
def rag(
    request: Request,
    payload: RAGRequest
) -> RAGResponse:
    
    answers = rag_pipeline(payload.query, qdrant_client)

    return RAGResponse(
        request_id=request.state.request_id,
        answers=answers
    )    