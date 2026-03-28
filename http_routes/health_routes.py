"""Health and root endpoints."""

from fastapi import APIRouter


router = APIRouter()


@router.get("/")
async def root():
    return {
        "message": "Environment Cards RAG API",
        "version": "2.0.0",
        "endpoints": {
            "query": "POST /query - Routed query endpoint",
            "query_stream": "POST /query/stream - Routed streaming query",
            "query_route": "POST /query/route - Intent router preview",
            "health": "GET /health - Health check",
        },
    }


@router.get("/health")
async def health():
    return {"status": "healthy"}

