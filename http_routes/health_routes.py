"""Health and root endpoints."""

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from executors.sensors_endpoint import get_sensor_latest

router = APIRouter()


@router.get("/")
async def root():
    return {
        "message": "Environment Cards RAG API",
        "version": "3.0.0",
        "endpoints": {
            "query": "POST /query - Routed query endpoint",
            "query_stream": "POST /query/stream - Routed streaming query",
            "health": "GET /health - Health check",
        },
    }


@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.get("/sensors/latest/{space}")
async def sensors_latest(space: str):
    return await run_in_threadpool(get_sensor_latest, space)
