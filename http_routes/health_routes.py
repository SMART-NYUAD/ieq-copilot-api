"""Health and root endpoints."""

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from executors.sensors_endpoint import get_sensor_latest
from http_routes.auth import require_api_key

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
            "roles": "GET /roles - Stakeholder roles accepted by /query",
        },
    }


@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.get("/roles", dependencies=[Depends(require_api_key)])
async def roles():
    """The stakeholder roles ``/query`` accepts, for a client to render a selector from.

    Served rather than documented-only so a frontend does not hardcode a closed vocabulary
    that then drifts away from ``prompting/roles.py``.
    """
    from core_settings import default_stakeholder_role
    from prompting.roles import role_catalog

    return {"roles": role_catalog(), "default": default_stakeholder_role()}


@router.get("/sensors/latest/{space}", dependencies=[Depends(require_api_key)])
async def sensors_latest(space: str):
    return await run_in_threadpool(get_sensor_latest, space)


@router.get("/ifc/summary", dependencies=[Depends(require_api_key)])
async def ifc_summary():
    """Debug: parsed structured summary of the IFC building model."""
    from core_settings import ifc_model_path
    from ifc_model.ifc_store import get_ifc_summary

    def _load():
        try:
            return {"available": True, **get_ifc_summary(ifc_model_path())}
        except FileNotFoundError:
            return {"available": False, "error": "IFC model file not found"}

    return await run_in_threadpool(_load)
