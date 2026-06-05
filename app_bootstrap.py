"""FastAPI application bootstrap."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

try:
    from core_settings import load_settings
    from http_routes.health_routes import router as health_router
    from http_routes.query_routes import router as query_router
except ImportError:
    from .core_settings import load_settings
    from .http_routes.health_routes import router as health_router
    from .http_routes.query_routes import router as query_router


@asynccontextmanager
async def _lifespan(app: FastAPI):
    try:
        from executors.db_support.api_client import warm_client

        await run_in_threadpool(warm_client)
    except Exception:
        pass
    yield
    try:
        from executors.db_support.api_client import close_client

        close_client()
    except Exception:
        pass


def create_app() -> FastAPI:
    settings = load_settings()
    app = FastAPI(
        title="Environment Cards RAG API",
        description="API for indoor air quality queries with knowledge-card grounding",
        version="3.0.0",
        lifespan=_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    app.include_router(health_router)
    app.include_router(query_router)
    return app


app = create_app()
