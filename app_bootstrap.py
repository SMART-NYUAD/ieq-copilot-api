"""FastAPI application bootstrap with organized route registration."""

import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from core_settings import load_settings
    from http_routes.health_routes import router as health_router
    from http_routes.openai_compat_routes import router as openai_compat_router
    from http_routes.query_routes import router as query_router
    from query_routing.observability import record_http_request_end, record_http_request_start
except ImportError:
    from .core_settings import load_settings
    from .http_routes.health_routes import router as health_router
    from .http_routes.openai_compat_routes import router as openai_compat_router
    from .http_routes.query_routes import router as query_router
    from .query_routing.observability import record_http_request_end, record_http_request_start


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = load_settings()
    app = FastAPI(
        title="Environment Cards RAG API",
        description="API for routed DB indoor air quality queries with knowledge-card grounding",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    @app.middleware("http")
    async def observability_http_middleware(request, call_next):
        start = time.perf_counter()
        method = str(request.method or "UNKNOWN")
        path = str(request.url.path or "/")
        record_http_request_start()
        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start) * 1000.0
            record_http_request_end(
                method=method,
                path=path,
                status_code=int(getattr(response, "status_code", 500)),
                duration_ms=duration_ms,
            )
            return response
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000.0
            record_http_request_end(
                method=method,
                path=path,
                status_code=500,
                duration_ms=duration_ms,
            )
            raise

    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(openai_compat_router)
    return app


app = create_app()

