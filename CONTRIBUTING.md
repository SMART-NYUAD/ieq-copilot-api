# Contributing Guide

## Development Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements-dev.txt`
3. Copy env template:
   - `cp .env.example .env`
4. Configure local services (Postgres + Ollama) and database access in `.env` (`DATABASE_URL` or `DB_*`).

## Running the API

- From this directory:
  - `python rag_api_server.py`
- Or explicit host/port:
  - `python rag_api_server.py 8001 0.0.0.0`

## Running Tests

Primary regression suite:

- `python -m unittest discover -s tests -p "test_general_qa_routing.py"`
- `python -m unittest discover -s tests -p "test_stream_route_metadata.py"`
- `python -m unittest discover -s tests -p "test_query_routes_preview.py"`
- `python -m unittest discover -s tests -p "test_llm_router_planner.py"`

Run all tests:

- `python -m unittest discover -s tests -p "test_*.py"`

## Coding Standards

- Keep endpoint handlers thin; business logic belongs in use-case/orchestration modules.
- Use progressive contracts for cross-layer payloads:
  - stable required core
  - optional extension fields in `extras`
- Route all evidence normalization through `evidence/evidence_layer.py`.
- Prefer focused comments for non-obvious behavior (heuristics, fallbacks, boundary handling).

## Pull Request Checklist

- Tests pass for touched areas.
- No lint errors in changed files.
- Public API behavior remains backward compatible unless intentionally changed.
- Docs are updated when behavior/contracts/settings change.
- New environment variables are reflected in `.env.example`.

