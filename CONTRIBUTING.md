# Contributing Guide

## Development Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements-dev.txt`
3. Copy env template:
   - `cp .env.example .env`
4. Configure local services (Postgres + Ollama) and database access in `.env` (`DATABASE_URL` or `DB_*`).
5. Preferred local runtime is Docker Compose with hot reload.

## Running the API

- Preferred (Docker + hot reload):
  - `docker compose up --build`
- Stop Docker runtime:
  - `docker compose down`
- View logs:
  - `docker compose logs -f rag-api`
- Optional direct Python runtime from this directory:
  - `python rag_api_server.py`
- Optional explicit host/port:
  - `python rag_api_server.py 8001 0.0.0.0`

## Running Tests

Run all tests:

- `python -m unittest discover -s tests -p "test_*.py"`

Tests must not reach the network — stub the sensor API with
`tests/fake_sensor_api.FakeSensorApiMixin`. Verify with:

- `python scripts/check_tests_hermetic.py`

## Coding Standards

- Keep endpoint handlers thin; business logic belongs in orchestration/executor modules.
- Add a new executor by writing one branch factory in `query_routing/query_orchestrator.py`.
  There is exactly one intent ladder (`plan_branch`); never add a second one to a renderer.
- Prefer tuning the router prompt over adding regex. Regex is emergency-only (`_fallback_plan`).
- Prefer focused comments for non-obvious behavior (heuristics, fallbacks, boundary handling).

## Pull Request Checklist

- Tests pass for touched areas.
- No lint errors in changed files.
- Public API behavior remains backward compatible unless intentionally changed.
- Docs are updated when behavior/contracts/settings change.
- New environment variables are reflected in `.env.example`.

