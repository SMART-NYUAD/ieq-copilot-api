# RAG API Server

FastAPI server for indoor air quality Q&A with **intent routing**:

- **DB path** for all routed questions (including former semantic/trend/anomaly card intents)
- **Knowledge-card grounding** for interpretation and guardrail context

The API keeps a single `/query` contract and decides the executor internally.

## Repository Readiness

This folder is structured to be standalone-repo friendly.

- Runtime dependencies: `requirements.txt`
- Dev/test dependencies: `requirements-dev.txt`
- Environment template: `.env.example`
- Container runtime: `Dockerfile`, `docker-compose.yml`
- Contributor workflow: `CONTRIBUTING.md`
- Architecture reference: `CLAUDE.md`
- API contracts: `docs/API_CONTRACTS.md`

## Architecture

The service uses a layered architecture:

1. **API layer** receives requests (`/query`, `/query/stream`), authenticates the caller,
   and builds the per-turn `ConversationContext`.
2. **Routing layer** turns the question into a `RoutePlan` via a single LLM call.
3. **Execution layer** runs exactly one branch (DB, knowledge, IFC, sensor inspection, or a
   viewer/heatmap/download UI control).
4. **Response layer** returns a contract-stable payload (sync JSON or SSE events).

Main modules by layer:

- `rag_api_server.py`: runtime entrypoint
- `app_bootstrap.py`: FastAPI app + route registration
- `core_settings.py`: centralized runtime settings (server, CORS, API keys, model endpoints)
- `http_routes/`: HTTP endpoints
  - `health_routes.py`, `query_routes.py`
  - `auth.py`: API-key dependency and caller identity
  - `route_helpers.py`: conversation context construction + turn persistence
- `query_routing/`: intent routing + orchestration
  - `intent_classifier.py` (intent taxonomy)
  - `llm_router_planner.py` (LLM route plan + emergency regex fallback)
  - `query_orchestrator.py` (branch execution and payload assembly)
  - `metadata_builders.py` (UI contract shared by sync and stream)
- `executors/`: execution engines
  - `db_query_executor.py` + `db_support/` (sensor data retrieval + LLM answer rendering)
  - `knowledge_executor.py`, `ifc_executor.py`, `sensor_inspection_executor.py`
  - `sse.py`: shared SSE frame builders (`token`, `sources`, `done`)
- `storage/`: conversation store (SQLite, caller-owned), Postgres client, guideline store
- `evidence/citation_processor.py`: numbered citation resolution shared by both paths

Primary architecture reference: `CLAUDE.md`.

## Local Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements-dev.txt
```

3. Copy environment template:

```bash
cp .env.example .env
```

The server now auto-loads `.env` from this folder at runtime (without requiring
manual `export`), and environment variables already set in your shell still win.

4. Ensure local integrations are available:
   - Ollama endpoint for planner/answer models
   - Postgres connectivity used by project modules
   - Database credentials in `.env` as `DATABASE_URL` (or `DB_*` components)

## Docker Development (Hot Reload)

1. Copy environment template if needed:

```bash
cp .env.example .env
```

2. Build and run in development mode:

```bash
docker compose up --build
```

The container runs Uvicorn with `--reload` and bind-mounts this repository into
`/app`, so Python file edits on your host automatically trigger server restart.

Useful commands:

```bash
docker compose down
docker compose logs -f rag-api
```

## Request Flow

1. Client calls `POST /query` (or `POST /query/stream`) with an API key.
2. `build_conversation_context` loads the caller's prior turns and builds the per-turn context.
3. `llm_router_planner.py` plans the intent in one LLM call, resolving follow-up references.
4. Orchestrator executes exactly one branch (DB, knowledge, IFC, sensor, or UI control).
5. For DB intents, retrieved rows are converted to a grounded LLM answer (with a deterministic
   fallback if the answer model is unreachable).
6. Unified response is returned with route metadata and citations; the stream sends citations
   in a terminal `sources` frame.

For architecture and routing behavior details, see `CLAUDE.md`.

## Documentation Map

- `CLAUDE.md`: current architecture, intent taxonomy, configuration, and endpoint list (authoritative)
- `docs/API_CONTRACTS.md`: request/response contracts (partially outdated — see its banner)
- `docs/router_architecture.md`, `docs/architecture_deep_dive.md`, `docs/BLUEPRINT_GUIDE.md`:
  earlier design notes, kept for history; they describe modules and routes that no longer exist

## Intent Types

Router outputs one of:

- `definition_explanation`
- `current_status_db`
- `point_lookup_db`
- `aggregation_db`
- `comparison_db`
- `anomaly_analysis_db`
- `forecast_db`
- `unknown_fallback`

## New Capabilities

- **Forecasting via the predictions API**
  - Questions like `Forecast PM2.5 in smart_lab` route to the DB executor's forecast branch.
  - The forecast itself is fetched from the predictions service (`PREDICTIONS_API_BASE_URL`),
    which serves roughly 6 hours ahead. No forecasting model runs inside this process.
  - The LLM only explains the returned predictions; it never invents future values.
  - When the predictions service is unavailable the branch says so rather than guessing.

- **Smarter lab resolution**
  - Lab names are discovered from the `app_lab` table (`name` column), not hardcoded.
  - Handles variants like `smart_lab`, `smart lab`, or just `smart` when unambiguous.
  - Comparison questions with two lab-like names (for example, `shores_office and concrete_lab`) automatically route to `comparison_db`.

- **Safer numeric explanations**
  - DB executor always fetches the data first (Smart CRG REST API), then passes structured
    rows + optional forecast to the LLM.
  - If the LLM fails or times out, a deterministic text fallback is returned.
  - Forecasts come from the predictions service and are never extrapolated by the LLM itself.

## Run the API

Preferred:

```bash
docker compose up --build
```

From this project directory:

```bash
python rag_api_server.py 8001 0.0.0.0
```

or from any location with an absolute path:

```bash
python /home/smart/RAG_API_SERVER/rag_api_server.py 8001 0.0.0.0
```

Docs UI:

- `http://localhost:8001/docs`

## Run Tests

Targeted regressions:

```bash
python -m unittest discover -s tests -p "test_context_resolution.py"        # follow-up resolution
python -m unittest discover -s tests -p "test_stream_metadata_parity.py"    # sync/stream metadata
python -m unittest discover -s tests -p "test_stream_error_and_sources.py"  # stream error + citations
python -m unittest discover -s tests -p "test_db_default_windows.py"        # time windows + clarify gate
python -m unittest discover -s tests -p "test_api_auth_and_ownership.py"    # auth + conversation ownership
```

Note that `discover -p` silently reports "Ran 0 tests" for a pattern that matches no file,
so check the test count when adding a new pattern.

All tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Endpoints

### `GET /`

Returns service info and endpoint list.

### `GET /health`

Basic health check. Unauthenticated, for liveness probes.

### Authentication

Set `RAG_API_KEYS` to a comma-separated list of keys to protect `/query`, `/query/stream`,
`/sensors/latest/{space}`, and `/ifc/summary`. Send the key as `X-API-Key: <key>` or
`Authorization: Bearer <key>`. When `RAG_API_KEYS` is unset those endpoints are open and all
callers share one conversation namespace — fine locally, not for a deployed service.

Each key owns the conversations it creates: passing a `conversation_id` created under a
different key returns `403`.

To inspect a routing decision without a dedicated preview endpoint, read `metadata` on the
`/query` response (`intent`, `route_confidence`, `planner_model`, `fallback_used`,
`resolved_question`).

### `POST /query`

Main non-streaming query endpoint.

Request body:

- `question` (required)
- `k` (optional, default `5`)
- `lab_name` (optional)

Example:

```bash
curl -X POST "http://127.0.0.1:8001/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the current CO2 in smart_lab?",
    "k": 3,
    "lab_name": "smart_lab"
  }'
```

Response shape:

```json
{
  "answer": "...",
  "timescale": "1hour",
  "cards_retrieved": 0,
  "recent_card": false,
  "metadata": {
    "route_type": "point_lookup_db",
    "route_confidence": 0.8,
    "route_reason": "point_lookup_phrase_with_metric",
    "executor": "db_query",
    "k_requested": 3,
    "lab_name": "smart_lab",
    "llm_used": true,
    "time_window": {
      "label": "last 24 hours",
      "start": "2026-03-01T22:00:00+00:00",
      "end": "2026-03-02T22:00:00+00:00"
    }
  }
}
```

### `POST /query/stream`

SSE streaming query endpoint.

Event types:

- `meta`: route and retrieval metadata
- `token`: streamed text chunks
- `citations`: final list of sources actually cited in answer
- `done`: completion marker
- `error`: error payload

Example:

```bash
curl -N -X POST "http://127.0.0.1:8001/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"question":"Show the trend of CO2 in smart_lab over time","k":3,"lab_name":"smart_lab"}'
```

### Citation Sources in Streaming

Tokens may include inline citation markers like `[1]`, `[2]`. The sources they refer to
arrive in a single `sources` event emitted after the last token and immediately before
`done` — the answer has to be complete before the server knows which markers the model
actually used:

```json
{
  "event": "sources",
  "citation_sources": [
    {
      "index": 1,
      "source_label": "RESET Air Standard v2.1",
      "section_ref": "Section 4: Performance Thresholds",
      "citation_tier": "regulatory",
      "source_url": "https://reset.build/standard/air"
    }
  ],
  "footnotes": [{ "index": 1, "source_label": "RESET Air Standard v2.1" }]
}
```

- `citation_sources`: every source that was offered to the model.
- `footnotes`: only the sources the answer actually cited, in appearance order.

These are the same two fields the sync `/query` response returns in its body, so a client
can render both paths identically. Frontend rendering: replace `[N]` with a superscript
that links to or highlights the matching `index`.

The full SSE event sequence is `status` → `meta` → (`meta_update` on the DB path) →
`token`… → `sources` → `done`. A failure mid-stream emits an `error` event followed by
`done`, so the stream always terminates.

## How DB Time Parsing Works

DB executor parses natural-language windows in questions, including:

- month names: `January`, `Jan`, optional year
- `last week`, `this week`
- weekdays: `Monday`, `last Monday`
- `today`, `yesterday`
- `last/past N hours`
- `last/past N days`

If no time phrase exists, defaults to `last 24 hours`.

## Route Behavior Examples

Expected routing examples:

- Knowledge/guardrail:
  - `What does IEQ mean?` -> `definition_explanation` (knowledge-card path)
  - `What day is today?` -> `definition_explanation` with non-domain guardrail response
- DB:
  - `What is the current CO2 in smart_lab?` -> `point_lookup_db`
  - `What is average humidity in smart_lab?` -> `aggregation_db`
  - `Compare smart_lab vs concrete_lab CO2` -> `comparison_db`

## DB + LLM Behavior

For DB routes:

1. SQL query is executed first.
2. Query result rows are passed to LLM with a grounded prompt.
3. If LLM fails, deterministic fallback answer is returned.
4. `metadata.llm_used` indicates if LLM rendering succeeded.

## Error Handling

Common HTTP status codes:

- `200`: success
- `400`: invalid input (for example, empty question)
- `401`: missing or invalid API key (when `RAG_API_KEYS` is set)
- `403`: `conversation_id` belongs to a different caller
- `500`: internal execution error (DB/LLM/runtime)

Runtime reliability notes:

- Non-streaming route execution is offloaded via threadpool to reduce event-loop blocking;
  streaming paths offload their blocking work (IFC parse, sensor API fetch) the same way.
- A streaming failure emits an `error` event carrying a stable code (for example
  `execution_error`, `stream_error`) followed by `done`, so the client is never left hanging.
- If the answer model is unreachable, each executor emits a deterministic, grounded fallback
  answer instead of failing or returning an empty stream.

## Notes

- Timescale is currently fixed to `1hour`.
- CORS is open (`allow_origins=["*"]`) in current server config.
- DB credentials and model endpoints come from this folder’s `.env` and the process environment.

