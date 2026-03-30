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
- Contributor workflow: `CONTRIBUTING.md`
- Release process: `RELEASE_CHECKLIST.md`
- API contracts: `docs/API_CONTRACTS.md`
- Blueprint guide: `docs/BLUEPRINT_GUIDE.md`

## Architecture

Main modules:

- `rag_api_server.py`: runtime entrypoint
- `app_bootstrap.py`: FastAPI app + route registration
- `core_settings.py`: centralized runtime settings (server, CORS, routing thresholds)
- `http_routes/`: HTTP endpoints
  - `health_routes.py`
  - `query_routes.py`
- `query_routing/`: intent routing + orchestration
  - `intent_classifier.py` (legacy fallback classifier)
  - `llm_router_planner.py`
  - `route_policy_engine.py`
  - `query_orchestrator.py`
  - `query_use_cases.py` (branch-specific application use cases)
- `http_routes/query_runtime.py`: shared runtime adapters used by both native and OpenAI-compatible routes
- `http_routes/route_helpers.py`: route metadata helpers + conversation persistence hooks
- `executors/`: execution engines
  - `db_query_executor.py` (SQL + LLM answer rendering)
- `executors/env_query_langchain.py`: knowledge-card retrieval + shared LLM chain utilities
- `evidence/evidence_layer.py`: explicit evidence normalization/repair layer
- `contracts/progressive_contracts.py`: progressive contracts (stable core + extensible fields)
- `storage/postgres_client.py`: shared DB cursor/connection helper

Detailed router design is documented in `docs/router_architecture.md`.
End-to-end request lifecycle (with sequence/state graphs) is documented in
`docs/architecture_deep_dive.md`.

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

## Request Flow

1. Client calls `POST /query` (or `POST /query/stream`).
2. Router plans intent via `llm_router_planner.py` (with legacy fallback to `intent_classifier.py`).
3. Orchestrator executes through DB or knowledge executors.
4. Executor provenance is normalized by `evidence/evidence_layer.py`.
5. For DB intents, SQL rows are converted to a grounded LLM answer (with deterministic fallback).
6. Unified response is returned with route and evidence metadata.

For a very detailed step-by-step flow (including clarify-gate behavior,
follow-up memory carry-over, policy rollout/shadow mode, and streaming internals),
see `docs/architecture_deep_dive.md`.

## Documentation Map

- `docs/architecture_deep_dive.md`: full architecture walkthrough with Mermaid graphs
- `docs/router_architecture.md`: routing policy, planner contract, and metadata details
- `docs/API_CONTRACTS.md`: request/response contracts and compatibility payloads
- `docs/BLUEPRINT_GUIDE.md`: implementation and blueprint guidance

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

- **Deterministic forecasting (Meta Prophet)**
  - Questions like `Forecast PM2.5 for next week in smart_lab` route to the DB executor.
  - Backend uses Meta Prophet to generate a short- to medium-horizon forecast.
  - The LLM only explains the forecast; it never invents future values.
  - `/query` and `/query/stream` responses include:
    - `metadata.forecast_model`, `metadata.forecast_confidence`, `metadata.forecast_horizon_hours`
    - A line chart with both **history** and **prediction** series.

- **Smarter lab resolution**
  - Lab names are discovered from the `app_lab` table (`name` column), not hardcoded.
  - Handles variants like `smart_lab`, `smart lab`, or just `smart` when unambiguous.
  - Comparison questions with two lab-like names (for example, `shores_office and concrete_lab`) automatically route to `comparison_db`.

- **Safer numeric explanations**
  - DB executor always runs SQL first, then passes structured rows + optional forecast to the LLM.
  - If the LLM fails or times out, a deterministic text fallback is returned.
  - Forecasts are clearly labeled with confidence and are never extrapolated by the LLM itself.

## Run the API

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
python -m unittest discover -s tests -p "test_general_qa_routing.py"
python -m unittest discover -s tests -p "test_stream_route_metadata.py"
python -m unittest discover -s tests -p "test_query_routes_preview.py"
python -m unittest discover -s tests -p "test_llm_router_planner.py"
```

All tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Endpoints

### `GET /`

Returns service info and endpoint list.

### `GET /health`

Basic health check.

### `POST /query/route`

Preview only: classify a question without executing query.

Request:

```json
{
  "question": "Compare smart_lab vs concrete_lab CO2 in the last 24 hours",
  "lab_name": "smart_lab"
}
```

Response:

```json
{
  "route_source": "llm_planner",
  "route_type": "comparison_db",
  "intent_category": "analytical_visualization",
  "route_confidence": 0.9,
  "route_reason": "comparison_keyword",
  "planner_model": "qwen3:30b",
  "planner_fallback_used": false
}
```

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
- `done`: completion marker
- `error`: error payload

Example:

```bash
curl -N -X POST "http://127.0.0.1:8001/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"question":"Show the trend of CO2 in smart_lab over time","k":3,"lab_name":"smart_lab"}'
```

### `GET /v1/models`

OpenAI-compatible model listing endpoint.

Example:

```bash
curl "http://127.0.0.1:8001/v1/models"
```

### `POST /v1/chat/completions` (OpenAI-compatible)

OpenAI-style chat endpoint that routes internally through the same `/query` logic.

Supported fields:

- `model` (optional, default `rag-router`)
- `messages` (required, uses last `role=user` message as query)
- `stream` (optional, default `false`)
- `temperature`, `max_tokens`, `user`, `metadata` (accepted for compatibility)
- `k` (optional, extension field for retrieval depth)
- `lab_name` (optional, extension field for space filter)

Non-stream example:

```bash
curl -X POST "http://127.0.0.1:8001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rag-router",
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "What is the current CO2 in smart_lab?"}
    ],
    "stream": false,
    "k": 3,
    "lab_name": "smart_lab"
  }'
```

Stream example:

```bash
curl -N -X POST "http://127.0.0.1:8001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rag-router",
    "messages": [
      {"role": "user", "content": "Find anomalies in CO2 in smart_lab"}
    ],
    "stream": true,
    "k": 3,
    "lab_name": "smart_lab"
  }'
```

OpenAI compatibility notes:

- Returns OpenAI-like objects:
  - non-stream: `chat.completion`
  - stream: `chat.completion.chunk` SSE + `[DONE]`
- Includes `x_router` metadata in non-stream responses so route/debug info is preserved.

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
- `500`: internal execution error (DB/LLM/runtime)

Runtime reliability notes:

- Non-streaming route execution is offloaded via threadpool to reduce event-loop blocking.
- Streaming errors now include stable error codes (for example, `execution_error`, `stream_error`) in payload metadata.

## Notes

- Timescale is currently fixed to `1hour`.
- CORS is open (`allow_origins=["*"]`) in current server config.
- DB credentials and model endpoints come from this folder’s `.env` and the process environment.

