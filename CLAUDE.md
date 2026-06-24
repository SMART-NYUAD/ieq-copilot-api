# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run (preferred, with hot reload):**
```bash
docker compose up --build
docker compose down
docker compose logs -f rag-api
```

**Run directly:**
```bash
python rag_api_server.py 8001 0.0.0.0
```

**Install dependencies:**
```bash
pip install -r requirements-dev.txt
```

**Run all tests:**
```bash
python -m unittest discover -s tests -p "test_*.py"
```

**Run a single test file:**
```bash
python -m unittest discover -s tests -p "test_llm_router_planner.py"
```

**Primary regression suite:**
```bash
python -m unittest discover -s tests -p "test_general_qa_routing.py"
python -m unittest discover -s tests -p "test_stream_route_metadata.py"
python -m unittest discover -s tests -p "test_query_routes_preview.py"
python -m unittest discover -s tests -p "test_llm_router_planner.py"
```

## Architecture

Five layers execute in order per request:

1. **API layer** (`http_routes/`) — validates shape, normalizes fields, builds `ConversationContext`
2. **Routing layer** (`query_routing/llm_router_planner.py`) — LLM produces a JSON route plan; policy validation enforces deterministic executor selection
3. **Execution layer** (`query_routing/query_orchestrator.py`, `executors/`) — runs exactly one branch: `clarify_gate`, `knowledge_qa`, or `db_query`
4. **Evidence layer** (`evidence/evidence_layer.py`) — normalizes and repairs provenance envelopes before response mapping
5. **Response layer** (`http_routes/`) — emits contract-stable sync JSON or SSE stream

### Request flow

```
POST /query or /query/stream
  → http_routes/query_routes.py          (input validation + ConversationContext)
  → query_routing/llm_router_planner.py  (LLM call → RoutePlan)
  → query_routing/query_orchestrator.py  (executor selection + branch execution)
      ├── executors/knowledge_executor.py    (definition/general questions)
      └── executors/db_query_executor.py     (all data questions: lookup/aggregation/comparison/anomaly/forecast)
  → evidence/evidence_layer.py           (normalize provenance)
  → response assembly + turn persistence
```

### ConversationContext

`build_conversation_context()` in `storage/conversation_context.py` is called once at the HTTP boundary and produces a frozen dataclass containing every view downstream components need: `effective_question`, `effective_lab`, `routing_snippet` (for the router LLM), `llm_history` (for the answer LLM), and carry-over slots (`carried_metric`, `carried_time_phrase`). No downstream layer reconstructs context independently — they all read from this single object. Carry-over values are passed as structured `planner_hints` to executors; they are **never appended to `effective_question`**.

### Routing

The router (`llm_router_planner.py`) sends a structured JSON prompt to an Ollama LLM and parses the response into a `RoutePlan`. The LLM returns exactly one intent from a fixed taxonomy. Regex fallback (`_fallback_plan`) is **emergency-only** — used only when the LLM is unreachable, and it covers only unambiguous structural keywords. The intent taxonomy:

- `definition_explanation` → knowledge executor
- `unknown_fallback` → knowledge executor
- `current_status_db`, `point_lookup_db`, `aggregation_db`, `comparison_db`, `anomaly_analysis_db`, `forecast_db` → DB executor
- `viewer_control` → viewer-control branch (opens a 3D view; `viewer_type` ∈ splat/ifc/pc/pano)
- `heatmap_control` → heatmap-control branch (toggles the heatmap overlay and selects its metric; `ui.heatmap_action` ∈ on/off, `ui.heatmap_metric` ∈ temperature/humidity/voc/pm25/null)
- `download_data` → download-control branch (hands the frontend a ready-to-use CSV/JSON export URL; `ui.download_url`, `ui.download_format` ∈ csv/json, `ui.download_type` ∈ aggregated/raw, `ui.download_start`/`ui.download_end` ISO-8601)
- `ifc_model_qa` → IFC executor (questions *about* the BIM/IFC building model)

Note the deliberate split between `viewer_control` and `ifc_model_qa`: *"open the IFC view"* is a UI action (`viewer_control`), while *"how many columns does the building have?"* is a question answered from the model (`ifc_model_qa`). `heatmap_control` is a sibling UI meta-action: *"turn on the heatmap and use temperature"* emits a `meta` event with action + metric and returns no sensor values. `download_data` is another sibling: *"export the last 7 days as CSV"* resolves the time window server-side (via `extract_time_window`, mirroring the DB path so the frontend never reconstructs date ranges) and emits a `meta` event carrying a fully-built `download_url` for the `DOWNLOAD_BASE_URL` endpoint (default `api.smart-crg.com/download/sensor-readings`, sensor `DOWNLOAD_SENSOR_ALIAS`) — the frontend renders a button, not an auto-download. It is distinct from `aggregation_db`: *"what was the average CO2 last week?"* is a data question (DB path), *"download last week's data"* is `download_data`. Like `viewer_control`, all three branches short-circuit before the DB/evidence layers and are LLM-routed (no regex; only a non-regex alias fallback fills a field the LLM omits).

### IFC executor

`executors/ifc_executor.py` answers questions about the BIM/IFC building model. `ifc_model/ifc_store.py` is a dependency-free STEP/ISO-10303-21 parser (no `ifcopenshell`) that extracts grounded facts — units, spatial hierarchy, storey elevations, an element inventory, per-element dimensions/properties, and materials — from the IFC file (default `smart.ifc`, override with `IFC_MODEL_PATH`). Parsed facts are cached by file mtime+size. The executor feeds those facts to the answer LLM with a strict "answer only from the model, never fabricate" directive; if the LLM is unreachable it returns a deterministic summary built from the parsed counts.

`ifc_model/ifc_geometry.py` resolves geometry into **world space** for measurements: it composes each element's `IfcLocalPlacement` chain (and `IfcMappedItem`/`IfcRepresentationMap` transforms) and projects BREP vertices and extruded-solid profiles to compute the overall world-coordinate bounding box (`dimensions`) and slab floor-plate polygon areas. From those, `ifc_store` derives **architectural metrics** (`architectural_metrics`): Gross Internal Area (GIA ≈ sum of floor-plate areas), footprint area, perimeter, floor-to-floor height, gross internal volume, wall thickness, and storey/envelope counts. A naive bounding box over raw points is deliberately avoided (it mixes local and world coordinates). NIA is not computed and is reported as such rather than guessed — every figure is grounded in resolved geometry or IFC attributes/properties. Computing world geometry requires retaining all entities during the parse, so the first `ifc_model_qa` call takes a few seconds; results are cached per process.

### DB executor

`executors/db_query_executor.py` always runs SQL first, then passes structured rows to the LLM for narrative rendering. If the LLM fails, a deterministic text fallback is returned. `metadata.llm_used` indicates whether LLM rendering succeeded. Forecasts use Meta Prophet; the LLM only explains, never invents future values.

### Key invariants

- Endpoint handlers stay thin — all business logic lives in orchestration/executor modules.
- All evidence normalization goes through `evidence/evidence_layer.py`.
- Stream (`/query/stream`) and sync (`/query`) metadata share the same builders in `query_routing/metadata_builders.py` — keep them in sync.
- `ConversationContext` is immutable (`frozen=True`) and built once per turn.

## Routing Preference

**Prefer LLM-based routing over regex.** Regex is a last resort (emergency fallback only). When improving or extending routing behavior, the right approach is to tune the system prompt in `llm_router_planner.py::_SYSTEM_PROMPT` and adjust intent definitions and examples — not to add regex patterns. The LLM handles ambiguity, typos, and natural language variation far better than keyword matching. Regex rules exist only in `_fallback_plan()` to cover the case where the Ollama endpoint is completely unreachable.

## Configuration

All runtime settings come from `.env` (auto-loaded by `core_settings.py`) and the process environment. Shell env vars win over `.env` values. Key variables:

| Variable | Purpose |
|---|---|
| `OLLAMA_ROUTER_MODEL` | LLM used for intent routing (default: `qwen3:30b-a3b-instruct-2507-q4_K_M`) |
| `OLLAMA_ROUTER_BASE_URL` | Ollama endpoint for the router |
| `OLLAMA_MODEL` / `OLLAMA_BASE_URL` | Separate model/endpoint for answer generation |
| `DATABASE_URL` | Postgres connection (or use `DB_*` components) |
| `ROUTER_CLARIFY_THRESHOLD` | Confidence below which queries trigger clarification |
| `IFC_MODEL_PATH` | Path to the IFC building model for `ifc_model_qa` (default: `./smart.ifc`) |
| `DOWNLOAD_BASE_URL` | Sensor-readings export endpoint for `download_data` (default: `https://api.smart-crg.com/download/sensor-readings`) |
| `DOWNLOAD_SENSOR_ALIAS` | Sensor alias passed to the download endpoint (default: `Atmocube Sensor 02`) |

Two distinct Ollama models are configured separately: one for routing (`OLLAMA_ROUTER_*`) and one for answer generation (`OLLAMA_*`). Keep these separate — the router runs at temperature 0.0 with constrained output; the answer model has different latency/quality tradeoffs.

## Debugging

- `GET /health/router` — router/SLO status
- `POST /query/route` — classify a question without executing (shows `route_type`, `route_confidence`, `planner_model`, etc.)
- `POST /query/db-proof` — SQL preview + row sample for DB-path queries
- `GET /observability/kpis` — fallback rates, latency, error trends
- `GET /ifc/summary` — parsed structured summary of the IFC building model (units, storeys, element counts, materials)
