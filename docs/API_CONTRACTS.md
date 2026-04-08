# API Contracts

This document defines stable API behavior for clients integrating with the RAG API server.

## Contract Philosophy

- Keep a stable core response shape.
- Add optional fields incrementally for new capabilities.
- Preserve backward compatibility unless a breaking change is explicitly versioned.

## `POST /query`

### Request

```json
{
  "question": "What is current CO2 in smart_lab?",
  "k": 5,
  "lab_name": "smart_lab",
  "allow_clarify": true,
  "conversation_id": "optional_conversation_id"
}
```

### Response (core shape)

```json
{
  "answer": "string",
  "timescale": "string",
  "cards_retrieved": 0,
  "recent_card": false,
  "conversation_id": "string_or_null",
  "turn_index": 1,
  "metadata": {},
  "visualization_type": "none|line|bar|scatter",
  "chart": {}
}
```

### Metadata (typical fields)

- routing: `route_source`, `route_type`, `intent_category`, `route_confidence`, `route_reason`
- planner: `planner_model`, `planner_fallback_used`, `planner_fallback_reason`
- execution: `executor`, `execution_intent`, `intent_rerouted_to_db`
- context: `query_signals`, `query_scope_class`, `k_requested`, `lab_name`, `resolved_lab_name`
- analysis: `time_window`, `forecast_*`, `correlation`
- evidence: normalized `evidence` object
- conversation: `conversation_id`, `conversation_context_applied`, `turn_index`

## `POST /query/stream` (SSE)

### Event types

- `meta`: execution + routing metadata
- `token`: streamed answer chunks
- `conversation`: optional persisted turn metadata
- `done`: completion marker
- `error`: error payload

`meta` contract note:

- Stream `meta` is produced by shared builders in `metadata_builders.py` and
  evidence-normalized through `evidence/evidence_layer.py`, matching non-stream
  contract semantics.

### `error` payload

```json
{
  "detail": "string",
  "code": "invalid_input|routing_error|execution_error|stream_error|internal_error",
  "scope": "string"
}
```

## `POST /v1/chat/completions` (OpenAI-compatible)

### Non-stream

- Returns OpenAI-like `chat.completion`.
- Includes:
  - `x_router` for route metadata
  - `x_visualization_type`
  - `x_chart`

### Stream

- Returns OpenAI-like `chat.completion.chunk` events.
- Ends with `data: [DONE]`.
- Errors include OpenAI-style `error` object with stable `code` where available.
- First chunk `x_router` uses the same shared stream metadata/evidence
  normalization path as native `POST /query/stream`.

## `GET /health/router` (router safety telemetry)

Returns active router mode, runtime observability metrics, and SLO gate status.

Key groups:

- `router_mode`: active strategy (`policy_engine_only`)
- `metrics`: planner/critic counters and sync/stream parity rates
- `thresholds`: target/max thresholds used for router safety gating
- `slo`: computed booleans (`*_target_ok`, `*_max_ok`, `rollout_blocked`)

## Evidence Contract

Evidence is normalized through `evidence/evidence_layer.py` and validated against schema contracts.

Core evidence fields:

- `evidence_version`
- `evidence_kind`
- `intent`
- `strategy`
- `metric_aliases`
- `resolved_scope`
- `resolved_time_window`
- `provenance_sources`
- `confidence_notes`
- `recommendation_allowed`

## Backward Compatibility Rules

- Do not remove existing top-level response fields without versioning.
- Optional metadata additions are allowed.
- Keep semantic meaning of existing fields stable across releases.

