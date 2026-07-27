# API Contracts

Stable request/response behavior for clients integrating with the RAG API server.
`CLAUDE.md` describes the internal architecture; this document is the external contract.

## Contract philosophy

- Keep a stable core response shape.
- Add optional fields incrementally for new capabilities.
- Preserve backward compatibility unless a breaking change is explicitly versioned.
- The sync body and the stream frames carry the same information — anything present in one
  is reachable from the other.

## Authentication

When `RAG_API_KEYS` is set, every endpoint below requires `X-API-Key: <key>` or
`Authorization: Bearer <key>`. When it is unset, the endpoints are open and all callers
share a single conversation namespace (development only).

| Status | Meaning |
|---|---|
| `400` | empty question |
| `401` | missing or invalid API key |
| `403` | `conversation_id` belongs to a different caller |
| `500` | internal execution error |

## `POST /query`

### Request

```json
{
  "question": "What is the current CO2 in smart_lab?",
  "k": 5,
  "lab_name": "smart_lab",
  "allow_clarify": true,
  "conversation_id": "optional_conversation_id"
}
```

- `k` — retrieval depth for the knowledge path (default 5, clamped to 1–8).
- `lab_name` — optional space hint; the question itself may also name one.
- `allow_clarify` — when `false`, a question the router would ask about is answered on a
  best-effort basis instead (default `true`).
- `conversation_id` — echo the value from a previous response to continue that
  conversation. Omit it to start a new one; a fresh id is returned.

### Response

```json
{
  "answer": "string",
  "timescale": "string",
  "cards_retrieved": 0,
  "recent_card": false,
  "conversation_id": "string_or_null",
  "turn_index": 1,
  "metadata": {},
  "footnotes": [],
  "citation_sources": []
}
```

- `citation_sources` — every guideline source offered to the model, each with a stable
  `index`.
- `footnotes` — the subset the answer actually cited, in order of appearance. Inline `[N]`
  markers in `answer` refer to these `index` values.

### `metadata`

| Field | Meaning |
|---|---|
| `executor` | branch that answered: `db_query`, `knowledge_qa`, `ifc_qa`, `sensor_inspection`, `viewer_control`, `heatmap_control`, `download_data`, `clarify_gate`, `guardrail` |
| `intent` | router intent (see `CLAUDE.md` for the taxonomy) |
| `route_confidence`, `planner_model`, `fallback_used` | routing provenance; `fallback_used` means the router LLM was unreachable and a regex plan was used |
| `resolved_question` | present only when prior turns were used to rewrite the question |
| `ui` | frontend contract; see below |
| `llm_used` | `false` when the answer came from a deterministic fallback |
| `lab_name`, `resolved_lab_name`, `time_window` | resolved scope (DB path) |
| `conversation_id`, `conversation_context_applied`, `turn_index` | conversation state |

### `metadata.ui`

Data branches carry `mode` (`status`, `analytical`, `conversational`, `clarify`), `panel`,
`primary_metric`, `metrics`, and `transition`.

Control branches carry their own action fields instead:

- `viewer_control`: `viewer_type` ∈ `splat` | `ifc` | `pc` | `pano`
- `heatmap_control`: `heatmap_action` ∈ `on` | `off`, `heatmap_metric`
- `download_data`: `download_needs_metric`, and when `false`, `download_slug`,
  `download_metric_type`, `download_start`, `download_end`, `download_interval`,
  `download_format`

## `POST /query/stream` (SSE)

Same request body. Frames arrive in this order:

| Event | Payload | Notes |
|---|---|---|
| `status` | `stage`, `message` | progress only; safe to ignore |
| `meta` | same fields as `metadata`, plus `timescale`, `cards_retrieved`, `recent_card`, `visualization_type`, `chart` | `citation_sources`/`footnotes` are empty here — see `sources` |
| `meta_update` | `timescale`, `time_window`, `resolved_lab_name`, `metrics_used`, `ui` | DB path only: supersedes the placeholder `meta` once the query resolved |
| `token` | `text` | append in order to build the answer |
| `sources` | `citation_sources`, `footnotes` | emitted after the last token, before `done` |
| `done` | — | terminal frame |
| `error` | `detail`, `code`, `scope` | emitted on failure, always followed by `done` |

The stream cannot know which `[N]` markers the answer used until generation finishes, which
is why citations arrive at the end rather than in `meta`.

A stream always terminates with `done`. If the answer model is unreachable, the executor
emits a deterministic grounded answer rather than an empty stream.

## `GET /sensors/latest/{space}`

Latest reading snapshot for one space.

## `GET /ifc/summary`

Parsed structured summary of the IFC building model (units, storeys, element counts,
materials).

## `GET /health`

`{"status": "healthy"}`. Unauthenticated, for liveness probes.
