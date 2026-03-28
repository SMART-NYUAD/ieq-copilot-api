# Blueprint Guide for Future Projects

Use this service as a reference architecture for retrieval + routing + tool execution APIs.

## Recommended Layering

1. **Transport layer** (`http_routes/`)
   - request validation
   - response formatting
   - stream framing
2. **Routing + orchestration** (`query_routing/`)
   - intent planning
   - use-case branching
   - critic and observability integration
3. **Executors** (`executors/`)
   - data/tool calls
   - domain-specific query building
4. **Evidence layer** (`evidence/`)
   - normalize provenance
   - validate/repair evidence payload
5. **Contracts + schemas** (`contracts/`, `http_schemas.py`)
   - stable core interfaces
   - progressive extension fields

## Progressive Contract Pattern

- Keep cross-layer contracts strict on required core fields.
- Allow extension via optional fields for fast iteration.
- Version contracts so breaking changes are explicit.

## Porting Checklist

When adapting this blueprint to a new project:

- Replace domain prompt content in `prompting/`.
- Replace DB query logic in `executors/db_query_executor.py`.
- Keep route helper patterns and evidence normalization flow.
- Keep error taxonomy and logging scopes for operability.
- Preserve API contract compatibility unless intentionally versioning.

## Minimum Repo Package

- `requirements.txt`
- `requirements-dev.txt`
- `.env.example`
- `README.md`
- `CONTRIBUTING.md`
- `RELEASE_CHECKLIST.md`
- `docs/API_CONTRACTS.md`
- `docs/router_architecture.md`

