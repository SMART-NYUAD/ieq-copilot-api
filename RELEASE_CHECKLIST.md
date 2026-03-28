# Release Checklist

## 1) Quality Gates

- [ ] Core regression tests pass:
  - [ ] `test_general_qa_routing.py`
  - [ ] `test_stream_route_metadata.py`
  - [ ] `test_query_routes_preview.py`
  - [ ] `test_llm_router_planner.py`
- [ ] No new lint errors in changed files.
- [ ] Runtime smoke test on `/health`, `/query`, `/query/stream`, `/v1/chat/completions`.

## 2) Contract and Compatibility Review

- [ ] `/query` response fields unchanged (or clearly versioned if changed).
- [ ] SSE event sequencing preserved (`meta` -> `token` -> `done` / `error`).
- [ ] OpenAI-compatible non-stream + stream responses remain valid.
- [ ] Evidence envelope remains valid and normalized by evidence layer.

## 3) Configuration and Security

- [ ] `.env.example` includes all required runtime settings.
- [ ] CORS values are explicit for target environments.
- [ ] Router and model endpoints are configured correctly.
- [ ] No hardcoded credentials or secrets in tracked files.

## 4) Documentation

- [ ] `README.md` setup and run steps are current.
- [ ] `docs/router_architecture.md` matches implemented flow.
- [ ] `docs/API_CONTRACTS.md` reflects actual request/response shapes.
- [ ] Any new modules include concise docstrings.

## 5) Operational Readiness

- [ ] Logging outputs include error scope and code for failures.
- [ ] Observability metrics are present in response metadata where expected.
- [ ] Background start/stop scripts still work for local operation.

