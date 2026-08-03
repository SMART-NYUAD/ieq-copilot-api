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
python -m unittest discover -s tests -p "test_branch_model.py"
```

**Check the suite stays hermetic** (no test may reach the network):
```bash
python scripts/check_tests_hermetic.py
```

**Score the router prompt against the live LLM** (network by design — not part of `discover`):
```bash
python tests/router_eval.py                  # golden router cases + golden conversation
python tests/router_eval.py --router-only    # fast loop while tuning _SYSTEM_PROMPT
python tests/router_eval.py --group context  # one capability: intent|context|clarify|ui-control|diagnostic|injection
```
Run this before and after any `_SYSTEM_PROMPT` change — it is the only thing that measures the
prompt, since the unittest suite is hermetic and stubs the router out. Cases live in
`tests/router_eval_cases.json`; a fix for a routing bug belongs there as a case first.

**Primary regression suite:**
```bash
python -m unittest discover -s tests -p "test_context_resolution.py"        # follow-up resolution
python -m unittest discover -s tests -p "test_stream_metadata_parity.py"    # sync/stream metadata
python -m unittest discover -s tests -p "test_stream_error_and_sources.py"  # stream error + citations
python -m unittest discover -s tests -p "test_branch_model.py"              # branch ladder + renderer parity
python -m unittest discover -s tests -p "test_router_failure_paths.py"      # router transport/parse failures
python -m unittest discover -s tests -p "test_db_default_windows.py"        # time windows
python -m unittest discover -s tests -p "test_api_auth_and_ownership.py"    # auth + conversation ownership
python -m unittest discover -s tests -p "test_stakeholder_roles.py"         # role vocabulary, prompts, widening
```

`discover -p` reports "Ran 0 tests" for a pattern matching no file rather than failing, so
check the reported count when changing these.

## Architecture

Four layers execute in order per request:

1. **API layer** (`http_routes/`) — authenticates the caller, validates shape, builds `ConversationContext`
2. **Routing layer** (`query_routing/llm_router_planner.py`) — one LLM call produces a JSON `RoutePlan` (intent, slots, resolved question, clarification decision)
3. **Execution layer** (`query_routing/query_orchestrator.py`, `executors/`) — `plan_branch` maps the plan to exactly one `Branch`; a renderer runs it
4. **Response layer** (`render_sync` / `render_stream`) — emits contract-stable sync JSON or SSE frames from that same branch

### Request flow

```
POST /query or /query/stream
  → http_routes/query_routes.py          (input validation + ConversationContext)
  → query_routing/llm_router_planner.py  (LLM call → RoutePlan)
  → query_routing/query_orchestrator.py  (plan_branch → one Branch)
      ├── executors/knowledge_executor.py    (definition/general questions)
      ├── executors/db_query_executor.py     (all data questions: lookup/aggregation/comparison/anomaly/forecast)
      ├── executors/ifc_executor.py          (building-model questions)
      ├── executors/sensor_inspection_executor.py  (per-device questions)
      └── (instant branches: viewer / heatmap / download / clarify / guardrail)
  → render_sync(branch) or render_stream(branch)
  → response assembly + turn persistence
```

### ConversationContext

`build_conversation_context()` in `storage/conversation_context.py` is called once at the HTTP boundary and produces a frozen dataclass containing every view downstream components need: `effective_question`, `effective_lab`, `routing_snippet` (for the router LLM), `llm_history` (for the answer LLM), carry-over slots (`carried_metric`, `carried_time_phrase`), and the resolved stakeholder `role`. No downstream layer reconstructs context independently — they all read from this single object. Carry-over values are passed as structured `planner_hints` to executors; they are **never appended to `effective_question`**.

### Routing

The router (`llm_router_planner.py`) sends a structured JSON prompt to an Ollama LLM (with `format: "json"`, temperature 0) and parses the response into a `RoutePlan`. The LLM returns exactly one intent from a fixed taxonomy, plus the resolved question and the clarification decision.

Failures are handled by class, because they call for different responses. A **transport** failure (Ollama unreachable) is retried up to `OLLAMA_ROUTER_MAX_RETRIES`. A **parse** failure (`_RouterParseError`: unusable content, unknown intent, truncated JSON) is *not* retried — the request is deterministic, so a retry reproduces it — and is logged with the offending content. Both end in the regex fallback (`_fallback_plan`), which is **emergency-only** and covers only unambiguous structural keywords. The intent taxonomy:

- `definition_explanation` → knowledge executor
- `unknown_fallback` → knowledge executor
- `current_status_db`, `point_lookup_db`, `aggregation_db`, `comparison_db`, `anomaly_analysis_db`, `forecast_db` → DB executor
- `viewer_control` → viewer-control branch (opens a 3D view; `viewer_type` ∈ splat/ifc/pc/pano)
- `heatmap_control` → heatmap-control branch (toggles the heatmap overlay and selects its metric; `ui.heatmap_action` ∈ on/off, `ui.heatmap_metric` ∈ temperature/humidity/voc/pm25/null)
- `download_data` → download-control branch (hands the frontend the discrete parameters for the `/spaces/{slug}/metrics/{metric_type}/download-agg-summary` endpoint — `ui.download_slug`, `ui.download_metric_type`, `ui.download_start`/`ui.download_end` ISO-8601, `ui.download_interval`, `ui.download_format` ∈ csv/json — never a pre-built URL. A **metric is required**: when none is named the branch sets `ui.download_needs_metric=true` and asks a follow-up question instead of returning parameters)
- `ifc_model_qa` → IFC executor (questions *about* the BIM/IFC building model)
- `sensor_inspection` → sensor-inspection executor (questions about *individual* sensors/devices — ranking them by a metric, e.g. *"which sensor has the highest temperature?"*, and sensor health/offline status, e.g. *"which sensors are faulty/offline?"*)

Note the deliberate split between `viewer_control` and `ifc_model_qa`: *"open the IFC view"* is a UI action (`viewer_control`), while *"how many columns does the building have?"* is a question answered from the model (`ifc_model_qa`). `heatmap_control` is a sibling UI meta-action: *"turn on the heatmap and use temperature"* emits a `meta` event with action + metric and returns no sensor values. `download_data` is another sibling: *"export the last 7 days of temperature as CSV"* resolves the time window server-side (via `extract_time_window`, mirroring the DB path so the frontend never reconstructs date ranges; **defaults to the last 24 hours** when no window is given) and emits a `meta` event carrying the discrete parameters (`download_slug`, `download_metric_type`, `download_start`/`download_end`, `download_interval`, `download_format`) for the `/spaces/{slug}/metrics/{metric_type}/download-agg-summary` endpoint — the frontend assembles the request and renders a button, not an auto-download. A metric is mandatory: if the user names none, the branch returns `download_needs_metric=true` with a follow-up question rather than parameters. The space slug defaults to `DOWNLOAD_SPACE_SLUG` and the interval to `DOWNLOAD_DEFAULT_INTERVAL`. It is distinct from `aggregation_db`: *"what was the average CO2 last week?"* is a data question (DB path), *"download last week's data"* is `download_data`. Like `viewer_control`, all three branches short-circuit before the DB/evidence layers and are LLM-routed.

**The router is asked only for the intent of these three branches, not for their parameters.** Which 3D view, heatmap on/off and its metric, and the export format/metric/interval are closed vocabularies that map deterministically from wording, so `_parse_llm_response` derives them from the plan's `resolved_question` via the alias maps in `llm_router_planner.py` (`_infer_viewer_type`, `_infer_heatmap_action`/`_infer_heatmap_metric`, `_infer_download_format`/`_infer_download_metric`/`_infer_download_interval`). Reading `resolved_question` rather than the raw question is what keeps elliptical requests working — *"export that as a csv"* resolves because the router already rewrote *"that"* into the metric. This is not a retreat to regex routing: the intent is still LLM-decided, and coreference stays in the one field only the LLM can fill. Adding a new UI parameter means extending its alias map, not the system prompt.

### IFC executor

`executors/ifc_executor.py` answers questions about the BIM/IFC building model. `ifc_model/ifc_store.py` is a dependency-free STEP/ISO-10303-21 parser (no `ifcopenshell`) that extracts grounded facts — units, spatial hierarchy, storey elevations, an element inventory, per-element dimensions/properties, and materials — from the IFC file (default `smart.ifc`, override with `IFC_MODEL_PATH`). Parsed facts are cached by file mtime+size. The executor feeds those facts to the answer LLM with a strict "answer only from the model, never fabricate" directive; if the LLM is unreachable it returns a deterministic summary built from the parsed counts.

`ifc_model/ifc_geometry.py` resolves geometry into **world space** for measurements: it composes each element's `IfcLocalPlacement` chain (and `IfcMappedItem`/`IfcRepresentationMap` transforms) and projects BREP vertices and extruded-solid profiles to compute the overall world-coordinate bounding box (`dimensions`) and slab floor-plate polygon areas. From those, `ifc_store` derives **architectural metrics** (`architectural_metrics`): Gross Internal Area (GIA ≈ sum of floor-plate areas), footprint area, perimeter, floor-to-floor height, gross internal volume, wall thickness, and storey/envelope counts. A naive bounding box over raw points is deliberately avoided (it mixes local and world coordinates). NIA is not computed and is reported as such rather than guessed — every figure is grounded in resolved geometry or IFC attributes/properties. Computing world geometry requires retaining all entities during the parse, so the first `ifc_model_qa` call takes a few seconds; results are cached per process.

### Sensor-inspection executor

`executors/sensor_inspection_executor.py` answers questions about *individual* sensors rather than the space as a whole. It fetches a per-device snapshot from `GET /spaces/{slug}/heatmap/metrics` (via `api_client.fetch_heatmap_metrics`), which carries each device's latest value, unit, and `latest_timestamp` per metric. It normalizes those into per-metric facts — value, unit, age, and a `stale` flag (`latest_timestamp` older than `SENSOR_STALE_HOURS`, default 24h) — then feeds a grounded textual snapshot (device listing, derived offline/stale list, per-metric ranking) to the answer LLM with a strict "answer only from the snapshot, never fabricate" directive. If the LLM is unreachable, a deterministic answer is computed from the same facts (highest/lowest ranking, or the stale-sensor list), never invented. This handles both "which sensor has the highest temperature?" (ranking) and "which sensors are offline/faulty?" (health) without the router sub-classifying. The space slug defaults to `DOWNLOAD_SPACE_SLUG`. The response is narrative-only (no structured `data` payload). Like the IFC executor, it short-circuits before the DB/evidence layers and is LLM-routed.

### DB executor

`executors/db_query_executor.py` always fetches the data first, then passes structured rows to the LLM for narrative rendering. If the LLM fails, a deterministic text fallback is returned. `metadata.llm_used` indicates whether LLM rendering succeeded.

Sensor data comes from the Smart CRG REST API via `executors/db_support/api_client.py` (`SENSOR_API_BASE_URL`), not from SQL — Postgres is used only for knowledge cards and guideline records. Forecasts are fetched from the predictions API (`PREDICTIONS_API_BASE_URL`, `_handle_forecast` in `query_handlers.py`), which serves ~6 hours ahead; no forecasting model runs in this process. The LLM only explains the returned predictions, never invents future values.

### Metric planning

*Which* metrics a question needs is decided once, in `executors/db_support/metric_planning.py`, and handed to the handlers as a `MetricPlan` (priority-ordered `metrics` + a `limit`; `plan.selected` is what to fetch). The router picks the family via `RoutePlan.metric_scope` ∈ `named` / `air_quality` / `ieq_index` / `comfort` / `diagnostic` / `full`; `classify_metric_scope` reproduces the choice from question text when the router LLM was unreachable — the same LLM-primary/keyword-fallback arrangement as `analysis_mode`.

The scope vocabulary is closed and the scope→metrics table lives in code: the router names a family, never a metric, so it cannot invent one and the mapping stays testable. `analysis_mode="diagnostic"` outranks a narrower scope, because a root-cause answer needs every contributing metric.

### Answer shape: `analysis_mode`

`RoutePlan.analysis_mode` decides how the answer is *shaped*, independently of the intent that fetched the data. `"diagnostic"` decomposes drivers; `"advisory"` leads with what to do about them. Both are LLM-decided with a keyword heuristic as the emergency path, and `"advisory"` wins when a question asks WHY and WHAT TO DO in one breath — the action is what was asked for.

`"advisory"` exists because *"how can I improve the air quality?"* and *"what would you recommend to improve VOC?"* were being answered with metric-by-metric status reports, one of them closing on "No immediate action is needed". **The cause was not a missing instruction.** Four separate prompts already said the model MUST give recommendations when asked (`SHARED_SYSTEM_PROMPT`, `PRESENTATION_STYLE_PROMPT`, `_BASE_DIRECTIVE`, `_BASE_AIR_QUALITY_POINT_LOOKUP`). Nothing upstream ever *recognised* the question as advisory, so the directive selected for it was a status directive — and the model followed that directive's structure ("provide an overall status", "include metric-by-metric interpretation") rather than the one sentence buried inside it. Probed before the fix, the advice questions scattered across `definition_explanation` (an advice question read as a glossary lookup), `current_status_db` + `analysis_mode="diagnostic"`, and one unparseable plan.

The lesson is the same one the definite-article bug taught, in a new place: **when the model ignores a rule, give the decision its own field and let it select the structure — do not restate the rule more loudly.** A fifth "you MUST provide recommendations" would have changed nothing. `db_response_directive` now checks advisory before every other directive, and `DB_TOOL_RESPONSE_DIRECTIVE_ADVISORY` suppresses the metric rundown and the missing-metric disclaimer that were crowding the advice out. The resolved flag rides `payload["response_mode"]` so `render_sync` and `render_stream` read one value instead of each re-deriving it. Cases live in `tests/router_eval.py` group `advice`; the answer-shape end of it is phase `P5` of the golden conversation, which asserts action vocabulary and excludes the "no action needed" non-answer.

### Stakeholder roles: who the answer is written for

`role` on the request declares the caller's stakeholder role — `occupant` (default),
`facility_manager`, `researcher`, `executive` — and selects the audience block spliced into the
answer model's system prompt. The vocabulary is closed and lives in `prompting/roles.py`; the
blocks live in `prompting/role_prompts.py`; `GET /roles` serves the list so a frontend does not
hardcode it.

**Role is the one signal the router is never asked for.** `intent`, `analysis_mode` and
`metric_scope` are all properties of the question and must be LLM-derived from its wording. A
role is a property of the *person*: "how is the air quality?" reads identically from an
occupant, an operator, an analyst and a director, so there is nothing in the text to infer it
from. It is declared explicitly, resolved in `build_conversation_context` as **the request's
`role`, or `DEFAULT_STAKEHOLDER_ROLE`** — nothing else — and carried on `ConversationContext`
like every other per-turn view.

**Role is per-message and deliberately stateless.** An earlier version stored the last-used role
on the conversation and inherited it when the field was omitted, which protected a client that
forgot to send it. That was dropped: it made an omitted field mean something different on turn 5
than on turn 1, so the same request body could produce two differently-shaped answers depending
on history — which makes the feature impossible to experiment with and hard to debug. Switching
role between consecutive turns of one conversation is the normal case. Do not reintroduce
persistence here; `test_stakeholder_roles.py` asserts the column's absence.

`occupant` is the default because **the system already had exactly one persona** — `SHARED_SYSTEM_PROMPT`
hardcoded *"Write for non-technical occupants: plain language, no jargon"*. The occupant block
began as that wording verbatim, which made the feature a provable no-op; it deliberately is not
any more. "No jargon" constrains vocabulary and says nothing about *volume*, so combined with the
completeness rule below an occupant still received every fetched pollutant with its value — the
researcher's answer in friendlier words. Occupant now says name as few metrics as the question
allows and never use an index acronym. **This changes default behavior for callers who send no
role**, which is the intent.

**The block is spliced in one place: the system prompt.** Not also into the DB response
directives — that is how the advisory bug started, with four prompts repeating an instruction
until the model followed none of them. `get_shared_prompt_template(directive, role)` covers the DB
and knowledge paths; the IFC and sensor-inspection executors hand-write their own system prompts
and take `role_addendum(role)`, which returns `""` for the default so those two prompts also stay
byte-identical. Instant branches (viewer/heatmap/download/clarify) make no model call, so role is
a no-op there but is still echoed in metadata.

**Role may widen the data fetched, never narrow it.** Only `researcher` widens, and the widening
is a *union* with the full pack rather than a swap to it (`_WIDENING_ROLES` in `metric_planning.py`).
That distinction is load-bearing: `SCOPE_COMFORT` is ten metrics and `SCOPE_FULL` is eight, so a
promote-to-full would silently drop `itc`/`iaq` — the same failure as the comfort comparison that
lost `sound` and `light`. A union cannot lose a metric however the packs are later edited. A named
metric stays named for every role.

**What differentiates the roles is the question each reader is actually asking**, not adjectives.
A first version differed mostly in tone and produced four answers that read alike. `facility_manager`
answers *do I need to intervene, and what do I check* — it names the plausible physical cause and
the confirming check (filter loading, damper position, AHU schedule, humidifier setpoint, lamp
failure) and states the intervention or its absence unasked, explicitly superseding the
presentation rule that withholds action guidance. `researcher` wants trends and **every** applicable
guideline, not just the governing one. `executive` wants *is everything alright, or is there an
alarm* — warm, ~60 words, naming who should look at it rather than what to do. `occupant` wants
*does this affect me* — as few metrics as possible, no index acronyms, every number translated into
what they would notice.

**Role never buys a shorter answer at the cost of a metric — and that boundary had to move.**
`_METRIC_COMPLETENESS` used to require every fetched pollutant appear with its value and unit,
which is what made the occupant answer a rundown; no role block could override a directive. The
rule now distinguishes what it was actually protecting from what it incidentally mandated: a metric
flagged **EXCEEDS / NEAR / not rated must appear**, absolutely, for every audience and whatever the
word cap, while metrics comfortably within range may be accounted for collectively ("everything
else is within range" — itself a claim that must be true). The PM2.5 bug stays fixed: 17.86 µg/m³
against WHO's 15 is `EXCEEDS`, and VOC with no comparable threshold is `not rated`, so both are
still mandatory. What is no longer mandatory is reciting the readings that were fine, which was
never what protected against the omission. Every role block, occupant included, carries the clause
scoping "fewer metrics" to the within-range case.

`metadata.role` / `role_source` (`request` | `default`) / `role_fallback_used` are emitted from `_branch_metadata`, so sync
and stream cannot disagree. An unrecognised role degrades to the default rather than returning 400
— a client bug should not fail the query — but `role_fallback_used` keeps it visible.

Because the router prompt is not touched, `tests/router_eval.py` is unaffected and needs no cases.

### Metric completeness in an assessment

An air-quality assessment **must report every pollutant the Threshold Assessment flags as EXCEEDS, NEAR, or not rated** — absolutely, whatever the audience or word cap — the verdict is set by the **worst** metric rather than the best, and a metric whose threshold is missing (or given in units the reading cannot be compared against) is still reported as such. Pollutants comfortably within range may be accounted for collectively rather than listed individually; that concession is what lets the `occupant` and `executive` roles be brief, and it was carved out only after establishing it does not touch the omission below (PM2.5 over WHO's limit was `EXCEEDS`, VOC was `not rated` — both remain mandatory). "Everything else is within range" is itself a claim and must be true of every metric it covers. `_METRIC_COMPLETENESS` in `prompting/db_prompts.py` carries these rules and is attached to the two directives that render an overall assessment — deliberately *not* to `_BASE_IEQ` (an IEQ ask wants the score family) or `_BASE_POINT_LOOKUP` (one named metric was asked for).

This exists because *"how is the air quality today?"* answered with CO2, IAQ, humidity and the comfort sub-indices and called the air "good, no signs of pollutant buildup" — while PM2.5 sat at 17.86 µg/m³, above the WHO 15 µg/m³ 24-hour guideline **in its own Citation Sources**, and VOC at 0.222 ppm. Router, metric pack and executor were all correct: both metrics were in the payload with `has_full_coverage: true`. The prompt gave the model discretion ("only the most important metric-by-metric interpretation") under a hard brevity cap ("at most 2 short bullets", "under 90 words") and never said a metric may not be dropped — so it kept the flattering ones. Brevity now explicitly yields to completeness in `PRESENTATION_STYLE_PROMPT`; if you re-tighten that cap, this regresses.

A second defect made VOC the cheapest metric to drop. Its guideline records were stored under the legacy key `tvoc` while `guideline_store.get_thresholds_for_metrics` looks up `voc`, so VOC really had no retrievable threshold — and an uncited numeric claim is discouraged elsewhere in the prompt, so the model preferred silence. The deployed CHECK constraint had drifted the same way (permitting `tvoc`, forbidding `voc`, the reverse of migration 002). `storage/migrations/003_normalize_voc_guideline_metric.sql` replaces the constraint and normalises the rows; `test_air_quality_completeness.py` compares the seed's metric keys against migration 002's CHECK so that drift cannot return silently.

**VOC thresholds exist in two units on purpose.** The Atmocube (Sensirion SGP41) reports TVOC in **ppm**, range 0–3, while every published TVOC threshold is in µg/m³ — so before this there was nothing to classify a VOC reading against, and the answer either went silent on VOC or borrowed another metric's threshold. Four companion records restate the same standards in ppm using the conversion published for TVOC sensor readings, **4.9 µg/m³ per ppb**: RESET Air Grade A and WELL v2 A04 500 µg/m³ → `0.102 ppm`, WHO 2010 comfort 300 µg/m³ → `0.061 ppm`, UBA/Seifert precautionary 950 µg/m³ → `0.194 ppm`. That factor is corroborated internally — 300 µg/m³ → 0.061 ppm reproduces the top of the 0.05–0.063 ppm band this system already treated as the VOC comfort range. The mass-based records are kept: they are what the standards actually publish, and the ppm figures are **derived**, which every one of their `claim_text` and `caveat_text` fields states. Note the sensor is ethanol-calibrated, TVOC is a summed indicator, and its mass/volume relationship depends on the compound mix — so these are comparison aids, not compliance limits.

### Threshold verdicts are computed, not prompted

`executors/db_support/threshold_assessment.py` resolves, per metric, which threshold applies, whether the reading is over it, and which citation index carries it. The result is rendered as a `## Threshold Assessment (computed — authoritative)` section built once in `_build_db_context_data`, so sync and stream cannot disagree, and the directives tell the model to state those verdicts as given and never introduce a threshold number that is not in that section.

The comparison moved out of the prompt because the model could not do it. Measured on real answers it got the direction roughly right and the attribution wrong, and under a stronger prompt it invented numbers: PM2.5 17.2 µg/m³ "above the EPA daily standard" (EPA's is 35; WHO's 15 is what was exceeded); VOC 1.25 ppm reported as "0.64 ppm, below RESET Air Grade A threshold of 0.8 ppm" when RESET Grade A is 0.102 ppm here — over twelve times the limit, with neither number real; IAQ 0.0, the worst value on a 0–100 higher-is-better scale, described as "consistent with low pollutant levels". Tightening the prompt made it worse, which is what settled the design.

Four rules earn their place, each from a wrong answer:

- **A threshold only applies in the reading's own unit.** VOC reads in ppm and most TVOC limits are in µg/m³; crossing that gap needs a molar-mass assumption, so a metric with no threshold in its unit is `unrated`, never guessed.
- **Unit spellings are folded.** The registry writes PM2.5 as `μg/m³` (U+03BC) and the seed as `µg/m³` (U+00B5). They render identically, and a naive comparison silently concludes the metric has no threshold.
- **The strictest applicable threshold governs**, so a clean verdict cannot be bought by quoting the most permissive standard.
- **A hard limit outranks a comfort-band edge.** `threshold_type` `max`/`min` is a limit; `range_max`/`range_min` is the top of an optimal band. 54.8 %RH is above EPA's 50 % optimal top but well under ASHRAE's 65 % limit — treating those alike flags an ordinary room. Band edges only decide when a metric has no hard limit, and then report as *outside optimal range*.

Index metrics (IEQ/IAQ/ITC/IAC/IIL) get their own vocabulary — `POOR`/`FAIR`/`GOOD` against the 0–100 bands — because nothing is "exceeded" when a score is low, and `EXCEEDS` invited the model to describe 0/100 as a breached limit rather than the worst possible score.

**Citation style:** a threshold *number* is quoted only for metrics flagged `EXCEEDS` or `NEAR`, where magnitude is the point. A metric within range says so and lets `[N]` carry the number — the frontend renders citations, so the reader can follow the source for detail instead of reading a list of limits that are all fine.

Do not move this comparison back into the prompt. Two attempts are on record: a "cite the source that publishes the number you applied" clause produced a fabricated *"RESET Air Grade A threshold of 0.07 ppm"* and was reverted, and every prose variant left attribution wrong roughly as often as right.

Two guideline records carry a NULL `threshold_value` **on purpose** and must not be "filled in": ASHRAE 62.1 defines ventilation rates, not a CO2 ppm limit (asserting one is the exact false claim `GUIDELINE_CITATIONS` rule 4 exists to prevent), and ASHRAE 55 defines thermal comfort through PMV rather than a fixed temperature. Both surface as `unrated` rather than being compared against something they do not define.

### Key invariants

- Endpoint handlers stay thin — all business logic lives in orchestration/executor modules.
- **A metric pack is never truncated below its own length.** The limit travels with the pack in `metric_planning`, and handlers only ever do `plan.selected` — they must not re-slice. This was previously a comment asking thirteen call sites to behave, and two of them didn't: a comfort comparison capped at 8 against a 10-metric pack silently dropped `sound` and `light`, the two metrics that make it a comfort assessment.
- **There is exactly one intent ladder.** `query_orchestrator.plan_branch` maps a `RoutePlan` to one `Branch`; `render_sync` and `render_stream` both consume that object and neither may branch on intent itself. Adding an executor means writing one branch factory — if you find yourself editing two ladders, the drift that lost the stream its citations is starting again.
- A `Branch` is either *instant* (`answer` set: a fixed confirmation or question, no model call) or *generated* (`run_sync`/`open_stream`). Blocking pre-work goes in `prepare` so the stream renderer can offload it and both renderers share its result.
- A streamed answer must carry the same citations as the sync one. The sync response returns `citation_sources` + `footnotes` in its body; the stream emits them in a terminal `sources` frame (`executors/sse.py::sources_event_for_answer`) resolved from the accumulated tokens. Any new streaming executor that grounds its answer in guideline records must emit that frame before `done`.
- Every streaming path terminates with `done`, even on failure — an executor that loses its LLM emits its deterministic fallback text first, and route-level failures go through `runtime_errors.stream_error_payload` (which emits `error` + `done`). A client must never be left waiting on a silent stream.
- `ConversationContext` is immutable (`frozen=True`) and built once per turn.
- Conversations are owned by the authenticated caller (`storage/conversation_store.py`). Anything that reads or writes turns passes the caller id through; reusing another caller's `conversation_id` raises `ConversationAccessError` → HTTP 403.

## Routing Preference

**Prefer LLM-based routing over regex.** The clarify decision follows the same rule: the router sets `needs_clarification` + `clarification_question` (honored only when the request passes `allow_clarify`), rather than a keyword gate in the DB executor deciding when scope is "underspecified".

 Regex is a last resort (emergency fallback only). When improving or extending routing behavior, the right approach is to tune the system prompt in `llm_router_planner.py::_SYSTEM_PROMPT` and adjust intent definitions and examples — not to add regex patterns. The LLM handles ambiguity, typos, and natural language variation far better than keyword matching. Regex rules exist only in `_fallback_plan()` to cover the case where the Ollama endpoint is completely unreachable.

**Worked example — the definite-article bug, fixed by prompting alone.** *"what is CO2?"* asks for the concept; *"what is the CO2?"* asks for this building's reading. The router got this wrong in both directions: it read acronyms (VOC, IEQ, PM2.5) as glossary entries even behind an article, and it copied the previous turn's route whenever the same metric came up again, so *"what is humidity?"* after a humidity reading returned a value instead of a definition. Prose rules did not fix it — measured on a 68-call probe (both directions, with and without prior conversation): prose rules alone 55/68, a symmetric article rule 58/68, a single categorical `value|concept` field 56/68. What worked, at 68/68, was **changing the output order**: two scratchpad fields, `metric_phrase` (quote the metric wording, article included) and `metric_has_article` (does it start with "the"?), declared *before* `intent`, plus an explicit statement that prior turns never add or remove the article. Nothing parses those fields — forcing the decision to be made and written down before the intent is the entire mechanism, which is why `_parse_llm_response` carries a comment telling you not to delete them.

The transferable lesson: when the model is ignoring a rule, **making it emit the decision as a field usually beats restating the rule more forcefully** — and a stronger restatement can overshoot into the opposite error, which is what the eval catches and hand-sampling does not. Measure variants against `tests/router_eval.py`; a routing bug is fixed when the scorecard says so.

## Configuration

All runtime settings come from `.env` (auto-loaded by `core_settings.py`) and the process environment. Shell env vars win over `.env` values. Key variables:

| Variable | Purpose |
|---|---|
| `OLLAMA_ROUTER_MODEL` | LLM used for intent routing (default: `qwen3:30b-a3b-instruct-2507-q4_K_M`) |
| `OLLAMA_ROUTER_BASE_URL` | Ollama endpoint for the router |
| `OLLAMA_MODEL` / `OLLAMA_BASE_URL` | Separate model/endpoint for answer generation |
| `DATABASE_URL` | Postgres connection (or use `DB_*` components). Resolved lazily on first use — the process no longer fails at import when it is unset (only the guideline/knowledge-card path requires it) |
| `SENSOR_API_BASE_URL` | Base URL of the Smart CRG sensor REST API for latest readings / agg-summaries (default: `http://192.168.50.99:7001`) |
| `PREDICTIONS_API_BASE_URL` | Base URL of the forecast/predictions API (default: `https://api.smart-crg.com`) |
| `OLLAMA_ROUTER_NUM_PREDICT` | Token budget for the router's JSON plan (default: `768`). Too low truncates the plan and silently drops the route to the regex fallback |
| `MAX_QUERY_WINDOW_DAYS` | Upper bound on a resolved query window (default: `366`). Aggregation is always hourly, so this is what stops one question fanning out into tens of thousands of upstream buckets. When it trims a window the label says so, since the label reaches the answer LLM |
| `DISPLAY_UTC_OFFSET_HOURS` | UTC offset used for **both** parsing question dates and labelling timestamps (default: `4`, Gulf Standard Time). `time_windows.target_tz()` is the single source — never hardcode an offset |
| `RAG_API_KEYS` | Comma-separated API keys accepted on `/query`, `/query/stream`, `/sensors/latest/{space}`, `/ifc/summary` (send `X-API-Key: <key>` or `Authorization: Bearer <key>`). **Unset by default, which leaves those endpoints open** and puts every caller in one shared conversation namespace; set it before exposing the service beyond localhost. Each key maps to a distinct caller id that owns its conversations — reusing another caller's `conversation_id` returns 403 |
| `RAG_API_CORS_ALLOW_ORIGINS` / `RAG_API_CORS_ALLOW_CREDENTIALS` | CORS origins/credentials. Credentials are auto-disabled when origins are wildcarded (`*`), since browsers reject that combination — set explicit origins to use credentialed CORS |
| `IFC_MODEL_PATH` | Path to the IFC building model for `ifc_model_qa` (default: `./smart.ifc`) |
| `DOWNLOAD_SPACE_SLUG` | Default `{slug}` for the `download_data` endpoint path when no space is named (default: `smart_lab`) |
| `DOWNLOAD_DEFAULT_INTERVAL` | Default aggregation interval for `download_data` when none is named (default: `1h`, emitted to the frontend as `1hr`) |
| `SENSOR_STALE_HOURS` | Age threshold (hours) above which a sensor reading is flagged faulty/offline by `sensor_inspection` (default: `24`) |
| `DEFAULT_STAKEHOLDER_ROLE` | Role used when neither the request nor the conversation supplies one — `occupant` (default) \| `facility_manager` \| `researcher` \| `executive`. `occupant` reproduces the single persona the assistant had before roles existed. An unrecognised value falls back to it |

Two distinct Ollama models are configured separately: one for routing (`OLLAMA_ROUTER_*`) and one for answer generation (`OLLAMA_*`). Keep these separate — the router runs at temperature 0.0 with constrained output; the answer model has different latency/quality tradeoffs.

## Endpoints

These are all the routes the app serves:

- `POST /query` — routed query, single JSON response
- `POST /query/stream` — routed query as SSE (`status`, `meta`, `meta_update`, `token`, `sources`, `done`, `error`)
- `GET /` — service info
- `GET /health` — liveness check (unauthenticated, for probes)
- `GET /sensors/latest/{space}` — latest reading snapshot for a space
- `GET /ifc/summary` — parsed structured summary of the IFC building model (units, storeys, element counts, materials)
- `GET /roles` — stakeholder roles accepted by `/query`, with labels and the server default

Everything except `/` and `/health` requires an API key when `RAG_API_KEYS` is set.

## Debugging

There is no route-preview, SQL-proof, or KPI endpoint — inspect a routing decision from
`metadata` on a normal `/query` response (`intent`, `route_confidence`, `planner_model`,
`fallback_used`, `resolved_question`) or from the `meta` frame on the stream. Router
degradation is logged as a warning by `_log_router_fallback` when the LLM is unreachable.

For a routing *behaviour* question ("did my prompt change break follow-ups?"), use
`python tests/router_eval.py` rather than sampling questions by hand — it reports a
per-capability scorecard and shouts when a case fell through to the regex fallback, which
is otherwise invisible except as `fallback_used` in the metadata.
