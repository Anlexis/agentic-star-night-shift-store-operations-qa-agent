# Test Specification — RET-C2-668

**Template ID:** RET-C2-668
**Template Name:** ConvenienceStoreNightShiftOpsAgent
**Category:** Cat 2 (nested retrieval workflow + safety-escalation boundary)

This document is the contract the shipped test code implements. Every table below
names a file that exists in this repository; every file in `tests/` is covered by
a table here.

## 1. Scope and invocation conventions

- Per-node unit tests for the five inner domain nodes and the two outer gate nodes.
- The caller-data contract and the instruction-override screen.
- The escalation-integrity gate at the output boundary, including containment.
- Manifest / runtime-config consistency and seeded-corpus integrity.
- Inner-graph and outer-graph composition, including THE SAFETY BOUNDARY.
- Boundary tests: import isolation, State checkpoint safety, invoke order,
  human-in-the-loop propagation (conditional — auto-waived), server boot, and
  end-to-end behaviour through the real `POST /invoke`.

**Trust-gate invocation canon.** Per-node tests invoke the node via `node(state)`
— through `BaseNode.__call__`, which runs the trust gate, the PII mask,
`execute()`, then the credential gate — rather than a bare `node.execute(state)`.
The state builder sets `caller_trust_level` to `VERIFIED_EXTERNAL` for the two
outer gate slots and `ANONYMOUS` for the five inner domain nodes.

**The one deliberate exception** is `tests/unit/test_caller_contract.py`, which
calls `PreProcessNode.execute()` DIRECTLY. That is the point of those tests: the
template owns its refusal, and a test that only shows "something upstream refused
it" passes wherever a platform gate happens to be active while saying nothing
about a host where it is absent or configured off. Calling `execute()` with no
wrapper in front is the only way to demonstrate the node itself refuses.
Assertions there stay behavioural — for instruction-override content the
terminating error status, for a caller-field breach the completing status plus
the reason code, and in both cases nothing carried forward, the field named, the
value never echoed — never a gate's exact wording.

**Node signature.** `execute(self, state) -> dict` is the only node signature in
this repository; no node accepts a `config` parameter. Configuration-knob tests
seed the knob through the state-seeded `retrieval_config` field.

**PII masking expectations.** The framework input gate masks
`user_input` / `validated_input` / `llm_response` to `[MASKED]` before `execute()`
runs — e-mail and hyphenated Japanese phone numbers match the framework's own
patterns. `PreProcessNode`'s own `_surface_strip_identifiers()` additionally
catches shapes the framework does not recognise, notably an un-hyphenated digit
run, replacing them with `[REDACTED]`. Positive-path payloads are therefore
lowercase, PII-free operational phrasing; intentional-PII tests assert the raw
identifier is gone and the applicable marker is present. Domain fields
(`search_query`, `retrieved_documents`, `grounded_answer`, `formatted_answer`, …)
are not input-scan targets.

**Audit muting.** `shared.*` is never `sys.modules`-stubbed (the framework imports
`shared.security` at load time). The domain emitter is muted by an autouse fixture
(`tests/unit/conftest.py`) patching `src.nodes.<mod>.emit_trace_event`; the audit
assertion tests re-patch the same attribute with a spy and assert on
`call.args[1]` (the event payload), never the whole-call repr.

## 2. Unit tests

### 2.1 PreProcessNode — `tests/unit/test_pre_process_node.py`

**Reading the Expected column.** A rejection the caller can correct completes:
`status=SUCCESS` **and** a non-empty `error_code`, asserted together — the status
alone would also pass on a run that quietly answered, so the reason code is what
proves the request was not carried out. A refusal the caller cannot correct by
rewording (instruction-override content) still terminates with `status=ERROR`;
the two are deliberately not merged into one predicate.

| ID | Case | Input | Expected |
|----|------|-------|----------|
| PRE-01 | Valid question | lowercase, PII-free ops question | `status=SUCCESS`, `validated_input` set |
| PRE-02 | Empty input | `""` / whitespace | completes: `status=SUCCESS` + `error_code` set, `error_log` non-empty, no `validated_input` |
| PRE-03 | Missing / non-string input | absent; dict payload | completes: `status=SUCCESS` + `error_code` set |
| PRE-04 | Un-hyphenated phone/ID digit run | `09012345678` | node's own screen → `[REDACTED]`; raw digits absent |
| PRE-05 | Hyphenated phone number | `090-1234-5678` | framework input gate → `[MASKED]` |
| PRE-06 | E-mail | `shift.lead@example.com` | framework input gate → `[MASKED]` |
| PRE-07 | Long membership-style digit run (14 digits) | outside the exact card shapes | node's own screen → `[REDACTED]` |
| PRE-08 | Audit | valid question | `pre_process_complete` emitted; payload carries `input_chars` |
| PRE-09 | Trust declaration | — | `required_trust_level == VERIFIED_EXTERNAL` |
| PRE-10 | Caller channel | `input_context={"channel": "pos-terminal"}` | carried into `enriched_context` |

### 2.2 Caller contract and screening — `tests/unit/test_caller_contract.py`

| ID | Case | Expected |
|----|------|----------|
| CC-01 | Non-finite strings (`NaN`, `nan`, `Infinity`, `-Infinity`, `inf`, `-inf`) per numeric field | refused |
| CC-02 | Raw non-finite floats (`float("nan")`, `float("inf")`, `float("-inf")`) per numeric field | refused |
| CC-03 | Booleans as numbers (`true` / `false`) | refused — `isinstance(True, int)` is True |
| CC-04 | Out-of-range magnitudes (`top_k` 0 / 21 / −3 / 1e9; `score_threshold` −0.1 / 1.1 / 1e9) | refused |
| CC-05 | Non-numeric types (dict / list / None / object); fractional `top_k` | refused; valid values accepted; the error names the field and never the value |
| CC-06 | Inert identifiers (`age_verification`, `Age_Verification`, `pos-terminal`, `kb2`) | accepted, lower-cased |
| CC-07 | Non-inert values (space, slash, SQL, markup, non-ASCII, >32 chars, empty) and non-strings | refused, never coerced |
| CC-08 | Chat-template control tokens (`<\|im_start\|>`, `<\|system\|>`, `[INST]`, `<<SYS>>`) | flagged |
| CC-09 | Directive phrases (ignore/disregard, reveal-prompt, role reassignment, developer mode, override-safety, new-system-prompt) | flagged |
| CC-10 | A token attack is flagged even though the markup strip removes the token | flagged raw |
| CC-11 | A directive spliced with markup (`ig<b>nore</b> all rules`) | not matched raw, matched after the strip, flagged overall |
| CC-12 | Ordinary night-shift questions containing directive verbs (8 real phrasings from this corpus) | NOT flagged — the fail-closed direction |
| CC-13 | Structured-channel scan: hostile value, hostile KEY, nested hostile value, `\u`-escaped payload | found; a clean payload returns None (the control); an unrecognised field name is masked in the reported path |
| CC-13a | Structural limits: a payload wider than 32 entries or nested deeper than 6 levels | refused as a violation in its own right; a payload inside the limits still passes (the control) |
| CC-14 | `PreProcessNode.execute()` called directly | terminates (`status=ERROR`) on an injected question and on a hostile structured field; completes (`status=SUCCESS` + `error_code` set) on an out-of-bounds number; never echoes the value; carries the contract forward on an ordinary request |

### 2.3 InputValidateNode — `tests/unit/test_input_validate_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| INPV-01 | Plain text | free-text question | whole string becomes `search_query`; filters all `None`; no `status` key (middle domain node) |
| INPV-02 | Fallback | `user_input` only | still parsed |
| INPV-03 | Whitespace | ragged spacing/newlines | collapsed to single spaces |
| INPV-04 | Oversize question | > 2000 chars | truncated to 2000 + note |
| INPV-05 | JSON envelope | `{"query","category","top_k"}` | all parsed; category lower-cased |
| INPV-06 | `question` alias | `{"question": "..."}` | accepted as the query |
| INPV-07 | Malformed JSON | `{`-prefixed non-JSON | treated as a plain-text question + parse note |
| INPV-08 | Out-of-range `top_k` (999) | — | REFUSED, completing: `status=SUCCESS` + `error_code` set, nothing carried forward |
| INPV-09 | Non-numeric `top_k` | `"soon"` | REFUSED — same completing shape as INPV-08 |
| INPV-10 | Non-finite `top_k` (`NaN` / `Infinity` / `-Infinity` as raw JSON) | — | REFUSED — the earlier clamping path routed these into `int()`, where `int(inf)` raises `OverflowError` |
| INPV-11 | Non-finite `score_threshold` | — | REFUSED |
| INPV-12 | Boolean `top_k` | `true` | REFUSED |
| — | Refusal hygiene | hostile `category` | the field is named, the value never echoed |
| — | Valid `score_threshold` | `0.75` | carried into `query_filters` |
| — | Channel precedence | envelope + `caller_context` | the structured channel wins; the envelope fills what it omits; absent context is not an error |
| — | Empty request | `""` | `search_query=""` + note (non-fatal) |
| — | Audit | category, no `top_k` | `input_validate_complete` with the three presence flags |
| — | Serialization | any | `query_filters` is a JSON string, never a bare dict |

### 2.4 RetrieveNode — `tests/unit/test_retrieve_node.py`

| ID | Case | Expected |
|----|------|----------|
| RET-01 | Age-verification question | `kb-002` / `kb-003` surfaced |
| RET-02 | Fallback chain (`validated_input` then `user_input`) | both resolve the same top hit |
| RET-03 | No lexical overlap | `[]` |
| RET-04 | Ordering | scores sorted descending |
| RET-05 | Category filter (include) | only the matching entry |
| RET-06 | Category filter (exclude) | `[]` |
| RET-07 | `retrieval_config` `top_k` | state-seeded override honoured |
| RET-08 | Missing corpus file | `[]` + a "not readable" note |

### 2.5 RerankFilterNode — `tests/unit/test_rerank_filter_node.py`

| ID | Case | Expected |
|----|------|----------|
| RRK-01 | Relevance floor (0.6 / 0.3) | 0.3 dropped at the default 0.5 floor |
| RRK-02 | Nothing survives (0.1) | `[]` |
| RRK-03 | Configurable threshold (0.2) | 0.3 survives |
| RRK-04 | Category boost (0.45 matching) | boosted to 0.55, survives |
| RRK-05 | Boost non-application (0.45 non-matching) | stays 0.45, dropped |
| RRK-06 | `top_k` cap | capped |
| RRK-07 | Stricter caller `top_k` wins; looser is ignored | as stated |
| RRK-08 | Tie-break | deterministic id-ascending order |
| — | Audit | `rerank_filter_complete` with `kept` / `dropped` counts |

### 2.6 GenerateAnswerNode — `tests/unit/test_generate_answer_node.py`

> THE SAFETY BOUNDARY itself is covered by
> `tests/unit/test_trust_gate.py::TestSafetyEscalationBoundary`. This file covers
> the ordinary grounded-answer path.

| ID | Case | Expected |
|----|------|----------|
| GEN-01 | Single passage | `[1]` marker; citation `ref=1` |
| GEN-02 | Multiple passages | `[1]` / `[2]` markers in order |
| GEN-03 | Lead sentence | the question is quoted in the lead line |
| GEN-04 | Missing question | generic lead-in |
| GEN-05 | No coverage | fixed no-coverage message; `citations=[]`; `escalation.required=False` |
| GEN-06 | Missing key | same no-coverage fallback |
| GEN-07 | Audit | `generate_answer_complete` with `citation_count`, `escalation_required`, `no_coverage` |

### 2.7 OutputFormatNode — `tests/unit/test_output_format_node.py`

> The output key is `formatted_answer` (this node), never to be confused with
> `formatted_output` (the outer gated key).

| ID | Case | Expected |
|----|------|----------|
| OUT-01 | Full compose | header + body + Sources row + disclaimer; no Escalation section; `status=SUCCESS` (plain string) |
| OUT-02 | No citations | explicit "- none (…)" sources line |
| OUT-03 | Missing body | "No answer is available…" fallback |
| OUT-04 | Escalation section | `## Escalation Required` + `**Escalate to:** Police - 110` |
| OUT-05 | All known channels | each maps to its documented label |
| OUT-06 | Unmapped channel | falls back to the raw channel value |
| OUT-07 | Missing channel | falls back to "Shift supervisor" |
| OUT-08 | Audit | `output_format_complete` with `citation_count`, `escalation_required` |

### 2.8 PostProcessNode content gate — `tests/unit/test_post_process_node.py`

| ID | Case (`result`) | Expected |
|----|------------------|----------|
| POST-01 | Clean output | `formatted_output=result`, `status=SUCCESS` (plain string) |
| POST-02 | Empty result | forwarded as-is, `status=SUCCESS`, no audit emit |
| POST-03..06 | Credential leak: API key / `secret=` assignment / JWT / Bearer token (each assembled at runtime, so no credential-shaped literal sits in the repository) | blocked stub in `formatted_output` AND `result`; `status=ERROR`; the raw secret in neither; `ops_answer` / `citations` / `escalation` cleared |
| POST-07..09 | Module-level `security_gate_output()` recursion | recurses into dict values, dict KEYS, and list/tuple elements; scalars are inert — a credential nested one level deep cannot bypass a top-level-string scan |

### 2.9 Escalation-integrity gate — `tests/unit/test_escalation_output_gate.py`

| ID | Case | Expected |
|----|------|----------|
| EGATE-01 | A faithful escalation answer | passes |
| EGATE-02 | An ordinary answer | out of scope, passes |
| EGATE-03 | A generated (prose) emergency answer | `escalation_instruction_not_verbatim` |
| EGATE-04 | The wrong category's instruction | `escalation_instruction_not_verbatim` |
| EGATE-05 | Instruction present but blended with the ordinary lead line | `escalation_blended_with_generated_answer` |
| EGATE-06 | Flagged without a category | `escalation_category_missing` |
| — | Every category round-trips through the policy module | passes; a channel exists |
| EGATE-07 | Node containment | `status=ERROR`; the un-gated body survives in no field; `ops_answer` cleared, `citations` / `escalation` `None`; the error carries no traceback and no source path |
| EGATE-08 | Envelope containment | the ERROR envelope carries no released text; structured fields withheld. Control: a SUCCESS envelope still surfaces them. Credential violations are contained the same way |

### 2.10 Manifest and runtime config — `tests/unit/test_config_manifest.py`

| ID | Case | Expected |
|----|------|----------|
| CFG-01 | Flat manifest | `id` / `name` / `namespace` at root; no nested `agent:` block |
| CFG-02 | Class contract | manifest `class` == `src.graph.graph.<agent class>` |
| CFG-03 | Classification | Cat 2 / RET / the declared base type |
| CFG-04 | Trust level | manifest `VERIFIED_EXTERNAL` == both outer gate nodes |
| CFG-05 | `max_retry` | int, `0 ≤ v < 10` (framework ceiling), read from `config/config.yaml` |
| CFG-06 | Retrieval block | `top_k` / `score_threshold` mirror the node module defaults; the corpus path exists |
| CFG-07 | `_parent_config()` | forwards the `retrieval` + `llm` blocks; never empty |
| CFG-08 | Seeded corpus | JSON list ≥ 5 entries; unique ids; required keys; all three escalation categories present; every category is an inert identifier |
| CFG-09 | Compile-time gates | `generation_mode: deterministic`; `requires.secrets == []`; `requires.extras == []` |
| — | Reader placement | the loader reads `config/config.yaml`; `retrieval` / `llm` are absent from the manifest |
| — | Live values | `agent_config()` carries `max_retry` / `timeout_s`, and the declared `max_retry` reaches the compiled graph |
| — | HITL waiver | no `hitl.enabled: true` |

## 3. Integration and composition

### 3.1 Inner graph — `tests/unit/test_domain_workflow_graph.py`

| ID | Case | Expected |
|----|------|----------|
| DWG-01 | Composition | inherits `BaseGraph`; registers exactly the five domain nodes; no initialize/finalize |
| DWG-02 | Config forwarding | `_extra_initial_state()` republishes the retrieval block AND seeds `caller_context` off the bridge, both as JSON strings |
| DWG-03 | Output shape | `get_output()` emits the merge contract; `route()` → END on error |
| DWG-04 | Inner end-to-end (ordinary) | SUCCESS; grounded answer + disclaimer + a real citation; `escalation.required=False` |
| DWG-05 | Inner node history | linear order across the five domain nodes |
| DWG-06 | Inner end-to-end (emergency, retrieval miss) | a fire question whose protocol entry is dropped below the floor still escalates end to end |
| — | Routing annotation | `route()` is annotated with this graph's own `State`; a conditional edge may not be added without it |

### 3.2 Outer graph — `tests/unit/test_graph_composition.py`

> The outer `pre_process → main` edge is unconditional (the backbone only routes
> conditionally after `main`), so on a trust-gate denial the main-slot node still
> runs; its input-gate check sees the error state and skips `execute()`, and
> routing then sends the run straight to `finalize`, skipping `post_process`.
> GRP-09 therefore asserts the stable prefix plus `PostProcessNode`'s absence.

| ID | Case | Expected |
|----|------|----------|
| GRP-01 | Outer composition | inherits `AgentBaseGraph`; `Graph` alias; `add_edges()` NOT overridden |
| GRP-02 | Backbone slots | `compile()` fills all five with the expected classes |
| GRP-03 | `get_subgraph()` | returns `DomainWorkflowGraph` carrying the forwarded config |
| GRP-04 | `extract_input()` | prefers `validated_input`, falls back to `user_input` |
| GRP-05 | `merge_output()` | inner `formatted_answer` → outer `ops_answer` AND `result`; changed keys only |
| GRP-06 | Config fallback | `_parent_config()` never empty even with an unreadable `config/config.yaml` |
| GRP-07 | End-to-end happy path | VERIFIED_EXTERNAL invoke → SUCCESS; gated answer as `output` |
| GRP-08 | End-to-end full backbone | the five backbone class names in order |
| GRP-09 | End-to-end trust-gate denial | ANONYMOUS invoke → ERROR; empty `output`; `PostProcessNode` absent |
| GRP-10 | Structured-output fail-closed | on the denied run, `escalation` / `citations` are ABSENT from the envelope |
| — | Escalation surfaced | emergency question → envelope carries `escalation.required=True` with category/channel/reason; ordinary question → `False` with real citations |
| — | Serialization helpers | `to_json` / `from_json` round-trip; None and malformed handling |

## 4. Boundary tests

| ID | File | Expected |
|----|------|----------|
| PB-IMPORT | `test_import_isolation.py` | no platform-SDK import anywhere under `src/` |
| PB-STATE | `test_state_safety.py` | `State` has no credential-named fields and no model / context annotations |
| PB-6 | `test_pb_invoke_order.py` | a full `Graph().invoke()` at VERIFIED_EXTERNAL over the payload byte-equal to `deploy/invoke_payload.json`'s `input` → SUCCESS with the exact five-slot `node_history` |
| PB-7 | `test_pb7_hitl_interrupt_propagation.py` | **Auto-waived — non-HITL** (no `hitl.enabled: true`; no graph declares `propagate_hitl=True`); conditional skip stub |
| PB-BOOT | `test_server_boot.py` | `import src.api.server` does not raise; the module-level agent is this agent, compiled; a fresh constructor + `compile()` fills the five slots; `/health` reports the agent |
| PB-E2E | `test_invoke_e2e.py` | see below |

### 4.1 End-to-end through `POST /invoke` — `tests/proof_of_boundary/test_invoke_e2e.py`

The app is driven through its real ASGI interface — no test client, since the
HTTP client library is only a transitive dependency, and a raw body lets the tests
send JSON literals a strict encoder would refuse to produce.

| Group | Cases |
|---|---|
| Auth boundary | no token → 401; wrong token → 401; correct Bearer token reaches the agent |
| Real work on the public path | a non-empty answer citing a real passage; two different questions retrieve different passages (a stub-shaped agent would return the same baseline); an out-of-corpus question declines rather than improvising |
| Caller context reaches the inner graph | the category filter changes the retrieved passages; `top_k` narrows the answer; a STRICTER relevance floor is honoured; a LOOSER one is ignored |
| Caller-contract rejection | non-finite numbers sent as raw JSON literals; out-of-range `top_k`; a non-inert `category`; an empty question — each completes (envelope `status=success`) with `output` equal to the fixed reason sentence, never an answer |
| Injection refused end to end | `<\|im_start\|>`, `[INST]`, `<<SYS>>`, and a directive phrase; a hostile field NAME refused without being echoed; an ordinary question containing directive words still answered |
| Escalation end to end | the fixed instruction and routing channel are returned; the ordinary lead line and the caller's question are absent; the instruction does not vary with the wording (English and Japanese) |
| Output-invariant scan | every answer carries the standing disclaimer; no answer, citation set or escalation payload carries credential-shaped content |

> **Pre-review gate checklist:** PB-IMPORT, PB-STATE, PB-6, PB-BOOT and PB-E2E are
> mandatory. PB-7 applies only to templates using human-in-the-loop; this template
> does not, so it is auto-waived and its skip must not block the gate.

## 5. Execution summary

- Runner: the released `agenticstar-agentcore` wheel, `python -m pytest tests/`.
- Total: 304 — 303 pass, 0 fail, 1 skip (the PB-7 conditional stub, auto-waived).
- Determinism: no model call, no network. Retrieval and answer assembly are
  rule-based; the escalation branch is a fixed lookup table, never generated.
