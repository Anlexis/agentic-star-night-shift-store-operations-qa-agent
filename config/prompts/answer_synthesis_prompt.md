# Answer Synthesis Prompt — RET-C2-668 (model-backed upgrade seam)

> **The shipped implementation does NOT use this prompt at runtime.**
> `GenerateAnswerNode` is deterministic: rule-based grounded assembly over
> `ranked_documents` on the ordinary path, and no node reads this file. It
> documents the synthesis contract for the model-backed upgrade described in
> `docs/02_design.md` ("Answer synthesis"), so that swap changes only the
> ordinary branch inside `GenerateAnswerNode.execute()`.

## ⚠️ The safety-escalation boundary is OUT of this seam

`GenerateAnswerNode` has two branches: the safety-escalation branch (fixed,
looked-up instruction text for `emergency_security` / `emergency_medical` /
`emergency_disaster` / `emergency_unspecified`) and the ordinary grounded-answer
branch. **Only the ordinary branch is in scope for this prompt.** The escalation
branch must remain a deterministic constant lookup afterwards as well — it must
never be routed through a model call, prompted, or made to depend on model
output. That is a permanent safety control, not a placeholder for something
better.

The output boundary enforces it independently: `PostProcessNode` re-checks the
rendered answer against `src/services/escalation_policy` and blocks an
escalation-flagged answer that is not its category's exact instruction. A
model-backed node that started generating emergency text would be caught there,
not merely discouraged here.

## Contract (model-backed GenerateAnswerNode, ordinary branch only)

- **Input:** the same `ranked_documents` JSON (id / title / category / source /
  score / excerpt) and `search_query` the shipped node reads.
- **Output:** the same state contract — `grounded_answer` (str, with numbered
  `[n]` citation markers), `citations` (JSON list of `{ref, id, title, source}`),
  and `escalation` (JSON `{"required": false, "category": null, "channel": null,
  "reason": null}` on this branch — the node must still emit the field, just
  always non-required, since it only ever runs when the escalation check upstream
  did not fire).
- **Grounding rule:** every factual statement in the answer must be traceable to
  one of the supplied passages via an `[n]` marker; content not present in the
  passages must not be asserted.
- **No-coverage rule:** when no passage supports the question, say so and
  recommend asking the shift supervisor for a manual review — never answer from
  parametric knowledge.
- **Tone:** neutral, operational, no individualized recommendations beyond what
  the manual states (the standing disclaimer is appended downstream by
  `OutputFormatNode`).

## Prompt template (ordinary branch)

```
You answer convenience-store night-shift operational questions strictly from
the store-operations-manual passages provided below. This prompt is used ONLY
when the caller's question was already confirmed NOT to be a safety, security,
or medical/disaster emergency — do not attempt to detect or handle an
emergency here; that check runs before this prompt is ever invoked.

Question:
{search_query}

Passages (each with a reference number):
{ranked_documents}

Rules:
1. Use ONLY the passages above. If they do not answer the question, say the
   operations manual has insufficient coverage and recommend asking the
   shift supervisor.
2. Mark every factual statement with the [n] reference of its passage.
3. Do not give guidance beyond what the manual states.
4. Keep the answer under 300 words.
```

## Configuration coupling

The `llm` block in `config/config.yaml` (`temperature`, `max_tokens`) is already
forwarded to the inner graph by `NightShiftOpsGraphNode._parent_config()` under
`config["configurable"]["llm"]`; a model-backed node reads it from there.
`config/agent.yaml` is the static manifest and carries no runtime values — a
reader pointed at it would find no such key and silently fall back to its own
default.
