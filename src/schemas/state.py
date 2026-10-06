"""AgentCore Platform v1.0"""

# State must be a flat TypedDict - never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.
#
# Checkpoint safety: structured fields (dict / list[dict]) are stored
# as JSON STRINGS, not bare Python containers - a bare dict/list in a
# checkpointed State field is a checkpoint-safety violation. Producers
# serialize with to_json() on write; consumers deserialize with from_json()
# on read.
#
# RET-C2-668 - Convenience Store Night Shift Operations Q&A Agent (Cat 2 RAG).
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph).  Fields below cover both layers.
#
# PII / confidentiality note: direct identifiers (phone numbers, long
# membership/employee-ID-like digit runs, e-mail) in the query payload are
# surface-stripped by PreProcessNode (trust + identifier screen) before any field is written to
# State. Only the normalised search query, KB passage summaries, and the
# final grounded answer / escalation routing are persisted - never raw
# customer or employee identifiers.

import json
from typing import Any, NotRequired, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a dict/list State field to a JSON string (checkpoint safety).

    None passes through unchanged so an 'unset' field stays distinguishable
    from an empty container.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON-string State field back to its dict/list.

    None / empty / malformed input -> the supplied ``default`` so a missing or
    corrupt field is non-fatal for the consuming node.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for RET-C2-668.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    Domain fields are NotRequired so the TypedDict is valid at graph
    initialisation, before any node has written a value.
    """

    # ------------------------------------------------------------------
    # Outer layer - set by PreProcessNode / NightShiftOpsGraphNode.merge_output
    # ------------------------------------------------------------------

    # Identifier-stripped, validated query payload produced by PreProcessNode
    # (trust + identifier screen).  Raw input is NOT persisted beyond PreProcessNode.
    validated_input: NotRequired[str]

    # Final night-shift ops answer, mapped from the inner graph's
    # formatted_answer output via merge_output.
    ops_answer: NotRequired[str]

    # JSON STRING (to_json) of the VALIDATED caller contract, written by
    # PreProcessNode from the invoke call's input_context. Deserialised shape:
    # {"category": str, "top_k": int, "score_threshold": float,
    #  "channel": str} - every key optional, every value already past its type,
    # range and alphabet bounds. Carried into the inner graph by
    # src/graph/context_bridge.py, because the framework does not forward
    # input_context across a nested-graph boundary.
    caller_context: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Inner layer - domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # InputValidateNode outputs
    # Normalised free-text search query (whitespace-collapsed, length-capped).
    search_query: NotRequired[str]

    # JSON STRING (to_json) of parsed structured query params. Deserialised
    # dict shape: {"category": str | None, "top_k": int | None}.
    # Consumers (RetrieveNode, RerankFilterNode) read it back via from_json().
    query_filters: NotRequired[Optional[str]]

    # Declared `retrieval` block (config/config.yaml) forwarded by
    # NightShiftOpsGraphNode._parent_config() ->
    # DomainWorkflowGraph._extra_initial_state().
    # JSON STRING (to_json) of {"top_k": int, "score_threshold": float,
    # "kb_path": str}. Consumers (RetrieveNode, RerankFilterNode) read it
    # back via from_json().
    retrieval_config: NotRequired[Optional[str]]

    # RetrieveNode output
    # JSON STRING (to_json) of scored KB candidates. Deserialised shape:
    # list[dict], each entry {"id": str, "title": str, "category": str,
    # "source": str, "score": float, "excerpt": str}.
    # Consumers (RerankFilterNode) read it back via from_json().
    retrieved_documents: NotRequired[Optional[str]]

    # RerankFilterNode output
    # JSON STRING (to_json) of reranked + threshold-filtered passages, capped
    # at top_k. Same entry shape as retrieved_documents.
    # Consumers (GenerateAnswerNode) read it back via from_json().
    ranked_documents: NotRequired[Optional[str]]

    # GenerateAnswerNode outputs
    # Rule-assembled grounded answer body with numbered citation markers, OR
    # (the safety boundary) a FIXED escalation instruction when an emergency
    # signal fires - never assembled from KB content in that case.
    grounded_answer: NotRequired[str]

    # JSON STRING (to_json) of citations. Deserialised shape: list[dict],
    # each entry {"ref": int, "id": str, "title": str, "source": str}.
    # Consumers (OutputFormatNode) read it back via from_json().
    citations: NotRequired[Optional[str]]

    # JSON STRING (to_json) of the escalation decision - the safety
    # differentiator. Deserialised shape:
    # {"required": bool, "category": str | None, "channel": str | None,
    # "reason": str | None}. Consumers (OutputFormatNode, and the outer
    # ConvenienceStoreNightShiftOpsAgent.get_output() structured-output
    # override) read it back via from_json().
    escalation: NotRequired[Optional[str]]

    # OutputFormatNode output
    # Final formatted answer (body + sources + escalation section when
    # required + standing disclaimer). Written by OutputFormatNode; surfaced
    # to the outer graph via get_output() -> merge_output().
    formatted_answer: NotRequired[str]

    # Validation / parse notes accumulated during intake (no PII).
    # JSON STRING (to_json) of list[str].
    intake_notes: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Tracing / audit - framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
    error_code: Optional[str]
    # node_history inherited from AgentState; listed here for clarity
    # node_history: Optional[List[str]]
