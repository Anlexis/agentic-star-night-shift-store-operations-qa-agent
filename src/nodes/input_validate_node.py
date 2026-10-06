"""AgentCore Platform v1.0"""

# RET-C2-668 - InputValidateNode
# Domain node 1: parse and normalise the incoming night-shift ops question, and
# settle the retrieval parameters for this request.
#
# Two channels can carry those parameters, and this node reconciles them:
#
#   caller_context   the STRUCTURED channel - the invoke call's input_context,
#                    already validated by PreProcessNode and carried across the
#                    graph boundary by src/graph/context_bridge.py. WINS.
#   the question     a JSON envelope inside the question text, for callers with
#                    no structured channel:
#                        {"query": "...", "category": "...", "top_k": N}
#                    Validated HERE, against the same bounds, because a value
#                    arriving this way never passed through PreProcessNode's
#                    contract check.
#
#   plain text       the whole string is the search query.
#
# Envelope parameters fail CLOSED: an out-of-range or non-finite value is a
# refusal, not a clamp. Clamping is how a caller's mistake becomes a silent,
# different query - and how a non-finite value used to reach int() and raise an
# unhandled OverflowError from inside the node.
#
# Categories seeded in config/kb/cvs_night_ops_kb.json: register_pos,
# age_verification, bill_payment, parcel_handling, food_safety,
# equipment_troubleshooting, shift_handover, emergency_security,
# emergency_medical, emergency_disaster.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import json
import re
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json
from src.services.caller_contract import (
    CallerFieldError,
    require_inert_identifier,
    require_score_threshold,
    require_top_k,
)
from src.services.failure_message import INPUT_REJECTED
from src.services.progress import emit_progress
from src.services.security import strip_markup

# Hard cap on the normalised query length (defence in depth on input size).
_MAX_QUERY_CHARS = 2000

_WHITESPACE_RE = re.compile(r"\s+")


class InputValidateNode(FunctionNode):
    """Parse the (possibly JSON-enveloped) request into a normalised query.

    Input state keys:
        validated_input | user_input: identifier-stripped request payload
        caller_context:               validated structured parameters (JSON)

    Output state keys (partial dict):
        search_query:  normalised free-text search query
        query_filters: JSON dict {"category", "top_k", "score_threshold"}
        intake_notes:  (when anomalies were seen) JSON list[str]
        status/error_log: (only when an envelope parameter is refused)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        raw = state.get("validated_input") or state.get("user_input", "")
        notes: List[str] = []

        query = ""
        envelope: Dict[str, Any] = {}

        if isinstance(raw, str) and raw.strip():
            payload: Any = None
            text = raw.strip()
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    notes.append(
                        "InputValidateNode: JSON-looking input did not parse - " "treated as plain text query."
                    )
            if isinstance(payload, dict):
                query = str(payload.get("query") or payload.get("question") or "")
                try:
                    if payload.get("category") is not None:
                        envelope["category"] = require_inert_identifier(payload["category"], "category")
                    if payload.get("top_k") is not None:
                        envelope["top_k"] = require_top_k(payload["top_k"])
                    if payload.get("score_threshold") is not None:
                        envelope["score_threshold"] = require_score_threshold(payload["score_threshold"])
                except CallerFieldError as exc:
                    # Fail closed, naming the field only - never the value.
                    emit_progress(INPUT_REJECTED)
                    return {
                        "status": AgentStatus.SUCCESS.value,
                        "error_code": "INVALID_REQUEST",
                        "error_log": [f"InputValidateNode: rejected caller input - {exc}"],
                    }
            else:
                query = text
        else:
            notes.append("InputValidateNode: empty request - no query to search.")

        # Normalise whitespace and cap length. The markup strip runs again here
        # because a query lifted out of a JSON envelope never passed through the
        # outer node's sanitizer.
        query = _WHITESPACE_RE.sub(" ", strip_markup(query)).strip()
        if len(query) > _MAX_QUERY_CHARS:
            query = query[:_MAX_QUERY_CHARS]
            notes.append(f"InputValidateNode: query truncated to {_MAX_QUERY_CHARS} chars.")

        # The structured channel wins: it is an explicit parameter, while the
        # envelope form is parameters smuggled through a text field.
        caller_context: Dict[str, Any] = from_json(state.get("caller_context"), {}) or {}
        filters: Dict[str, Optional[Any]] = {
            "category": None,
            "top_k": None,
            "score_threshold": None,
        }
        for key in filters:
            if caller_context.get(key) is not None:
                filters[key] = caller_context[key]
            elif envelope.get(key) is not None:
                filters[key] = envelope[key]

        emit_trace_event(
            "input_validate_complete",
            {
                "query_chars": len(query),
                "has_category_filter": filters["category"] is not None,
                "has_top_k_override": filters["top_k"] is not None,
                "has_score_threshold_override": filters["score_threshold"] is not None,
            },
            state,
        )

        out: Dict[str, Any] = {
            "search_query": query,
            "query_filters": to_json(filters),
        }
        if notes:
            out["intake_notes"] = to_json(notes)
        return out
