"""AgentCore Platform v1.0"""

# RET-C2-668 - PreProcessNode (outer pre_process slot).
#
# Node contract: extend FunctionNode; implement execute(self, state) -> dict -
# no config parameter. Return ONLY the fields this node changes, never full
# state. Read input_context via state.get("input_context", {}) - read-only.
# Never import from mediator/, api/, or another agent.
#
# This node OWNS the caller-data contract. Everything a caller can send arrives
# on one of two channels and both are handled here before anything reaches the
# domain workflow:
#
#   user_input      the operational question, free text
#   input_context   structured parameters: category, top_k, score_threshold,
#                   channel (see src/services/caller_contract.py)
#
# Three things happen, in this order, and each of them fails CLOSED:
#
#   1. the question must be present and be text;
#   2. instruction-override content - directives aimed at a model, including
#      chat-template control tokens - is REFUSED on BOTH channels. The
#      structured channel is scanned depth-first, keys included, on the PARSED
#      payload, so escaping in the request body cannot carry a directive past
#      it. The screen is the template's own (src/services/security.py) and runs
#      inside execute(), so calling execute() directly still refuses: the
#      guarantee does not depend on any platform gate being present or
#      configured on;
#   3. every structured field is validated against explicit type, range and
#      alphabet bounds. A refusal names the FIELD, never the value, and an
#      unrecognised field NAME is masked rather than echoed back.
#
# A surface identifier screen then redacts direct identifiers (phone numbers,
# long membership/employee-ID-like digit runs, e-mail) from the question, so raw
# identifiers never reach the domain nodes or a checkpoint.

import re
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import to_json
from src.services.caller_contract import (
    CallerFieldError,
    find_instruction_override,
    validate_caller_context,
)
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED
from src.services.progress import emit_progress
from src.services.security import contains_instruction_override, sanitize_question

# Surface-level identifier patterns redacted before validated_input is written.
# Downstream domain nodes only ever operate on the normalised question text and
# knowledge-base passage summaries, never raw phone numbers or ID-like numbers.
_PII_PATTERNS: List["re.Pattern[str]"] = [
    # Japanese phone numbers (mobile/landline, optionally hyphenated):
    # e.g. 090-1234-5678, 03-1234-5678.
    re.compile(r"\b0\d{1,4}[- ]?\d{1,4}[- ]?\d{3,4}\b"),
    # Long identifier-like digit runs (10-19 digits, optionally grouped) -
    # membership numbers, employee IDs, card-like numbers.
    re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{2,11}\b"),
    # E-mail addresses.
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
]
_PII_REPLACEMENT = "[REDACTED]"


def _surface_strip_identifiers(text: str) -> str:
    """Redact obvious direct-identifier tokens from a free-text string."""
    for pattern in _PII_PATTERNS:
        text = pattern.sub(_PII_REPLACEMENT, text)
    return text


class PreProcessNode(FunctionNode):
    """Validate the caller contract and shape the request for the inner graph.

    Rejects empty / invalid input and instruction-override content before the
    domain workflow runs, and surface-strips phone numbers, ID-like digit runs
    and e-mail addresses from the payload.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        input_context = state.get("input_context", {})  # read-only

        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            emit_progress(EMPTY_INPUT)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "EMPTY_INPUT",
                "error_log": ["PreProcessNode: user_input is empty or missing"],
            }

        # Instruction-override screen, fail CLOSED. A refusal leaves no
        # validated_input and no caller_context, so nothing downstream can pick
        # up a partially-accepted request.
        #
        # Both screens below TERMINATE, unlike the checks around them. Those
        # describe a value the caller can correct and resend; refused content is
        # not a value to correct, and reporting it the same way would read as an
        # invitation to reword the request until it gets through. The two are
        # separated by which check fires, never by the wording of a message.
        if contains_instruction_override(user_input):
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "instruction_override", "where": "user_input"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["PreProcessNode: request refused - instruction-override content in user_input"],
            }

        override_path = find_instruction_override(input_context) if isinstance(input_context, (dict, list)) else None
        if override_path is not None:
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "instruction_override", "where": override_path},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PreProcessNode: request refused - instruction-override content in {override_path}"],
            }

        try:
            caller_context = validate_caller_context(input_context)
        except CallerFieldError as exc:
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "caller_field", "where": "input_context"},
                state,
            )
            # Fail closed, naming the field only - the rejected value is never
            # echoed into the log.
            emit_progress(INPUT_REJECTED)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "INVALID_REQUEST",
                "error_log": [f"PreProcessNode: rejected caller input - {exc}"],
            }

        validated_input = _surface_strip_identifiers(sanitize_question(user_input.strip()))

        # A night-shift ops question was accepted and surface-redacted (no
        # direct identifiers in the payload). Field presence only, never text.
        emit_trace_event(
            "pre_process_complete",
            {
                "input_chars": len(validated_input),
                "caller_fields": sorted(caller_context),
            },
            state,
        )

        return {
            "validated_input": validated_input,
            # Stored as a JSON string (every structured state value is
            # msgpack-safe); the graph node reads it back at the boundary and
            # puts it on the bridge into the inner graph.
            "caller_context": to_json(caller_context),
            "enriched_context": {
                "source": "ConvenienceStoreNightShiftOpsAgent",
                "channel": caller_context.get("channel", "unknown"),
            },
            "status": AgentStatus.SUCCESS.value,
        }
