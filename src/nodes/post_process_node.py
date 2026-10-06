"""AgentCore Platform v1.0"""

# RET-C2-668 - PostProcessNode (outer post_process slot; the output boundary).
#
# Reads the final night-shift ops answer from state["result"], populated by
# NightShiftOpsGraphNode.merge_output() from the inner graph's formatted_answer,
# and surfaces it as the finalized output only after TWO INDEPENDENT gates pass.
#
# Gate 1 - disallowed content. The module-level `security_gate_output(content)`
# scan is called from execute(). It RECURSES into nested dict / list / tuple
# structures, so a credential-shaped value nested inside a returned payload
# cannot bypass it by not being a top-level string. The SAME function is reused
# by ConvenienceStoreNightShiftOpsAgent.get_output() (src/graph/graph.py) to
# re-scan the whitelisted `escalation` / `citations` payload before exposing it.
#
# Gate 2 - escalation integrity. This template's stated output invariant is that
# an emergency answer is a FIXED, looked-up instruction: never assembled from
# knowledge-base content, never from the caller's question. Enforcing that
# inside the generating node only proves the node agreed with itself, so the
# boundary re-checks the RENDERED answer against src/services/escalation_policy
# - the same constants, read independently of the state the generator worked
# from. An escalation-flagged answer that does not carry its category's exact
# instruction, or that carries the ordinary path's lead line (the one place the
# caller's own question is echoed back), is not a shippable emergency answer.
#
# On EITHER violation the node returns ERROR and CLEARS every output-bearing
# field. Returning ERROR without clearing is not containment: the framework's
# output envelope falls back to state["result"] even on an error status, so a
# gate that merely flags would still ship the un-gated answer inside the error
# response. Nothing about the rejected content - no offending text, no
# traceback, no source path - travels in the error either.

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, TOO_LONG

from src.schemas.state import from_json
from src.services.escalation_policy import ORDINARY_ANSWER_LEAD, instruction_for

logger = logging.getLogger(__name__)

# Disallowed output content patterns.
# Each tuple: (name, compiled regex) - order matters (most specific first).
_DISALLOWED_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    # API key patterns: sk-..., pk-..., ak-...
    ("api_key", re.compile(r"\b(?:sk|pk|ak)-[A-Za-z0-9]{16,}", re.IGNORECASE)),
    # JWT: three base64url segments separated by dots
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    # Bearer token in Authorization-like context
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/]{20,}", re.IGNORECASE)),
    # Credential assignment patterns
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
]

_SANITISED_STUB = (
    "[OUTPUT BLOCKED by the output gate - the generated answer did not pass its "
    "content checks. Retry the question; if it recurs, ask your shift supervisor.]"
)


def security_gate_output(content: Any) -> Optional[str]:
    """Run the output content gate - RECURSES into nested containers.

    `content` may be a string, or a dict / list / tuple built from a
    JSON-decoded structured field (e.g. the whitelisted `escalation` /
    `citations` payload built by ConvenienceStoreNightShiftOpsAgent.
    get_output()). Every string leaf reachable from `content` is scanned - dict
    KEYS as well as values, list/tuple elements, nested arbitrarily deep.
    Non-string, non-container leaves (int / float / bool / None) are inert and
    skipped.

    Returns the name of the first matched violation, or None if clean.
    """
    if content is None:
        return None
    if isinstance(content, str):
        for name, pattern in _DISALLOWED_PATTERNS:
            if pattern.search(content):
                return name
        return None
    if isinstance(content, dict):
        for key, value in content.items():
            if isinstance(key, str):
                violation = security_gate_output(key)
                if violation:
                    return violation
            violation = security_gate_output(value)
            if violation:
                return violation
        return None
    if isinstance(content, (list, tuple)):
        for item in content:
            violation = security_gate_output(item)
            if violation:
                return violation
        return None
    # int / float / bool / other scalar - nothing to scan.
    return None


def escalation_gate_output(rendered: str, escalation: Dict[str, Any]) -> Optional[str]:
    """Check the rendered answer against the escalation policy, independently.

    Returns the name of the first violation, or None when the answer is
    consistent with what was flagged.

    Only escalation-flagged answers are constrained here. An ordinary answer is
    grounded in retrieved passages and is checked by the content gate above;
    this gate exists for the one output class where the template promises the
    text is a constant rather than a generation.
    """
    if not escalation.get("required"):
        return None

    category = escalation.get("category")
    if not isinstance(category, str) or not category:
        return "escalation_category_missing"

    if instruction_for(category) not in rendered:
        # Flagged as an emergency, but the body is not the looked-up
        # instruction for that category - so it was generated, not looked up.
        return "escalation_instruction_not_verbatim"

    if ORDINARY_ANSWER_LEAD in rendered:
        # The ordinary-path lead line is the only place the caller's question is
        # echoed back. Its presence on an escalation answer means the emergency
        # response was blended with question-derived text.
        return "escalation_blended_with_generated_answer"

    return None


# Reason code -> the sentence the caller reads. A code with no entry falls
# back to the generic one rather than leaking the code itself.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "QUESTION_TOO_LONG": TOO_LONG,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """The output boundary: content gate + escalation-integrity gate.

    Outer backbone post_process slot. Reads state["result"] (the merged
    formatted_answer from NightShiftOpsGraphNode.merge_output()) and applies
    both gates before the response is returned to the caller.

    Input state keys:
        result:     final formatted ops answer (from merge_output)
        escalation: JSON {"required", "category", "channel", "reason"}

    Output state keys (partial dict):
        formatted_output: gated output (unchanged answer if clean; blocked stub
                          on violation)
        result:           gated alongside formatted_output
        ops_answer / citations / escalation: CLEARED on violation, so no
                          un-gated text survives anywhere the response envelope
                          can reach
        status:           AgentStatus.SUCCESS.value or AgentStatus.ERROR.value
                          (plain strings - never the bare enum in State)
        error_log:        (on error) list of error messages, naming the
                          violation class only
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def _blocked(self, violation: str, message: str) -> Dict[str, Any]:
        """Containment: replace the output AND clear every output-bearing field."""
        logger.error("PostProcessNode: OUTPUT BLOCKED - violation type: %s", violation)
        return {
            "formatted_output": _SANITISED_STUB,
            "result": _SANITISED_STUB,
            "ops_answer": "",
            "citations": None,
            "escalation": None,
            "status": AgentStatus.ERROR.value,
            "error_log": [message],
        }

    def execute(self, state: AgentState) -> Dict[str, Any]:
        # A run declined upstream has nothing to format. Render the reason as
        # the caller-facing body and carry the marker onward.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event("post_process_degraded", {"reason": marker}, state)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
                "formatted_output": message,
                "result": message,
            }
        result = state.get("result") or ""

        if not result or not str(result).strip():
            # No answer was generated - forward as-is (non-fatal).
            return {
                "formatted_output": result,
                "status": AgentStatus.SUCCESS.value,
            }

        rendered = str(result)

        violation = security_gate_output(rendered)
        if violation:
            return self._blocked(
                violation,
                f"PostProcessNode: output blocked - disallowed content detected ({violation})",
            )

        escalation: Dict[str, Any] = from_json(state.get("escalation"), {}) or {}
        escalation_violation = escalation_gate_output(rendered, escalation)
        if escalation_violation:
            return self._blocked(
                escalation_violation,
                "PostProcessNode: output blocked - escalation answer failed its "
                f"integrity check ({escalation_violation})",
            )

        # Clean - record that a finalized answer was emitted.
        emit_trace_event(
            "post_process_complete",
            {
                "output_chars": len(rendered),
                "escalation_required": bool(escalation.get("required")),
            },
            state,
        )

        return {
            "formatted_output": result,
            "status": AgentStatus.SUCCESS.value,
        }
