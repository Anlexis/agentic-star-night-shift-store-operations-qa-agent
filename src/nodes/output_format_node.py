"""AgentCore Platform v1.0"""

# RET-C2-668 - OutputFormatNode
# Domain node 5 (terminal): compose the final formatted answer - the
# grounded answer (or fixed escalation instruction) body, the Sources list,
# an Escalation section when required, and the standing operational
# disclaimer. This composition is part of THIS node's domain output
# contract, not of the outer post_process slot (post_process only gates, it
# does not compose).
#
# Wired by the inner graph (DomainWorkflowGraph). get_output() of the inner
# graph surfaces formatted_answer + citations + escalation + status to the
# outer merge_output().
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json

# Standing operational disclaimer - appended to EVERY answer this template
# emits (both the grounded-answer path and the escalation path).
_OPERATIONAL_DISCLAIMER = (
    "This guidance is generated from the seeded store-operations manual and "
    "does not replace your store's written protocols or a supervisor's "
    "judgement. For an active emergency, escalate immediately - do not wait "
    "for confirmation."
)

_CHANNEL_LABELS: Dict[str, str] = {
    "police_110": "Police - 110",
    "ambulance_119": "Ambulance - 119",
    "fire_119_and_evacuate": "Fire / evacuate - 119",
    "supervisor": "Shift supervisor",
}


class OutputFormatNode(FunctionNode):
    """Compose the final answer: body + sources + escalation + disclaimer.

    Input state keys:
        grounded_answer: answer body with [n] citation markers, or a fixed
                         escalation instruction
        citations:       JSON list [{ref, id, title, source}]
        escalation:      JSON {"required", "category", "channel", "reason"}

    Output state keys (partial dict):
        formatted_answer: final rendered answer string
        status:           AgentStatus.SUCCESS.value (plain string — never write
                          the bare enum to State)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        grounded_answer = state.get("grounded_answer") or ("No answer is available for this request.")
        citations: List[Dict[str, Any]] = from_json(state.get("citations"), []) or []
        escalation: Dict[str, Any] = from_json(state.get("escalation"), {}) or {}
        escalation_required = bool(escalation.get("required", False))

        lines: List[str] = []
        lines.append("# Night Shift Operations Answer")
        lines.append("")
        if escalation_required:
            lines.append("## Escalation Required")
            lines.append("")
        lines.append(grounded_answer)
        lines.append("")

        if escalation_required:
            channel = escalation.get("channel")
            label = _CHANNEL_LABELS.get(str(channel), str(channel) if channel else "Shift supervisor")
            lines.append(f"**Escalate to:** {label}")
            lines.append("")

        lines.append("## Sources")
        if citations:
            for citation in citations:
                if not isinstance(citation, dict):
                    continue
                ref = citation.get("ref", "?")
                title = str(citation.get("title", "")).strip()
                source = str(citation.get("source", "")).strip()
                suffix = f" ({source})" if source else ""
                lines.append(f"- [{ref}] {title}{suffix}")
        else:
            lines.append("- none (no knowledge-base passage cleared the relevance threshold)")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"*{_OPERATIONAL_DISCLAIMER}*")

        formatted_answer = "\n".join(lines)

        # Audit: final answer composed (disclaimer attached).
        emit_trace_event(
            "output_format_complete",
            {
                "answer_chars": len(formatted_answer),
                "citation_count": len(citations),
                "escalation_required": escalation_required,
            },
            state,
        )

        return {
            "formatted_answer": formatted_answer,
            "status": AgentStatus.SUCCESS.value,
        }
