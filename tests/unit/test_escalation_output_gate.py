# RET-C2-668 — Unit Tests: the escalation-integrity gate at the output boundary
#
# This template's stated output invariant is that an emergency answer is a
# FIXED, looked-up instruction — never assembled from knowledge-base content,
# never from the caller's question. Proving that inside GenerateAnswerNode only
# shows the node agreeing with itself, so PostProcessNode re-checks the
# RENDERED answer against src/services/escalation_policy independently.
#
# The tests below drive the boundary with answers the generator would never
# produce, which is the only way to show the second layer is real rather than
# decorative.
#
# CONTAINMENT is the other half. Returning ERROR without clearing is not
# containment: the framework's output envelope falls back to state["result"]
# even on an error status, so a gate that merely flags would still ship the
# un-gated answer inside the error response.
#
# Mirrors docs/03_test_spec.md EGATE-01..EGATE-08.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import ConvenienceStoreNightShiftOpsAgent
from src.nodes.post_process_node import PostProcessNode, escalation_gate_output
from src.schemas.state import to_json
from src.services.escalation_policy import (
    ESCALATION_MESSAGES,
    ORDINARY_ANSWER_LEAD,
    channel_for,
    instruction_for,
)

_SECURITY_TEXT = ESCALATION_MESSAGES["emergency_security"]
_MEDICAL_TEXT = ESCALATION_MESSAGES["emergency_medical"]


def _rendered(body: str) -> str:
    """A rendered answer in the shape OutputFormatNode produces."""
    return (
        "# Night Shift Operations Answer\n\n## Escalation Required\n\n"
        f"{body}\n\n**Escalate to:** Police - 110\n\n## Sources\n- none\n"
    )


def _state(result_text: str, escalation: dict) -> dict:
    return {
        "result": result_text,
        "escalation": to_json(escalation),
        "ops_answer": result_text,
        "citations": to_json([{"ref": 1, "id": "kb-009", "title": "t", "source": "s"}]),
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }


_FLAGGED = {"required": True, "category": "emergency_security", "channel": "police_110"}


class TestGateFunction:
    def test_egate_01_a_faithful_escalation_answer_passes(self):
        assert escalation_gate_output(_rendered(_SECURITY_TEXT), _FLAGGED) is None

    def test_egate_02_an_ordinary_answer_is_out_of_scope(self):
        ordinary = f"{ORDINARY_ANSWER_LEAD}, the following passages answer the question"
        assert escalation_gate_output(ordinary, {"required": False}) is None

    def test_egate_03_a_generated_emergency_answer_is_caught(self):
        # Flagged as an emergency, but the body is prose rather than the
        # looked-up instruction — exactly what an improvised answer looks like.
        improvised = _rendered("If a robbery is in progress, do not resist and comply with demands.")
        assert escalation_gate_output(improvised, _FLAGGED) == "escalation_instruction_not_verbatim"

    def test_egate_04_the_wrong_category_instruction_is_caught(self):
        assert escalation_gate_output(_rendered(_MEDICAL_TEXT), _FLAGGED) == "escalation_instruction_not_verbatim"

    def test_egate_05_a_blended_answer_is_caught(self):
        # The instruction IS present, but so is the ordinary path's lead line —
        # the one place the caller's own question is echoed back.
        blended = _rendered(f"{_SECURITY_TEXT}\n\n{ORDINARY_ANSWER_LEAD}, the following passages apply")
        assert escalation_gate_output(blended, _FLAGGED) == "escalation_blended_with_generated_answer"

    def test_egate_06_a_flag_without_a_category_is_caught(self):
        assert escalation_gate_output(_rendered(_SECURITY_TEXT), {"required": True}) == "escalation_category_missing"

    @pytest.mark.parametrize("category", ["emergency_security", "emergency_medical", "emergency_disaster"])
    def test_every_category_round_trips_through_the_policy(self, category):
        rendered = _rendered(instruction_for(category))
        assert escalation_gate_output(rendered, {"required": True, "category": category}) is None
        assert channel_for(category)


class TestNodeContainment:
    """EGATE-07 — a violation CLEARS every output-bearing field."""

    def test_a_violating_escalation_answer_is_blocked_and_cleared(self):
        improvised = _rendered("Just handle it however seems best at the time.")
        result = PostProcessNode()(_state(improvised, _FLAGGED))

        assert result["status"] == AgentStatus.ERROR.value
        # The un-gated body must survive nowhere.
        assert "handle it however" not in result["formatted_output"]
        assert "handle it however" not in result["result"]
        assert result["ops_answer"] == ""
        assert result["citations"] is None
        assert result["escalation"] is None

    def test_the_error_carries_no_traceback_and_no_source_path(self):
        improvised = _rendered("Just handle it however seems best at the time.")
        result = PostProcessNode()(_state(improvised, _FLAGGED))
        joined = " ".join(str(e) for e in result["error_log"])
        assert "Traceback" not in joined
        assert ".py" not in joined
        assert "/src/" not in joined
        assert "handle it however" not in joined

    def test_a_faithful_escalation_answer_is_forwarded(self):
        result = PostProcessNode()(_state(_rendered(_SECURITY_TEXT), _FLAGGED))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == _rendered(_SECURITY_TEXT)


class TestEnvelopeContainment:
    """EGATE-08 — nothing un-gated escapes through the response envelope.

    The framework's get_output() falls back to state["result"] even on an error
    status, so this asserts on the ENVELOPE a caller actually receives, not on
    the node's return dict.
    """

    def _envelope(self, state: dict) -> dict:
        agent = ConvenienceStoreNightShiftOpsAgent()
        return agent.get_output(state)

    def test_error_envelope_carries_no_released_text(self):
        leaked = "the un-gated inner answer nobody should see"
        blocked = PostProcessNode()(_state(_rendered(leaked), _FLAGGED))
        # The state a caller's envelope would be built from, after the gate.
        state = {**_state(_rendered(leaked), _FLAGGED), **blocked}
        envelope = self._envelope(state)

        assert envelope["status"] == AgentStatus.ERROR.value
        assert leaked not in str(envelope)
        assert "escalation" not in envelope, "structured fields must be withheld on a non-success"
        assert "citations" not in envelope

    def test_success_envelope_still_surfaces_the_structured_fields(self):
        # The control: the withholding above is a gate, not a permanent hole.
        state = {
            **_state(_rendered(_SECURITY_TEXT), _FLAGGED),
            **PostProcessNode()(_state(_rendered(_SECURITY_TEXT), _FLAGGED)),
        }
        envelope = self._envelope(state)
        assert envelope["status"] == AgentStatus.SUCCESS.value
        assert envelope["escalation"]["required"] is True
        assert envelope["citations"]

    def test_credential_violation_is_contained_in_the_envelope_too(self):
        leaked = "please use " + "sk-" + ("A" * 24) + " to authenticate"
        blocked = PostProcessNode()(_state(leaked, {"required": False}))
        state = {**_state(leaked, {"required": False}), **blocked}
        envelope = self._envelope(state)
        assert envelope["status"] == AgentStatus.ERROR.value
        assert "sk-" + ("A" * 24) not in str(envelope)
