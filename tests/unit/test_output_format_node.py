# RET-C2-668 — Unit Tests: OutputFormatNode (inner domain node 5, terminal)
#
# Invocation canon: node(state) via BaseNode.__call__. Inner Cat-2 domain node
# -> TrustLevel.ANONYMOUS. This node's OWN output key is `formatted_answer`
# (NOT `formatted_output` — that key belongs to the outer PostProcessNode,
# which gates state["result"] after NightShiftOpsGraphNode.merge_output()
# maps this node's formatted_answer -> result).
#
# Mirrors docs/03_test_spec.md OUT-01..OUT-08.
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.output_format_node import OutputFormatNode
from src.schemas.state import to_json

_DISCLAIMER_SNIPPET = "does not replace your store's written protocols"


def _make_state(**extra) -> dict:
    state = {
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestNormalFormatting:
    def test_out_01_header_body_sources_and_disclaimer_present(self):
        citations = [{"ref": 1, "id": "kb-004", "title": "utility bill payment", "source": "Store Operations Manual"}]
        result = OutputFormatNode()(
            _make_state(
                grounded_answer="Based on the store operations manual...\n\n[1] utility bill payment: ...",
                citations=to_json(citations),
                escalation=to_json({"required": False, "category": None, "channel": None, "reason": None}),
            )
        )
        out = result["formatted_answer"]
        assert out.startswith("# Night Shift Operations Answer")
        assert "## Escalation Required" not in out
        assert "**Escalate to:**" not in out
        assert "## Sources" in out
        assert "- [1] utility bill payment (Store Operations Manual)" in out
        assert _DISCLAIMER_SNIPPET in out
        assert result["status"] == AgentStatus.SUCCESS.value
        # `type(...) is str`, not isinstance: AgentStatus subclasses str, so
        # isinstance would pass for the enum member this guards against.
        assert type(result["status"]) is str  # noqa: E721

    def test_out_02_no_citations_shows_the_none_placeholder(self):
        result = OutputFormatNode()(
            _make_state(
                grounded_answer="no coverage answer text",
                citations=to_json([]),
                escalation=to_json({"required": False, "category": None, "channel": None, "reason": None}),
            )
        )
        assert "- none (no knowledge-base passage cleared the relevance threshold)" in result["formatted_answer"]

    def test_out_03_missing_grounded_answer_falls_back(self):
        result = OutputFormatNode()(_make_state(citations=to_json([]), escalation=to_json({})))
        assert "No answer is available for this request." in result["formatted_answer"]


class TestEscalationFormatting:
    def test_out_04_escalation_section_and_known_channel_label(self):
        result = OutputFormatNode()(
            _make_state(
                grounded_answer="Do not confront; comply; move to safety; call 110.",
                citations=to_json([]),
                escalation=to_json(
                    {"required": True, "category": "emergency_security", "channel": "police_110", "reason": "x"}
                ),
            )
        )
        out = result["formatted_answer"]
        assert "## Escalation Required" in out
        assert "**Escalate to:** Police - 110" in out

    def test_out_05_all_known_channel_labels(self):
        expected = {
            "police_110": "Police - 110",
            "ambulance_119": "Ambulance - 119",
            "fire_119_and_evacuate": "Fire / evacuate - 119",
            "supervisor": "Shift supervisor",
        }
        for channel, label in expected.items():
            result = OutputFormatNode()(
                _make_state(
                    grounded_answer="fixed instruction",
                    citations=to_json([]),
                    escalation=to_json({"required": True, "category": "x", "channel": channel, "reason": "x"}),
                )
            )
            assert f"**Escalate to:** {label}" in result["formatted_answer"]

    def test_out_06_unmapped_channel_falls_back_to_the_raw_value(self):
        result = OutputFormatNode()(
            _make_state(
                grounded_answer="fixed instruction",
                citations=to_json([]),
                escalation=to_json({"required": True, "category": "x", "channel": "regional_hq", "reason": "x"}),
            )
        )
        assert "**Escalate to:** regional_hq" in result["formatted_answer"]

    def test_out_07_missing_channel_falls_back_to_shift_supervisor(self):
        result = OutputFormatNode()(
            _make_state(
                grounded_answer="fixed instruction",
                citations=to_json([]),
                escalation=to_json({"required": True, "category": "x", "channel": None, "reason": "x"}),
            )
        )
        assert "**Escalate to:** Shift supervisor" in result["formatted_answer"]


class TestOutputFormatAudit:
    def test_out_08_domain_audit_payload(self, monkeypatch):
        import src.nodes.output_format_node as mod
        from unittest.mock import MagicMock

        spy = MagicMock()
        monkeypatch.setattr(mod, "emit_trace_event", spy)
        citations = [{"ref": 1, "id": "kb-004", "title": "t", "source": "s"}]
        OutputFormatNode()(
            _make_state(
                grounded_answer="answer body",
                citations=to_json(citations),
                escalation=to_json({"required": False, "category": None, "channel": None, "reason": None}),
            )
        )
        events = [call.args[0] for call in spy.call_args_list]
        assert "output_format_complete" in events
        payload = spy.call_args_list[events.index("output_format_complete")].args[1]
        assert payload["citation_count"] == 1
        assert payload["escalation_required"] is False


class TestOutputFormatTrustDeclaration:
    def test_admits_anonymous(self):
        assert OutputFormatNode.required_trust_level is TrustLevel.ANONYMOUS
