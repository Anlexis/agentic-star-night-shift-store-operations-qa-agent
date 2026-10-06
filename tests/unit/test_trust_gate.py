# RET-C2-668 — Unit Tests: trust gate (bootstrap seed suite, extended at implementation)
#
# This suite retargets an earlier single-node seed (superseded by the 5
# domain nodes) to InputValidateNode / the domain nodes, and
# ADDS TestSafetyEscalationBoundary — proof of the safety-escalation
# boundary: for a safety/security/emergency
# class, GenerateAnswerNode must return escalation routing, never an
# improvised answer.
#
# Trust-gate contract: tests must invoke nodes via node(state) —
# through BaseNode.__call__, which runs the trust gate → PII mask → execute()
# → output gate — never via node.execute(state) directly, which bypasses the gate.
# A denial RETURNS an error dict (never raises) with status ERROR and
# "trust gate denied" in error_log; execute() never runs, so execute-only
# output keys are ABSENT from the returned dict.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json, to_json


def _make_state(
    trust_value: str, user_input: str = "a customer wants to buy beer at 2am, how do I check their ID?", **extra
) -> dict:
    state = {
        "user_input": user_input,
        "caller_trust_level": trust_value,
        "node_history": [],
        "error_log": [],
        "session_id": "test-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestTrustGate:
    """trust gate tests — all invocations go through node(state) / __call__."""

    def test_anonymous_caller_allowed_on_inner_node(self):
        """Trust gate: an ANONYMOUS caller passes an ANONYMOUS inner domain node."""
        node = InputValidateNode()  # required_trust_level = ANONYMOUS
        result = node(_make_state(TrustLevel.ANONYMOUS.value, validated_input="how do I check ID for alcohol sales"))
        assert "trust gate denied" not in str(result.get("error_log", []))
        assert result.get("search_query"), "inner node should produce a normalised query"

    def test_anonymous_caller_denied_on_pre_process(self):
        """trust-gate rejection: ANONYMOUS caller on the VERIFIED_EXTERNAL PreProcessNode.

        __call__ must RETURN an error dict (never raise) with status ERROR and
        'trust gate denied' in the error_log. execute() never ran, so the
        execute-only output key (validated_input) must be ABSENT.
        """
        node = PreProcessNode()  # required_trust_level = VERIFIED_EXTERNAL
        result = node(_make_state(TrustLevel.ANONYMOUS.value))
        assert result.get("status") == AgentStatus.ERROR.value
        error_log = result.get("error_log", [])
        assert any(
            "trust gate denied" in str(e) for e in error_log
        ), f"Expected 'trust gate denied' in error_log, got: {error_log}"
        assert "validated_input" not in result, "execute() must not run on a trust-gate denial — validated_input leaked"

    def test_verified_external_caller_passes_pre_process(self):
        """Trust gate: a VERIFIED_EXTERNAL caller clears the pre_process gate and
        the node writes the identifier-stripped validated_input."""
        node = PreProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value))
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("validated_input")

    def test_pre_process_empty_input_rejected_after_gate(self):
        """The trust gate passes, then the node's own input validation declines empty input.

        An empty request is a value the caller can correct, so the node
        completes carrying the reason rather than terminating. What this test is
        about — the gate let the caller through and the NODE did the rejecting —
        is unchanged, and the reason code is what proves the node rejected."""
        node = PreProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value, user_input=""))
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("error_code") == "EMPTY_INPUT"
        assert any("empty" in str(e) for e in result.get("error_log", []))

    def test_anonymous_caller_denied_on_post_process(self):
        """trust-gate rejection on the other VERIFIED_EXTERNAL outer slot (post_process).

        The denial dict carries no execute-only key (formatted_output ABSENT).
        """
        node = PostProcessNode()  # required_trust_level = VERIFIED_EXTERNAL
        result = node(
            _make_state(
                TrustLevel.ANONYMOUS.value,
                result="a clean night-shift ops answer",
            )
        )
        assert result.get("status") == AgentStatus.ERROR.value
        assert any("trust gate denied" in str(e) for e in result.get("error_log", []))
        assert (
            "formatted_output" not in result
        ), "execute() must not run on a trust-gate denial — formatted_output leaked"

    def test_verified_external_caller_passes_post_process(self):
        """Trust gate: a VERIFIED_EXTERNAL caller clears the post_process gate."""
        node = PostProcessNode()
        result = node(
            _make_state(
                TrustLevel.VERIFIED_EXTERNAL.value,
                result="a clean night-shift ops answer",
            )
        )
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("formatted_output")


class TestOutputGate:
    """output gate — the post_process scan is real, not a no-op.

    Invoked through node(state) so the whole chain runs, same as production.
    """

    def test_credential_bearing_output_is_blocked(self):
        """A credential-shaped result is replaced by the sanitised stub.

        The token is assembled at runtime so no credential-shaped literal is
        stored in the repository.
        """
        leaked = "please use " + "sk-" + ("A" * 24) + " to authenticate"
        node = PostProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value, result=leaked))
        assert result.get("status") == AgentStatus.ERROR.value
        assert "sk-" not in str(
            result.get("formatted_output", "")
        ), "output gate must not pass the credential through to formatted_output"
        assert any(
            "output blocked" in str(e) for e in result.get("error_log", [])
        ), f"Expected an output-gate block in error_log, got: {result.get('error_log')}"

    def test_clean_output_passes_the_gate(self):
        """A clean result is returned unchanged — the gate is a filter, not a wall."""
        clean = "Store manuals require ID verification for any customer who appears under 40."
        node = PostProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value, result=clean))
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("formatted_output") == clean


class TestTrustLevelMatrix:
    """The template's declared trust matrix (docs/02_design.md §Security Gates).

    Outer S-gate slots require VERIFIED_EXTERNAL (the manifest's
    required_trust_level); the five inner domain nodes run behind that outer
    boundary and are declared ANONYMOUS per the Cat-2 nested convention.
    """

    def test_outer_gate_nodes_require_verified_external(self):
        assert PreProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL
        assert PostProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL

    def test_inner_domain_nodes_admit_anonymous(self):
        for node_cls in (
            InputValidateNode,
            RetrieveNode,
            RerankFilterNode,
            GenerateAnswerNode,
            OutputFormatNode,
        ):
            assert node_cls.required_trust_level is TrustLevel.ANONYMOUS, (
                f"{node_cls.__name__} must declare TrustLevel.ANONYMOUS " "(inner Cat-2 domain node)"
            )


class TestSafetyEscalationBoundary:
    """Proves the safety-escalation boundary:

    for a safety/security/medical/disaster emergency class, GenerateAnswerNode
    must return escalation routing — a FIXED instruction, never assembled
    from KB content or the caller's query — and set escalation.required=True.
    Two independent detection signals are each tested in isolation, plus the
    normal (non-emergency) path is confirmed unaffected.
    """

    def test_kb_category_signal_triggers_escalation(self):
        """Signal 1: the top-ranked passage's KB category is an emergency
        category — escalation fires even with a neutral-sounding query."""
        node = GenerateAnswerNode()  # required_trust_level = ANONYMOUS
        ranked = [
            {
                "id": "kb-009",
                "title": "強盗 robbery or threatening customer response protocol",
                "category": "emergency_security",
                "source": "Emergency Protocol Manual, security incidents chapter",
                "score": 0.9,
                "excerpt": "If a robbery is in progress, or a customer is behaving in a threatening or armed manner, do not resist...",
            }
        ]
        result = node(
            _make_state(
                TrustLevel.ANONYMOUS.value,
                search_query="what should I do right now",
                ranked_documents=to_json(ranked),
            )
        )
        escalation = from_json(result.get("escalation"), {})
        assert escalation.get("required") is True
        assert escalation.get("category") == "emergency_security"
        assert escalation.get("channel") == "police_110"
        assert "kb-009" not in result.get(
            "grounded_answer", ""
        ), "escalation text must be a fixed instruction, not an excerpt render"
        assert (
            "do not resist" not in result.get("grounded_answer", "").lower()
        ), "escalation text must not be copied from the KB passage content"

    def test_escalation_text_is_query_independent(self):
        """The escalation instruction is a lookup, not a generation: the SAME
        category yields the IDENTICAL grounded_answer regardless of the
        caller's exact wording — proof it is not improvised per request."""
        node = GenerateAnswerNode()
        ranked = [
            {
                "id": "kb-010",
                "title": "急病 sudden medical emergency response protocol",
                "category": "emergency_medical",
                "source": "Emergency Protocol Manual, medical incidents chapter",
                "score": 0.9,
                "excerpt": "Call an ambulance at 119 immediately.",
            }
        ]
        result_a = node(
            _make_state(
                TrustLevel.ANONYMOUS.value,
                search_query="my coworker just collapsed and is not moving",
                ranked_documents=to_json(ranked),
            )
        )
        result_b = node(
            _make_state(
                TrustLevel.ANONYMOUS.value,
                search_query="someone is unresponsive on the floor near the register",
                ranked_documents=to_json(ranked),
            )
        )
        assert result_a.get("grounded_answer") == result_b.get("grounded_answer")
        escalation_a = from_json(result_a.get("escalation"), {})
        assert escalation_a.get("channel") == "ambulance_119"

    def test_keyword_safety_net_triggers_without_kb_match(self):
        """Signal 2: an emergency keyword in the raw query still escalates
        even when retrieval/rerank surfaced NOTHING (ranked_documents empty) —
        a retrieval miss must not suppress escalation."""
        node = GenerateAnswerNode()
        result = node(
            _make_state(
                TrustLevel.ANONYMOUS.value,
                search_query="there's a fire near the back storage room",
                ranked_documents=to_json([]),
            )
        )
        escalation = from_json(result.get("escalation"), {})
        assert escalation.get("required") is True
        assert escalation.get("category") == "emergency_disaster"
        assert escalation.get("channel") == "fire_119_and_evacuate"

    def test_japanese_keyword_safety_net_triggers(self):
        """The keyword safety net also matches the Japanese emergency terms
        named in the proposal (強盗・急病・災害)."""
        node = GenerateAnswerNode()
        result = node(
            _make_state(
                TrustLevel.ANONYMOUS.value,
                search_query="強盗が来ました、どうすればいいですか",
                ranked_documents=to_json([]),
            )
        )
        escalation = from_json(result.get("escalation"), {})
        assert escalation.get("required") is True
        assert escalation.get("category") == "emergency_security"

    def test_normal_question_is_not_escalated(self):
        """A normal operational question takes the grounded-answer path —
        the safety boundary must not over-trigger on routine questions."""
        node = GenerateAnswerNode()
        ranked = [
            {
                "id": "kb-004",
                "title": "収納代行 utility bill payment acceptance and cutoff",
                "category": "bill_payment",
                "source": "Store Operations Manual, bill payment agency (収納代行) chapter",
                "score": 0.8,
                "excerpt": "収納代行 accepts utility, tax, and insurance bills presented with a valid barcode.",
            }
        ]
        result = node(
            _make_state(
                TrustLevel.ANONYMOUS.value,
                search_query="what is the cutoff time for 収納代行 bill payments",
                ranked_documents=to_json(ranked),
            )
        )
        escalation = from_json(result.get("escalation"), {})
        assert escalation.get("required") is False
        assert "[1]" in result.get("grounded_answer", "")

    def test_generate_answer_node_admits_anonymous(self):
        assert GenerateAnswerNode.required_trust_level is TrustLevel.ANONYMOUS
