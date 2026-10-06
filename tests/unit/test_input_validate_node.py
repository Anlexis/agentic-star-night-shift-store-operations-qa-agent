# RET-C2-668 — Unit Tests: InputValidateNode (inner domain node 1)
#
# Invocation canon: node(state) via BaseNode.__call__. InputValidateNode is an
# inner Cat-2 domain node and declares TrustLevel.ANONYMOUS (the external
# trust gate lives on the outer backbone pre_process slot).
#
# Reads validated_input | user_input — BOTH are framework PII-scan fields,
# so every payload here is intentionally PII-free (PII masking itself is
# covered by test_pre_process_node.py / test_trust_gate.py).
#
# Mirrors docs/03_test_spec.md INPV-01..INPV-12.
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.input_validate_node import InputValidateNode
from src.schemas.state import from_json, to_json


def _assert_declined(result):
    """A rejection the caller can correct: the run COMPLETES carrying the reason.

    Both halves matter. The status says the calling surface's turn was not
    ended, and the reason code says the request was nonetheless not carried out
    — asserting only the status would pass on a run that quietly answered.

    The refusals that still terminate keep asserting AgentStatus.ERROR; the two
    are deliberately not merged into one predicate.
    """
    assert result["status"] == AgentStatus.SUCCESS.value, result
    assert result.get("error_code"), result




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


class TestPlainTextQuery:
    def test_inpv_01_plain_text_becomes_the_search_query(self):
        result = InputValidateNode()(_make_state(validated_input="what is the cutoff time for utility bill payments"))
        # InputValidateNode's execute() does not set "status" at all — it is a
        # middle domain node; only PreProcessNode / OutputFormatNode /
        # PostProcessNode manage that field. A clean parse leaves it absent.
        assert "status" not in result
        assert result["search_query"] == "what is the cutoff time for utility bill payments"
        filters = from_json(result["query_filters"])
        assert filters == {"category": None, "top_k": None, "score_threshold": None}
        assert "intake_notes" not in result

    def test_inpv_02_falls_back_to_user_input_when_validated_input_absent(self):
        result = InputValidateNode()(_make_state(user_input="how do I log a voided transaction"))
        assert result["search_query"] == "how do I log a voided transaction"

    def test_inpv_03_whitespace_is_collapsed(self):
        result = InputValidateNode()(_make_state(validated_input="how   do i \n\t check id"))
        assert result["search_query"] == "how do i check id"

    def test_inpv_04_query_truncated_over_max_chars(self):
        long_query = "x" * 2100
        result = InputValidateNode()(_make_state(validated_input=long_query))
        assert len(result["search_query"]) == 2000
        notes = from_json(result["intake_notes"], [])
        assert any("truncated" in n for n in notes)


class TestJsonEnvelope:
    def test_inpv_05_parses_query_category_top_k(self):
        payload = '{"query": "utility bill cutoff time", "category": "Bill_Payment", "top_k": 3}'
        result = InputValidateNode()(_make_state(validated_input=payload))
        assert result["search_query"] == "utility bill cutoff time"
        filters = from_json(result["query_filters"])
        assert filters == {"category": "bill_payment", "top_k": 3, "score_threshold": None}

    def test_inpv_06_question_key_is_an_accepted_alias(self):
        payload = '{"question": "how do I check id for tobacco"}'
        result = InputValidateNode()(_make_state(validated_input=payload))
        assert result["search_query"] == "how do I check id for tobacco"

    def test_inpv_07_malformed_json_looking_input_falls_back_to_plain_text(self):
        payload = "{not actually valid json, category: oops"
        result = InputValidateNode()(_make_state(validated_input=payload))
        assert result["search_query"] == payload
        notes = from_json(result["intake_notes"], [])
        assert any("did not parse" in n for n in notes)


class TestEnvelopeNumericGuard:
    """Envelope parameters fail CLOSED — a bad value is a refusal, not a clamp.

    Clamping was the old behaviour and it is the wrong one twice over: it turns
    a caller's mistake into a silently different query, and the coercion path it
    needs is exactly where a non-finite value used to reach int() and raise an
    unhandled OverflowError from inside the node.
    """

    def test_inpv_08_out_of_range_top_k_is_refused(self):
        payload = '{"query": "handover checklist", "top_k": 999}'
        result = InputValidateNode()(_make_state(validated_input=payload))
        _assert_declined(result)
        assert any("top_k" in str(e) for e in result["error_log"])
        assert "query_filters" not in result, "a refused request must carry nothing forward"

    def test_inpv_09_non_numeric_top_k_is_refused(self):
        payload = '{"query": "handover checklist", "top_k": "soon"}'
        result = InputValidateNode()(_make_state(validated_input=payload))
        _assert_declined(result)
        assert any("top_k" in str(e) for e in result["error_log"])

    @pytest.mark.parametrize(
        "literal",
        ["NaN", "Infinity", "-Infinity"],
        ids=["nan", "inf", "neg_inf"],
    )
    def test_inpv_10_non_finite_top_k_is_refused(self, literal):
        # Python's json parses bare NaN / Infinity out of a request body, and
        # int(float("inf")) raises OverflowError — so an unguarded coercion
        # crashes the node instead of refusing the request.
        payload = '{"query": "handover checklist", "top_k": ' + literal + "}"
        result = InputValidateNode()(_make_state(validated_input=payload))
        _assert_declined(result)
        assert any("finite" in str(e) or "between" in str(e) for e in result["error_log"])

    @pytest.mark.parametrize(
        "literal",
        ["NaN", "Infinity", "-Infinity"],
        ids=["nan", "inf", "neg_inf"],
    )
    def test_inpv_11_non_finite_score_threshold_is_refused(self, literal):
        # A NaN relevance floor is the dangerous one: every comparison against
        # NaN is False, so it silently disables the filter this agent's
        # grounding rests on.
        payload = '{"query": "handover checklist", "score_threshold": ' + literal + "}"
        result = InputValidateNode()(_make_state(validated_input=payload))
        _assert_declined(result)

    def test_inpv_12_boolean_top_k_is_refused(self):
        # isinstance(True, int) is True in Python, so a bare int check accepts
        # `true` as 1.
        payload = '{"query": "handover checklist", "top_k": true}'
        result = InputValidateNode()(_make_state(validated_input=payload))
        _assert_declined(result)

    def test_refusal_never_echoes_the_rejected_value(self):
        payload = '{"query": "handover checklist", "category": "SELECT * FROM secrets"}'
        result = InputValidateNode()(_make_state(validated_input=payload))
        _assert_declined(result)
        joined = " ".join(str(e) for e in result["error_log"])
        assert "SELECT" not in joined and "secrets" not in joined
        assert "category" in joined

    def test_valid_score_threshold_is_carried(self):
        payload = '{"query": "handover checklist", "score_threshold": 0.75}'
        result = InputValidateNode()(_make_state(validated_input=payload))
        filters = from_json(result["query_filters"])
        assert filters["score_threshold"] == 0.75


class TestStructuredChannelPrecedence:
    """The validated structured channel beats the in-band JSON envelope."""

    def test_caller_context_wins_over_the_envelope(self):
        payload = '{"query": "handover checklist", "category": "food_safety", "top_k": 2}'
        result = InputValidateNode()(
            _make_state(
                validated_input=payload,
                caller_context=to_json({"category": "shift_handover", "top_k": 7}),
            )
        )
        filters = from_json(result["query_filters"])
        assert filters["category"] == "shift_handover"
        assert filters["top_k"] == 7

    def test_envelope_fills_in_what_the_structured_channel_omits(self):
        payload = '{"query": "handover checklist", "top_k": 2}'
        result = InputValidateNode()(
            _make_state(
                validated_input=payload,
                caller_context=to_json({"category": "shift_handover"}),
            )
        )
        filters = from_json(result["query_filters"])
        assert filters["category"] == "shift_handover"
        assert filters["top_k"] == 2

    def test_absent_caller_context_is_not_an_error(self):
        result = InputValidateNode()(_make_state(validated_input="handover checklist"))
        filters = from_json(result["query_filters"])
        assert filters == {"category": None, "top_k": None, "score_threshold": None}


class TestEmptyInput:
    def test_empty_request_produces_empty_query_and_a_note(self):
        result = InputValidateNode()(_make_state(validated_input=""))
        assert result["search_query"] == ""
        notes = from_json(result["intake_notes"], [])
        assert any("empty request" in n for n in notes)


class TestInputValidateAudit:
    def test_domain_audit_payload(self, monkeypatch):
        import src.nodes.input_validate_node as mod
        from unittest.mock import MagicMock

        spy = MagicMock()
        monkeypatch.setattr(mod, "emit_trace_event", spy)
        InputValidateNode()(
            _make_state(validated_input='{"query": "handover checklist", "category": "shift_handover"}')
        )
        events = [call.args[0] for call in spy.call_args_list]
        assert "input_validate_complete" in events
        payload = spy.call_args_list[events.index("input_validate_complete")].args[1]
        assert payload["has_category_filter"] is True
        assert payload["has_top_k_override"] is False
        assert payload["has_score_threshold_override"] is False


class TestInputValidateTrustDeclaration:
    def test_admits_anonymous(self):
        assert InputValidateNode.required_trust_level is TrustLevel.ANONYMOUS
