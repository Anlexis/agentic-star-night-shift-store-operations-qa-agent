# RET-C2-668 — Unit Tests: PreProcessNode (outer pre_process slot; trust + identifier screen)
#
# Invocation canon: every test invokes the node via
# node(state) — BaseNode.__call__ -> trust gate -> PII mask ->
# execute() -> credential gate — never a bare node.execute(state).
# PreProcessNode requires VERIFIED_EXTERNAL, so its behavioural tests build
# the state at that level (the ANONYMOUS rejection lives in
# test_trust_gate.py).
#
# Input-gate layering note (verified against the released agenticstar-agentcore==1.0.0
# wheel — shared.security.pii_detector): the FRAMEWORK input gate masks
# user_input / validated_input / llm_response BEFORE execute() runs — e-mail
# and hyphenated JP phone numbers (0XX-XXXX-XXXX) match the framework's own
# patterns and surface as [MASKED]. The NODE's own _surface_strip_identifiers()
# then runs on the (already-masked) text and additionally catches identifier
# shapes the framework does NOT recognise — an unhyphenated JP phone/ID digit
# run (no separators) — replacing them with [REDACTED]. Intentional-PII tests
# therefore assert the raw identifier is GONE and the applicable marker
# ([MASKED] or [REDACTED], per which layer actually matches) is present.
#
# Mirrors docs/03_test_spec.md PRE-01..PRE-09.
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from unittest.mock import MagicMock

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

import src.nodes.pre_process_node
from src.nodes.pre_process_node import PreProcessNode


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



# Lowercase, no digits/@ — PII-free so the PII mask leaves the payload untouched.
_VALID_QUERY = "a customer wants to buy beer at 2am, how do I check their id"


def _make_state(user_input=_VALID_QUERY, **extra) -> dict:
    state = {
        "user_input": user_input,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPreProcessSuccess:
    def test_pre_01_valid_query_accepted(self):
        result = PreProcessNode()(_make_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: State carries the plain string, never the enum.
        # `type(...) is str`, not isinstance: AgentStatus subclasses str, so
        # isinstance would pass for the enum member this guards against.
        assert type(result["status"]) is str  # noqa: E721
        assert result["validated_input"] == _VALID_QUERY

    def test_enriched_context_carries_channel(self):
        result = PreProcessNode()(_make_state(input_context={"channel": "pos-terminal"}))
        assert result["enriched_context"]["channel"] == "pos-terminal"
        assert result["enriched_context"]["source"] == "ConvenienceStoreNightShiftOpsAgent"

    def test_missing_channel_defaults_to_unknown(self):
        result = PreProcessNode()(_make_state())
        assert result["enriched_context"]["channel"] == "unknown"


class TestPreProcessRejection:
    def test_pre_02_empty_input_is_error(self):
        result = PreProcessNode()(_make_state(user_input=""))
        _assert_declined(result)
        assert result["error_log"]
        # No validated_input is produced on the reject path.
        assert "validated_input" not in result

    def test_whitespace_only_is_error(self):
        result = PreProcessNode()(_make_state(user_input="   \n\t "))
        _assert_declined(result)

    def test_pre_03_missing_user_input_is_error(self):
        state = _make_state()
        del state["user_input"]
        result = PreProcessNode()(state)
        _assert_declined(result)

    def test_non_string_input_is_error(self):
        result = PreProcessNode()(_make_state(user_input={"malicious": "dict"}))
        _assert_declined(result)


class TestPreProcessIdentifierScreen:
    """PRE-04..PRE-06: raw identifiers never survive into validated_input."""

    def test_pre_04_unhyphenated_phone_digit_run_redacted_by_node_screen(self):
        # No separators -> outside the framework's phone_jp pattern (which
        # requires a '-'/space between groups) -- caught only by the node's
        # own surface strip ([REDACTED] path). Verified empirically against
        # the released wheel's shared.security.pii_detector.
        raw = "call the customer back at 09012345678 to confirm"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "09012345678" not in vi
        assert "[REDACTED]" in vi

    def test_pre_05_hyphenated_jp_phone_masked_by_framework_s2_gate(self):
        # Hyphenated 0XX-XXXX-XXXX matches the framework's own phone_jp
        # pattern and is masked before execute() ever sees it.
        raw = "call the customer back at 090-1234-5678 to confirm"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert "090-1234-5678" not in vi
        assert "[MASKED]" in vi

    def test_pre_06_email_masked_by_framework_s2_gate(self):
        raw = "escalate to shift.lead@example.com right away"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert "shift.lead@example.com" not in vi
        assert "[MASKED]" in vi

    def test_pre_07_long_membership_style_digit_run_redacted_by_node_screen(self):
        # A 14-digit run (outside credit_card's exact-16 and my_number_jp's
        # exact-12 shapes) is not touched by the framework gate; the node's
        # own 4-4-(2..11) pattern catches it.
        raw = "the membership card number is 12345678901234 on file"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "12345678901234" not in vi
        assert "[REDACTED]" in vi


class TestPreProcessAudit:
    def test_pre_08_domain_audit_payload(self, monkeypatch):
        """Audit: the accepted request emits pre_process_complete; the assertion
        targets call.args[1] — the event payload — never the whole call repr."""
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.pre_process_node, "emit_trace_event", spy)
        PreProcessNode()(_make_state())
        events = [call.args[0] for call in spy.call_args_list]
        assert "pre_process_complete" in events
        payload = spy.call_args_list[events.index("pre_process_complete")].args[1]
        assert payload["input_chars"] == len(_VALID_QUERY)


class TestPreProcessTrustDeclaration:
    def test_pre_09_requires_verified_external(self):
        assert PreProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL
