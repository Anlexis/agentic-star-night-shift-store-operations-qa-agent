# RET-C2-668 — Unit Tests: GenerateAnswerNode (inner domain node 4)
#
# Invocation canon: node(state) via BaseNode.__call__. Inner Cat-2 domain node
# -> TrustLevel.ANONYMOUS.
#
# Scope: this file covers the NON-emergency grounded-answer assembly path,
# citation numbering, and the no-coverage fallback. THE SAFETY BOUNDARY itself
# ( KB-category signal, keyword safety net, and
# proof that the escalation text is a fixed lookup never assembled from KB
# content) is already exhaustively covered by
# tests/unit/test_trust_gate.py::TestSafetyEscalationBoundary (the
# seed suite) — deliberately not duplicated here.
#
# Mirrors docs/03_test_spec.md GEN-01..GEN-07.
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.schemas.state import from_json, to_json


def _doc(doc_id, category, title, source="Store Operations Manual", excerpt="excerpt text") -> dict:
    return {
        "id": doc_id,
        "title": title,
        "category": category,
        "source": source,
        "score": 0.8,
        "excerpt": excerpt,
    }


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


class TestGroundedAnswerAssembly:
    def test_gen_01_single_passage_gets_one_numbered_citation(self):
        ranked = [_doc("kb-004", "bill_payment", "収納代行 utility bill payment acceptance and cutoff")]
        result = GenerateAnswerNode()(
            _make_state(
                search_query="what is the cutoff time for utility bill payments",
                ranked_documents=to_json(ranked),
            )
        )
        assert "[1]" in result["grounded_answer"]
        citations = from_json(result["citations"])
        assert citations == [
            {
                "ref": 1,
                "id": "kb-004",
                "title": "収納代行 utility bill payment acceptance and cutoff",
                "source": "Store Operations Manual",
            }
        ]
        escalation = from_json(result["escalation"])
        assert escalation == {"required": False, "category": None, "channel": None, "reason": None}

    def test_gen_02_multiple_passages_are_numbered_in_order(self):
        ranked = [
            _doc("kb-002", "age_verification", "Age verification for alcohol sales"),
            _doc("kb-003", "age_verification", "Age verification for tobacco sales"),
        ]
        result = GenerateAnswerNode()(_make_state(search_query="age verification", ranked_documents=to_json(ranked)))
        assert "[1]" in result["grounded_answer"]
        assert "[2]" in result["grounded_answer"]
        citations = from_json(result["citations"])
        assert [c["ref"] for c in citations] == [1, 2]
        assert [c["id"] for c in citations] == ["kb-002", "kb-003"]

    def test_gen_03_lead_sentence_quotes_the_query(self):
        ranked = [_doc("kb-001", "register_pos", "POS register total mismatch")]
        result = GenerateAnswerNode()(_make_state(search_query="how do i log a void", ranked_documents=to_json(ranked)))
        assert '"how do i log a void"' in result["grounded_answer"]

    def test_gen_04_missing_query_uses_the_generic_lead_in(self):
        ranked = [_doc("kb-001", "register_pos", "POS register total mismatch")]
        result = GenerateAnswerNode()(_make_state(search_query="", ranked_documents=to_json(ranked)))
        assert "the most relevant passages are:" in result["grounded_answer"]


class TestNoCoverage:
    def test_gen_05_empty_ranked_documents_yields_the_fixed_no_coverage_message(self):
        result = GenerateAnswerNode()(
            _make_state(search_query="quantum telepathy sandwich recipes", ranked_documents=to_json([]))
        )
        assert result["grounded_answer"] == (
            "The store operations manual does not contain sufficient coverage to "
            "answer this question. Rephrase the query with more specific terms, or "
            "ask your shift supervisor for a manual review."
        )
        assert from_json(result["citations"]) == []
        escalation = from_json(result["escalation"])
        assert escalation["required"] is False

    def test_gen_06_missing_ranked_documents_key_also_yields_no_coverage(self):
        result = GenerateAnswerNode()(_make_state(search_query="anything"))
        assert "does not contain sufficient coverage" in result["grounded_answer"]


class TestGenerateAnswerAudit:
    def test_gen_07_domain_audit_payload(self, monkeypatch):
        import src.nodes.generate_answer_node as mod
        from unittest.mock import MagicMock

        spy = MagicMock()
        monkeypatch.setattr(mod, "emit_trace_event", spy)
        ranked = [_doc("kb-004", "bill_payment", "utility bill payment")]
        GenerateAnswerNode()(_make_state(search_query="bill payment", ranked_documents=to_json(ranked)))
        events = [call.args[0] for call in spy.call_args_list]
        assert "generate_answer_complete" in events
        payload = spy.call_args_list[events.index("generate_answer_complete")].args[1]
        assert payload["citation_count"] == 1
        assert payload["escalation_required"] is False
        assert payload["no_coverage"] is False


class TestGenerateAnswerTrustDeclaration:
    def test_admits_anonymous(self):
        assert GenerateAnswerNode.required_trust_level is TrustLevel.ANONYMOUS
