# RET-C2-668 — Unit Tests: RerankFilterNode (inner domain node 3)
#
# Invocation canon: node(state) via BaseNode.__call__. Inner Cat-2 domain node
# -> TrustLevel.ANONYMOUS. Fixtures are hand-built candidate lists (isolating
# this node from RetrieveNode) mirroring the real retrieved_documents shape
# {id, title, category, source, score, excerpt}.
#
# Mirrors docs/03_test_spec.md RRK-01..RRK-07.
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.rerank_filter_node import RerankFilterNode
from src.schemas.state import from_json, to_json


def _doc(doc_id, category, score, title="t") -> dict:
    return {
        "id": doc_id,
        "title": title,
        "category": category,
        "source": "Store Operations Manual",
        "score": score,
        "excerpt": "excerpt text",
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


class TestScoreThresholdFilter:
    def test_rrk_01_below_threshold_candidate_is_dropped(self):
        candidates = [_doc("kb-A", "bill_payment", 0.6), _doc("kb-B", "parcel_handling", 0.3)]
        result = RerankFilterNode()(_make_state(retrieved_documents=to_json(candidates)))
        kept_ids = [c["id"] for c in from_json(result["ranked_documents"])]
        assert kept_ids == ["kb-A"]

    def test_rrk_02_nothing_survives_returns_empty_list(self):
        candidates = [_doc("kb-A", "bill_payment", 0.1)]
        result = RerankFilterNode()(_make_state(retrieved_documents=to_json(candidates)))
        assert from_json(result["ranked_documents"]) == []

    def test_rrk_03_configurable_score_threshold(self):
        candidates = [_doc("kb-A", "bill_payment", 0.3)]
        result = RerankFilterNode()(
            _make_state(
                retrieved_documents=to_json(candidates),
                retrieval_config=to_json({"score_threshold": 0.2}),
            )
        )
        assert [c["id"] for c in from_json(result["ranked_documents"])] == ["kb-A"]


class TestCategoryBoost:
    def test_rrk_04_category_match_boost_can_push_a_candidate_over_threshold(self):
        # 0.45 alone would be dropped at the default 0.5 threshold; the +0.1
        # category-match boost pushes it to 0.55, which survives.
        candidates = [_doc("kb-A", "age_verification", 0.45)]
        result = RerankFilterNode()(
            _make_state(
                retrieved_documents=to_json(candidates),
                query_filters=to_json({"category": "age_verification"}),
            )
        )
        ranked = from_json(result["ranked_documents"])
        assert [c["id"] for c in ranked] == ["kb-A"]
        assert ranked[0]["score"] == 0.55

    def test_rrk_05_boost_does_not_apply_to_a_non_matching_category(self):
        candidates = [_doc("kb-A", "bill_payment", 0.45)]
        result = RerankFilterNode()(
            _make_state(
                retrieved_documents=to_json(candidates),
                query_filters=to_json({"category": "age_verification"}),
            )
        )
        assert from_json(result["ranked_documents"]) == []


class TestTopKCap:
    def test_rrk_06_caps_survivors_at_configured_top_k(self):
        candidates = [_doc(f"kb-{i}", "bill_payment", 0.9 - i * 0.01) for i in range(5)]
        result = RerankFilterNode()(
            _make_state(
                retrieved_documents=to_json(candidates),
                retrieval_config=to_json({"top_k": 2}),
            )
        )
        assert len(from_json(result["ranked_documents"])) == 2

    def test_rrk_07_stricter_caller_top_k_override_wins(self):
        candidates = [_doc(f"kb-{i}", "bill_payment", 0.9 - i * 0.01) for i in range(5)]
        result = RerankFilterNode()(
            _make_state(
                retrieved_documents=to_json(candidates),
                query_filters=to_json({"top_k": 1}),
            )
        )
        assert len(from_json(result["ranked_documents"])) == 1

    def test_looser_caller_top_k_override_is_ignored(self):
        candidates = [_doc(f"kb-{i}", "bill_payment", 0.9 - i * 0.01) for i in range(5)]
        result = RerankFilterNode()(
            _make_state(
                retrieved_documents=to_json(candidates),
                query_filters=to_json({"top_k": 10}),  # looser than the default (5) -> ignored
            )
        )
        assert len(from_json(result["ranked_documents"])) == 5

    def test_deterministic_tie_break_by_id_ascending(self):
        candidates = [_doc("kb-9", "bill_payment", 0.7), _doc("kb-1", "bill_payment", 0.7)]
        result = RerankFilterNode()(_make_state(retrieved_documents=to_json(candidates)))
        assert [c["id"] for c in from_json(result["ranked_documents"])] == ["kb-1", "kb-9"]


class TestRerankAudit:
    def test_domain_audit_payload(self, monkeypatch):
        import src.nodes.rerank_filter_node as mod
        from unittest.mock import MagicMock

        spy = MagicMock()
        monkeypatch.setattr(mod, "emit_trace_event", spy)
        candidates = [_doc("kb-A", "bill_payment", 0.9), _doc("kb-B", "bill_payment", 0.1)]
        RerankFilterNode()(_make_state(retrieved_documents=to_json(candidates)))
        events = [call.args[0] for call in spy.call_args_list]
        assert "rerank_filter_complete" in events
        payload = spy.call_args_list[events.index("rerank_filter_complete")].args[1]
        assert payload["kept"] == 1
        assert payload["dropped"] == 1


class TestRerankTrustDeclaration:
    def test_admits_anonymous(self):
        assert RerankFilterNode.required_trust_level is TrustLevel.ANONYMOUS
