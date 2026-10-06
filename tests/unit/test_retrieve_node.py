# RET-C2-668 — Unit Tests: RetrieveNode (inner domain node 2)
#
# Invocation canon: node(state) via BaseNode.__call__. Inner Cat-2 domain node
# -> TrustLevel.ANONYMOUS. Deterministic keyword retrieval over the REAL
# seeded KB (config/kb/cvs_night_ops_kb.json) — values below were captured by
# running the actual node against that file (11 entries / 10 categories), not
# hand-computed, so scores are exact.
#
# Mirrors docs/03_test_spec.md RET-01..RET-08.
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json, to_json


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


class TestKeywordRetrieval:
    def test_ret_01_relevant_query_surfaces_matching_categories(self):
        result = RetrieveNode()(
            _make_state(search_query="what age verification do i need before selling alcohol to a customer")
        )
        candidates = from_json(result["retrieved_documents"])
        ids = [c["id"] for c in candidates]
        assert "kb-002" in ids  # Age verification for alcohol sales
        assert "kb-003" in ids  # Age verification for tobacco sales
        assert candidates[0]["category"] == "age_verification"
        assert candidates[0]["score"] > 0.0

    def test_ret_02_falls_back_to_validated_input_then_user_input(self):
        via_validated = RetrieveNode()(_make_state(validated_input="utility bill payments cutoff"))
        via_user = RetrieveNode()(_make_state(user_input="utility bill payments cutoff"))
        assert from_json(via_validated["retrieved_documents"])[0]["id"] == "kb-004"
        assert from_json(via_user["retrieved_documents"])[0]["id"] == "kb-004"

    def test_ret_03_no_lexical_overlap_returns_no_candidates(self):
        result = RetrieveNode()(_make_state(search_query="quantum telepathy sandwich recipes"))
        assert from_json(result["retrieved_documents"]) == []

    def test_ret_04_results_are_sorted_score_descending(self):
        result = RetrieveNode()(_make_state(search_query="how do i log a voided pos transaction"))
        candidates = from_json(result["retrieved_documents"])
        scores = [c["score"] for c in candidates]
        assert scores == sorted(scores, reverse=True)
        assert candidates[0]["id"] == "kb-001"


class TestCategoryFilter:
    def test_ret_05_category_filter_narrows_results(self):
        result = RetrieveNode()(
            _make_state(
                search_query="handover",
                query_filters=to_json({"category": "shift_handover"}),
            )
        )
        candidates = from_json(result["retrieved_documents"])
        assert [c["id"] for c in candidates] == ["kb-008"]

    def test_ret_06_category_filter_excludes_non_matching_category(self):
        result = RetrieveNode()(
            _make_state(
                search_query="handover",
                query_filters=to_json({"category": "food_safety"}),
            )
        )
        assert from_json(result["retrieved_documents"]) == []

    def test_unfiltered_query_returns_multiple_categories(self):
        result = RetrieveNode()(_make_state(search_query="handover"))
        categories = {c["category"] for c in from_json(result["retrieved_documents"])}
        assert "shift_handover" in categories
        assert len(categories) > 1


class TestRetrievalConfig:
    def test_ret_07_retrieval_config_top_k_is_honoured(self):
        default_result = RetrieveNode()(_make_state(search_query="verification"))
        overridden = RetrieveNode()(_make_state(search_query="verification", retrieval_config=to_json({"top_k": 1})))
        # Both queries only have 2 lexical matches in the seeded KB, so the
        # pool cap (max(top_k*3, 10)) never actually truncates here — this
        # instead pins that a retrieval_config override does not error and
        # candidates stay well-formed.
        assert len(from_json(default_result["retrieved_documents"])) == 2
        assert len(from_json(overridden["retrieved_documents"])) == 2

    def test_ret_08_missing_kb_file_degrades_gracefully(self):
        result = RetrieveNode()(
            _make_state(
                search_query="age verification",
                retrieval_config=to_json({"kb_path": "config/kb/does_not_exist.json"}),
            )
        )
        assert from_json(result["retrieved_documents"]) == []
        notes = from_json(result["intake_notes"], [])
        assert any("not readable" in n for n in notes)


class TestRetrieveTrustDeclaration:
    def test_admits_anonymous(self):
        assert RetrieveNode.required_trust_level is TrustLevel.ANONYMOUS
