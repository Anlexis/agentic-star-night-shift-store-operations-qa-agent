# RET-C2-668 — Unit Tests: PostProcessNode (outer post_process slot; output gate)
#
# Invocation canon: node(state) via BaseNode.__call__. PostProcessNode is the
# second outer S-gate slot and requires VERIFIED_EXTERNAL (like PreProcessNode);
# the ANONYMOUS rejection lives in test_trust_gate.py.
#
# Output-gate layering: the node's own module-level security_gate_output() scan runs
# INSIDE execute() and replaces a violating answer with the sanitised stub
# (returned dict — no exception). Unlike a gate that scans only a
# which scans only a top-level string), THIS gate recurses into nested
# dict/list/tuple structures, so a credential nested inside a returned
# A peer template / a peer template) — it is also reused by
# ConvenienceStoreNightShiftOpsAgent.get_output() to re-scan the whitelisted
# escalation/citations payload (src/graph/graph.py). Intentional-credential
# tests assert the raw secret never survives into formatted_output OR result.
#
# Mirrors docs/03_test_spec.md POST-01..POST-09.
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.post_process_node import PostProcessNode, security_gate_output

_CLEAN_ANSWER = (
    "# Night Shift Operations Answer\n\n"
    "[1] utility bill payment: accepts bills with a valid barcode before cutoff.\n"
)

# Credential-shaped strings assembled at runtime so no such literal sits in
# the repository (credential-scan hygiene).
_FAKE_API_KEY = "sk-" + ("A" * 24)
_FAKE_JWT = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12


def _make_state(result_text, **extra) -> dict:
    state = {
        "result": result_text,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPostProcessClean:
    def test_post_01_clean_output_passes_through(self):
        result = PostProcessNode()(_make_state(_CLEAN_ANSWER))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: State carries the plain string, never the enum.
        # `type(...) is str`, not isinstance: AgentStatus subclasses str, so
        # isinstance would pass for the enum member this guards against.
        assert type(result["status"]) is str  # noqa: E721
        assert result["formatted_output"] == _CLEAN_ANSWER

    def test_post_02_empty_result_is_non_fatal(self):
        result = PostProcessNode()(_make_state(""))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == ""


class TestPostProcessS3Gate:
    def _assert_blocked(self, result, secret):
        assert result["status"] == AgentStatus.ERROR.value
        assert any("output blocked" in str(e) for e in result["error_log"])
        # The raw secret must not survive into either surfaced field.
        assert secret not in str(result.get("formatted_output", ""))
        assert secret not in str(result.get("result", ""))
        assert "[OUTPUT BLOCKED by the output gate" in result["formatted_output"]
        # Containment: no output-bearing field survives a block.
        assert result["ops_answer"] == ""
        assert result["citations"] is None
        assert result["escalation"] is None

    def test_post_03_api_key_is_blocked(self):
        result = PostProcessNode()(_make_state(f"# Report\n\n<!-- debug api_key={_FAKE_API_KEY} -->\n"))
        self._assert_blocked(result, _FAKE_API_KEY)

    def test_post_04_credential_assignment_is_blocked(self):
        secret = "password=super_secret_value_123"
        result = PostProcessNode()(_make_state(f"# Report\n\ninternal note: {secret}\n"))
        self._assert_blocked(result, "super_secret_value_123")

    def test_post_05_jwt_is_blocked(self):
        result = PostProcessNode()(_make_state(f"# Report\n\nsession token {_FAKE_JWT}\n"))
        self._assert_blocked(result, _FAKE_JWT)

    def test_post_06_bearer_token_is_blocked(self):
        secret = "Bearer abcdefghijklmnopqrstuvwxyz0123456789"
        result = PostProcessNode()(_make_state(f"# Report\n\nauthorization: {secret}\n"))
        self._assert_blocked(result, secret)


class TestSecurityGateOutputRecursion:
    """The module-level helper is reused by graph.py's get_output() to scan
    the structured escalation/citations payload — it must recurse into dict
    values, dict KEYS, and list/tuple elements, and treat scalars as inert."""

    def test_post_07_detects_a_credential_nested_inside_a_dict_value(self):
        # "secret=..." matches ONLY the credential_assignment pattern (no
        # sk-/pk-/ak- prefix, so the api_key pattern does not also fire).
        payload = {"channel": "supervisor", "note": "contact secret=abcdefgh12345678"}
        assert security_gate_output(payload) == "credential_assignment"

    def test_post_08_detects_a_credential_inside_a_list_of_dicts(self):
        payload = [{"id": "kb-001", "title": "ok"}, {"id": "kb-002", "title": f"see {_FAKE_JWT}"}]
        assert security_gate_output(payload) == "jwt"

    def test_post_09_scalars_and_clean_structures_are_inert(self):
        assert security_gate_output(None) is None
        assert security_gate_output(42) is None
        assert security_gate_output(True) is None
        assert security_gate_output({"required": False, "category": None, "channel": None}) is None
        assert security_gate_output([{"ref": 1, "id": "kb-001", "title": "Age verification"}]) is None


class TestPostProcessTrustDeclaration:
    def test_requires_verified_external(self):
        assert PostProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL
