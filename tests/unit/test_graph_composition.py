# RET-C2-668 — Unit Tests: nested Cat-2 graph composition (outer + end-to-end)
#
# Drives the REAL outer agent (ConvenienceStoreNightShiftOpsAgent / Graph)
# end-to-end via AgentBaseGraph.invoke(). The e2e context is
# InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL) — the
# manifest's declared caller level; for_internal() is NEVER used (it would
# over-privilege the run and hide trust-gate regressions).
#
# Values below (node_history, get_output shapes) were captured by running the
# actual compiled outer graph — not hand-computed. In particular, the
# ANONYMOUS-denial node_history is NOT simply [Initialize, PreProcess]: the
# pre_process -> main edge is UNCONDITIONAL (AgentBaseGraph only routes
# conditionally AFTER "main"), so NightShiftOpsGraphNode's own __call__ still
# runs (its required_trust_level is the BaseNode default ANONYMOUS, which an
# ANONYMOUS caller satisfies) — but its input-gate-style status check sees the
# still-error state from pre_process and skips execute(), passing the error
# state through unchanged. route() then sees status=error and sends the run
# to finalize, skipping post_process. This test therefore asserts only the
# stable prefix + PostProcessNode's absence (matches the conventional form's
# own assertion shape), not the full history.
#
# Mirrors docs/03_test_spec.md GRP-01..GRP-10.
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import pathlib

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import (
    ConvenienceStoreNightShiftOpsAgent,
    Graph,
    NightShiftOpsGraphNode,
)
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json, to_json

_AGE_QUERY = "What age verification do I need before selling alcohol to a customer?"
_FIRE_QUERY = "there's a fire near the back storage room"
_NO_COVERAGE_QUERY = "quantum telepathy sandwich recipes"


def _run(user_input: str, trust: TrustLevel = TrustLevel.VERIFIED_EXTERNAL) -> dict:
    ctx = InvocationContext(caller_trust_level=trust, caller_id="unit-suite")
    return Graph().invoke(user_input, ctx=ctx)


class TestOuterGraphConstruction:
    def test_grp_01_inherits_agent_base_graph_directly(self):
        assert issubclass(ConvenienceStoreNightShiftOpsAgent, AgentBaseGraph)

    def test_grp_01_graph_alias(self):
        assert Graph is ConvenienceStoreNightShiftOpsAgent

    def test_state_schema_is_state(self):
        assert ConvenienceStoreNightShiftOpsAgent().state_schema is State

    def test_grp_02_compile_fills_all_backbone_slots(self):
        agent = ConvenienceStoreNightShiftOpsAgent()
        agent.compile()
        for slot in ("initialize", "pre_process", "main", "post_process", "finalize"):
            assert agent._nodes.get(slot) is not None, f"backbone slot not filled: {slot}"
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], NightShiftOpsGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_add_edges_is_not_overridden(self):
        # Backbone wiring belongs to the framework — the template must not
        # redefine it.
        assert "add_edges" not in ConvenienceStoreNightShiftOpsAgent.__dict__


class TestMainSlotGraphNode:
    def test_grp_03_get_subgraph_returns_the_inner_graph(self):
        subgraph = NightShiftOpsGraphNode().get_subgraph()
        assert isinstance(subgraph, DomainWorkflowGraph)
        assert subgraph.config["configurable"]["retrieval"], "inner config must carry the retrieval block"

    def test_grp_04_extract_input_prefers_validated_input(self):
        node = NightShiftOpsGraphNode()
        assert node.extract_input({"validated_input": "VI", "user_input": "UI"}) == "VI"
        assert node.extract_input({"user_input": "UI"}) == "UI"

    def test_grp_05_merge_output_maps_the_inner_contract(self):
        node = NightShiftOpsGraphNode()
        citations = to_json([{"ref": 1, "id": "kb-002", "title": "t", "source": "s"}])
        escalation = to_json({"required": False, "category": None, "channel": None, "reason": None})
        delta = node.merge_output(
            {},
            {
                "formatted_answer": "ANSWER",
                "citations": citations,
                "escalation": escalation,
                "status": AgentStatus.SUCCESS.value,
            },
        )
        # The inner formatted_answer surfaces as BOTH ops_answer and result
        # (PostProcessNode's output gate reads state["result"]).
        assert delta == {
            # The boundary now also carries the degraded-completion marker; on
            # an answered run neither side set one, so it crosses empty.
            "error_code": "",
            "ops_answer": "ANSWER",
            "result": "ANSWER",
            "citations": citations,
            "escalation": escalation,
            "status": AgentStatus.SUCCESS.value,
        }

    def test_error_strategy_is_propagate_and_hitl_is_contained(self):
        assert NightShiftOpsGraphNode.error_strategy == "propagate"
        assert NightShiftOpsGraphNode.propagate_hitl is False

    def test_main_slot_node_itself_admits_anonymous(self):
        # NightShiftOpsGraphNode does not override required_trust_level — the
        # external trust gate lives on the backbone pre_process slot only.
        assert NightShiftOpsGraphNode.required_trust_level is TrustLevel.ANONYMOUS

    def test_grp_06_parent_config_never_empty_without_the_config_file(self, monkeypatch):
        # Even with an unreadable config file the forwarded blocks carry the
        # fallback values — never {}, which would leave the inner nodes with
        # nothing seeded and no way to tell that from a deliberate empty block.
        import src.services.runtime_config as runtime_config

        monkeypatch.setattr(runtime_config, "_CONFIG_PATH", pathlib.Path("/nonexistent/config.yaml"))
        cfg = NightShiftOpsGraphNode()._parent_config()
        assert cfg["configurable"]["retrieval"]["kb_path"] == "config/kb/cvs_night_ops_kb.json"
        assert cfg["configurable"]["llm"]


class TestEndToEndInvoke:
    """Full agent run: outer backbone + inner domain workflow, no LLM."""

    def test_grp_07_invoke_returns_success(self):
        result = _run(_AGE_QUERY)
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"Expected success, got {result.get('status')}. result={result!r}"

    def test_grp_07_output_is_the_gated_formatted_answer(self):
        output = _run(_AGE_QUERY).get("output")
        assert isinstance(output, str) and output.strip()
        assert output.startswith("# Night Shift Operations Answer")
        assert "[1]" in output
        assert "does not replace your store's written protocols" in output

    def test_grp_08_e2e_traverses_the_post_process_gate(self):
        history = _run(_AGE_QUERY).get("node_history", [])
        assert history == [
            "InitializeNode",
            "PreProcessNode",
            "NightShiftOpsGraphNode",
            "PostProcessNode",
            "FinalizeNode",
        ]

    def test_no_coverage_query_still_terminates_success(self):
        result = _run(_NO_COVERAGE_QUERY)
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert "does not contain sufficient coverage" in result.get("output", "")

    def test_grp_09_anonymous_caller_is_denied_at_the_outer_boundary(self):
        """Trust gate at graph level: an ANONYMOUS invoke is refused by the
        VERIFIED_EXTERNAL pre_process slot. No domain answer is ever produced
        and post_process (the output gate) never runs."""
        result = _run(_AGE_QUERY, trust=TrustLevel.ANONYMOUS)
        assert result.get("status") == AgentStatus.ERROR.value
        assert not result.get("output")
        history = result.get("node_history", [])
        assert "PostProcessNode" not in history
        assert history[:2] == ["InitializeNode", "PreProcessNode"]

    def test_grp_10_structured_fields_withheld_on_a_non_success_status(self):
        # Fail-closed: get_output() only adds escalation/citations when
        # status == SUCCESS — an ANONYMOUS-denied run must not carry them.
        result = _run(_AGE_QUERY, trust=TrustLevel.ANONYMOUS)
        assert "escalation" not in result
        assert "citations" not in result


class TestEscalationSurfacedOnOuterEnvelope:
    """The outer get_output() override whitelists escalation/citations
    onto the base envelope, SUCCESS-only, re-scanned by the output gate."""

    def test_emergency_query_surfaces_escalation_true_through_the_full_backbone(self):
        result = _run(_FIRE_QUERY)
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("node_history") == [
            "InitializeNode",
            "PreProcessNode",
            "NightShiftOpsGraphNode",
            "PostProcessNode",
            "FinalizeNode",
        ]
        assert result.get("escalation") == {
            "required": True,
            "category": "emergency_disaster",
            "channel": "fire_119_and_evacuate",
            "reason": "query matched the emergency keyword safety net",
        }
        assert result.get("citations") == []
        assert "## Escalation Required" in result.get("output", "")

    def test_normal_query_surfaces_escalation_false_and_real_citations(self):
        result = _run(_AGE_QUERY)
        assert result.get("escalation", {}).get("required") is False
        citations = result.get("citations")
        assert citations and citations[0]["id"] == "kb-002"


class TestStateRoundTrip:
    """Serialization helpers: producers to_json() on write, consumers from_json()."""

    def test_to_from_json_list_round_trip(self):
        original = [{"id": "kb-002", "score": 0.58, "title": "age verification"}]
        assert from_json(to_json(original)) == original

    def test_to_from_json_dict_round_trip(self):
        original = {"category": "age_verification", "top_k": 3}
        assert from_json(to_json(original)) == original

    def test_to_json_none_passes_through(self):
        assert to_json(None) is None

    def test_from_json_malformed_returns_default(self):
        assert from_json("{not valid json", default=[]) == []
        assert from_json(None, default={}) == {}
        assert from_json("", default=[]) == []
