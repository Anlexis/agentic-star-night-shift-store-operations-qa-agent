# RET-C2-668 — Unit Tests: DomainWorkflowGraph (inner BaseGraph)
#
# Inner-graph composition + a full inner invoke() over the seeded KB. The
# inner graph runs the 5 domain nodes (all ANONYMOUS) — the outer
# boundary is the AgentBaseGraph backbone's concern and is covered in
# test_graph_composition.py / the PoB suite.
#
# Values below (node_history, formatted_answer content, escalation dict) were
# captured by running the actual compiled graph against the real seeded KB —
# not hand-computed.
#
# Mirrors docs/03_test_spec.md DWG-01..DWG-06.
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from langgraph.graph import END

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus

from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import NightShiftOpsGraphNode
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, from_json

_AGE_QUERY = "what age verification do i need before selling alcohol to a customer"
_FIRE_QUERY = "there's a fire near the back storage room"
_NO_COVERAGE_QUERY = "quantum telepathy sandwich recipes"

_INNER_NODE_ORDER = [
    "InputValidateNode",
    "RetrieveNode",
    "RerankFilterNode",
    "GenerateAnswerNode",
    "OutputFormatNode",
]


class TestInnerGraphConstruction:
    def test_dwg_01_inherits_base_graph(self):
        assert issubclass(DomainWorkflowGraph, BaseGraph)

    def test_dwg_01_registers_the_five_domain_nodes(self):
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert set(inner._nodes.keys()) == {
            "input_validate",
            "retrieve",
            "rerank_filter",
            "generate_answer",
            "output_format",
        }
        assert isinstance(inner._nodes["input_validate"], InputValidateNode)
        assert isinstance(inner._nodes["retrieve"], RetrieveNode)
        assert isinstance(inner._nodes["rerank_filter"], RerankFilterNode)
        assert isinstance(inner._nodes["generate_answer"], GenerateAnswerNode)
        assert isinstance(inner._nodes["output_format"], OutputFormatNode)

    def test_inner_graph_name_and_schema(self):
        inner = DomainWorkflowGraph()
        assert inner.name == "ret_c2_668_night_shift_ops_workflow"
        assert inner.state_schema is State

    def test_initialize_finalize_are_not_registered(self):
        # Outer backbone concerns must not leak into the inner topology.
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert "initialize" not in inner._nodes
        assert "finalize" not in inner._nodes


class TestConfigForwarding:
    def test_dwg_02_extra_initial_state_republishes_retrieval_block(self):
        inner = DomainWorkflowGraph(config={"configurable": {"retrieval": {"top_k": 2}}})
        extra = inner._extra_initial_state()
        assert set(extra.keys()) == {"retrieval_config", "caller_context"}
        # Structured values travel as JSON strings, never bare containers.
        assert isinstance(extra["retrieval_config"], str)
        assert from_json(extra["retrieval_config"]) == {"top_k": 2}

    def test_extra_initial_state_seeds_the_caller_context_off_the_bridge(self):
        # The framework does not forward input_context into a subgraph, so the
        # outer node stashes the validated contract and this hook picks it up.
        from src.graph.context_bridge import set_caller_context

        set_caller_context({"category": "food_safety", "top_k": 3})
        try:
            extra = DomainWorkflowGraph()._extra_initial_state()
            assert from_json(extra["caller_context"]) == {"category": "food_safety", "top_k": 3}
        finally:
            set_caller_context({})

    def test_extra_initial_state_with_no_config_is_empty_block(self):
        assert from_json(DomainWorkflowGraph()._extra_initial_state()["retrieval_config"]) == {}


class TestRoutingAnnotation:
    """A path callable's annotation IS its input schema.

    The graph library projects away every field the annotation does not
    declare, so a routing callable annotated with the framework base state
    cannot see a flag a domain node wrote — the branch never runs in a real
    invocation while a unit test that calls the method with a full dict stays
    green. This topology wires no conditional edge today; the guard is here so
    that anyone who adds one inherits the correct annotation rather than the
    bug.
    """

    def test_route_is_annotated_with_this_graphs_own_state(self):
        import typing

        from src.schemas.state import State

        hints = typing.get_type_hints(DomainWorkflowGraph.route)
        assert hints["state"] is State, (
            "route() must be annotated with the graph's own State — the "
            "framework base state would project the domain fields away"
        )

    def test_no_conditional_edge_is_wired_without_a_checked_annotation(self):
        # If a future change wires one, this fails until the callable it names
        # is annotated with State.
        import inspect
        import typing

        from src.schemas.state import State

        source = inspect.getsource(DomainWorkflowGraph.add_edges)
        if "add_conditional_edges" in source:
            hints = typing.get_type_hints(DomainWorkflowGraph.route)
            assert hints["state"] is State


class TestOutputShape:
    def test_dwg_03_get_output_shapes_the_merge_contract(self):
        inner = DomainWorkflowGraph()
        out = inner.get_output(
            {
                "formatted_answer": "ANSWER",
                "citations": "[]",
                "escalation": "{}",
                "status": AgentStatus.SUCCESS.value,
                "intake_notes": None,
                "trace_id": "t1",
                "correlation_id": "c1",
                "node_history": ["InputValidateNode"],
            }
        )
        assert out["formatted_answer"] == "ANSWER"
        assert out["citations"] == "[]"
        assert out["escalation"] == "{}"
        assert out["status"] == AgentStatus.SUCCESS.value
        assert out["node_history"] == ["InputValidateNode"]

    def test_route_returns_end_on_error(self):
        inner = DomainWorkflowGraph()
        assert inner.route({"status": AgentStatus.ERROR.value}) == END
        assert inner.route({"status": AgentStatus.SUCCESS.value}) == "output_format"


class TestInnerEndToEnd:
    def _invoke(self, payload: str) -> dict:
        # Same construction path the outer GraphNode uses: manifest-derived
        # config via _parent_config(); domain nodes take NO ctor args.
        inner = DomainWorkflowGraph(config=NightShiftOpsGraphNode()._parent_config())
        return inner.invoke(payload, session_id="inner-e2e")

    def test_dwg_04_normal_query_produces_a_grounded_answer_with_citations(self):
        result = self._invoke(_AGE_QUERY)
        assert result["status"] == AgentStatus.SUCCESS.value
        answer = result["formatted_answer"]
        assert answer.startswith("# Night Shift Operations Answer")
        assert "[1]" in answer
        assert "does not replace your store's written protocols" in answer
        citations = from_json(result["citations"])
        assert citations[0]["id"] == "kb-002"
        escalation = from_json(result["escalation"])
        assert escalation["required"] is False

    def test_dwg_05_inner_node_history_is_the_linear_topology(self):
        assert self._invoke(_AGE_QUERY)["node_history"] == _INNER_NODE_ORDER

    def test_no_coverage_query_still_terminates_success(self):
        result = self._invoke(_NO_COVERAGE_QUERY)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "does not contain sufficient coverage" in result["formatted_answer"]

    def test_dwg_06_emergency_query_escalates_even_when_rerank_drops_every_candidate(self):
        """The KB does surface an emergency_disaster passage for this query
        (RetrieveNode), but its score (0.25) is below the 0.5 rerank
        threshold, so ranked_documents is empty by the time GenerateAnswerNode
        runs. The independent keyword safety net still fires -- proving a
        retrieval/rerank miss cannot suppress escalation end-to-end through
        the real compiled inner graph (not just at the node level)."""
        result = self._invoke(_FIRE_QUERY)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["node_history"] == _INNER_NODE_ORDER
        escalation = from_json(result["escalation"])
        assert escalation == {
            "required": True,
            "category": "emergency_disaster",
            "channel": "fire_119_and_evacuate",
            "reason": "query matched the emergency keyword safety net",
        }
        assert from_json(result["citations"]) == []
        assert "## Escalation Required" in result["formatted_answer"]
