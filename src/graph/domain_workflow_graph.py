"""AgentCore Platform v1.0"""

# RET-C2-668 - DomainWorkflowGraph (inner BaseGraph)
#
# This is the INNER graph for the Cat 2 two-layer nested architecture.
# It encapsulates the full night-shift ops Q&A domain workflow:
#
#   START -> input_validate -> retrieve -> rerank_filter
#         -> generate_answer -> output_format -> END
#
# Called by NightShiftOpsGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   - Inherits BaseGraph (fully custom topology - no forced backbone)
#   - Implements all 7 BaseGraph ABC methods
#   - register_nodes() does NOT call super() (abstract in BaseGraph)
#   - register_nodes() instantiates every domain node with NO ctor args
#   - Does NOT register initialize / finalize (outer backbone concerns)
#   - get_output() designed together with NightShiftOpsGraphNode.merge_output()
#   - No platform SDK imports
#   - Not placed under src/subagents/

from typing import Any, Dict

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import get_caller_context
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, to_json


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for RET-C2-668.

    Inherits BaseGraph directly for a fully custom node topology.
    Called by NightShiftOpsGraphNode.get_subgraph() in graph.py, which
    passes the declared runtime config (`_parent_config()`) into the ctor.

    Pipeline (linear):
        START
          -> input_validate  (InputValidateNode)  - parse + normalise the query
          -> retrieve        (RetrieveNode)       - keyword-score the seeded KB
          -> rerank_filter   (RerankFilterNode)   - boost / threshold / top_k cut
          -> generate_answer (GenerateAnswerNode) - grounded answer OR fixed
                                                     escalation instruction
                                                     (THE SAFETY BOUNDARY)
          -> output_format   (OutputFormatNode)   - final format + escalation
                                                     section + disclaimer
          -> END

    All nodes are FunctionNode subclasses returning partial-dict state updates.
    initialize / finalize are outer backbone concerns - not registered here.
    """

    # -- Identity --------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "ret_c2_668_night_shift_ops_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across inner and outer graph."""
        return State

    # -- Config validation -----------------------------------------------------

    def _validate_config(self) -> None:
        """Validate inner graph config before compilation.

        The forwarded `retrieval` block (top_k / score_threshold / kb_path) is
        read per-call by the domain nodes with safe defaults, so absence is
        non-fatal. Validation is permissive here rather than raising
        ConfigError.
        """
        pass

    # -- Config forwarding into state (config/config.yaml -> inner nodes) -------

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the inner state with the declared config AND the caller contract.

        `retrieval_config`: NightShiftOpsGraphNode._parent_config() forwards the
        `retrieval` + `llm` blocks from config/config.yaml under
        config["configurable"]; this hook makes the `retrieval` block reachable
        by the domain nodes as a JSON-string state field (structured values
        travel as JSON strings, never as bare containers). RetrieveNode and
        RerankFilterNode read it from there - node constructors take no
        arguments and execute() takes no config parameter, so state seeding is
        the only route in.

        `caller_context`: the validated caller contract, picked up off the
        bridge. The framework invokes this graph without forwarding
        input_context, so an inner node reading it directly would always see
        {}; the outer graph node stashes the validated values just before the
        invoke and this hook seeds them here.
        """
        retrieval = (self.config or {}).get("configurable", {}).get("retrieval") or {}
        return {
            "retrieval_config": to_json(retrieval),
            "caller_context": to_json(get_caller_context()),
        }

    # -- Node registration -----------------------------------------------------

    def register_nodes(self) -> None:
        """Register all 5 domain nodes.

        No super() call - BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.

        Every node is instantiated with NO constructor arguments - FunctionNode
        subclasses take no __init__; config flows in via State seeding only
        (execute(self, state) -> dict, no config parameter).
        Every key registered here is referenced in add_edges().
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # -- Edge wiring -----------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the linear night-shift ops domain topology.

        Each step passes its partial-dict output into the shared State.
        For this template the topology is intentionally linear - no
        conditional branching between domain nodes (the escalation decision
        is an internal branch INSIDE GenerateAnswerNode's execute(), not a
        graph-level conditional edge). route() is implemented as required by
        the ABC but add_conditional_edges() is not used.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # -- Routing ---------------------------------------------------------------

    def route(self, state: State) -> str:
        """Conditional routing - required by BaseGraph ABC.

        This topology is linear, so add_conditional_edges() is not used and
        this method is never called at runtime; it exists to satisfy the ABC.
        Returns END on error so an unexpected call cannot re-enter a processing
        node.

        The annotation is this graph's OWN State, not the framework base state,
        and that is load-bearing rather than cosmetic: the graph library reads a
        path callable's annotation as its input schema and PROJECTS AWAY every
        field the annotation does not declare. Annotated with the base state,
        a routing flag written by a domain node would be missing on every call,
        the branch would never be taken in a real run, and unit tests that call
        route() directly with a full dict would still pass. Anyone wiring a
        conditional edge here must keep this annotation.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # -- Output shape ----------------------------------------------------------

    def get_output(self, state: State) -> Dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by NightShiftOpsGraphNode.merge_output() in
        graph.py as the `sub_result` argument. Both methods are designed
        together to guarantee field-name consistency:

            Inner get_output()  emits: "formatted_answer", "citations",
                                       "escalation", "status", ...
            Outer merge_output() reads: sub_result.get("formatted_answer"),
                                        sub_result.get("citations"),
                                        sub_result.get("escalation"),
                                        sub_result.get("status")

        Additional fields (intake_notes, trace_id, correlation_id,
        node_history) are surfaced for observability / downstream extension.
        """
        return {
            # the reason must leave the subgraph or the outer graph cannot report it
            "error_code": state.get("error_code"),
            "formatted_answer": state.get("formatted_answer"),
            "citations": state.get("citations"),
            "escalation": state.get("escalation"),
            "status": state.get("status"),
            "intake_notes": state.get("intake_notes"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
