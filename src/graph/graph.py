"""AgentCore Platform v1.0"""

# RET-C2-668 - Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Convenience Store Night Shift Operations Q&A Agent (Cat 2 RAG domain workflow).
#
# Architecture (Cat 2):
#
#   Outer backbone (fixed - identical to Cat 1, do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
#                                             |  (RETRY, up to max_retry)
#                                             -> pre_process
#
#   `main` slot is a GraphNode subclass (NightShiftOpsGraphNode) that
#   delegates the full night-shift ops domain workflow to DomainWorkflowGraph
#   (inner BaseGraph: input_validate -> retrieve -> rerank_filter ->
#   generate_answer -> output_format).
#
#   Domain complexity, INCLUDING the safety-escalation boundary, is
#   fully encapsulated inside the inner graph. The outer backbone is never
#   modified.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (multi-step topology)
#
# Class-name contract:
#   graph.py class:           ConvenienceStoreNightShiftOpsAgent (this file)
#   config/agent.yaml class:  "ConvenienceStoreNightShiftOpsAgent"  <- must match
#   src/api/server.py import: from src.graph.graph import ConvenienceStoreNightShiftOpsAgent
#
# Rules enforced:
#   - ConvenienceStoreNightShiftOpsAgent inherits AgentBaseGraph (L1 Base - direct inheritance)
#   - super().register_nodes() called first (fills initialize + finalize)
#   - NightShiftOpsGraphNode assigned to self._nodes["main"]
#   - _parent_config() forwards the retrieval/llm blocks from
#     config/config.yaml (never {})
#   - merge_output() returns only changed keys
#   - add_edges() NOT overridden on the outer graph
#   - No platform SDK imports
#   - get_output() EXTENDS super().get_output() - structured
#     `escalation` / `citations` keys surfaced on SUCCESS only, fail-closed
#     (re-scanned via the recursive security_gate_output, never the pre-gate
#     raw state[...])

from typing import Any, ClassVar, Dict, List

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.graph.base_graph import BaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import set_caller_context
from src.nodes.post_process_node import PostProcessNode, security_gate_output
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json
from src.services.runtime_config import llm_config, retrieval_config


class NightShiftOpsGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of the outer agent.

    Wraps DomainWorkflowGraph (inner Cat 2 BaseGraph RAG + escalation pipeline).
    Called by AgentBaseGraph backbone after pre_process and before post_process.

    Contracts:
      get_subgraph()    - instantiate DomainWorkflowGraph with the forwarded
                          declared runtime config (_parent_config())
      extract_input()   - pull validated_input (identifier-stripped) from outer state
      merge_output()    - map sub_result fields into outer state delta (changed keys only)
      error_strategy    - "propagate": re-raise inner errors as SubgraphError (fail-fast)
    """

    # "propagate": re-raise inner graph exceptions as SubgraphError (default - fail fast).
    # "handle": call on_subgraph_error() instead - use for graceful degradation.
    error_strategy: ClassVar[str] = "propagate"

    # False: HITL interrupts are handled inside the inner graph only.
    # True: surface inner HITL interrupt to the outer caller.
    propagate_hitl: ClassVar[bool] = False

    def _parent_config(self) -> Dict[str, Any]:
        """Forward the declared `retrieval` + `llm` blocks to the inner graph.

        Both blocks are read from config/config.yaml (the runtime file), never
        from config/agent.yaml - the manifest carries identity and the
        compile-time gates only, so a reader pointed at it would find no such
        key, silently fall back to its own default, and leave every declared
        value dead. Returned under config["configurable"], never empty.

        The inner graph republishes the `retrieval` block into inner state
        (DomainWorkflowGraph._extra_initial_state()) so RetrieveNode /
        RerankFilterNode read live top_k / score_threshold values. The `llm`
        block is forwarded verbatim for the documented answer-synthesis upgrade
        (unused today; the escalation branch never uses it, then or now).
        """
        return {"configurable": {"retrieval": retrieval_config(), "llm": llm_config()}}

    def get_subgraph(self) -> "BaseGraph":
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported lazily (inside the method) to avoid
        circular-import risk at module load time and to match the pattern in
        the bundled Cat 2 sample.

        The inner graph receives the declared runtime config via its BaseGraph
        ctor; its domain NODES still take no constructor arguments and read
        config exclusively via State seeding (execute(self, state) -> dict -
        no config parameter).
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def execute(self, state: AgentState) -> dict[str, Any]:
        """Skip the inner graph when the request was already found unacceptable.

        A request declined by pre_process has no validated input to act on, so
        running the inner graph would only produce a second, vaguer reason for
        the same rejection - and overwrite the specific one already settled.
        """
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        result: dict[str, Any] = super().execute(state)
        return result

    def extract_input(self, state: AgentState) -> str:
        """Return the string input passed into inner_graph.invoke().

        PreProcessNode validates and identifier-strips the raw question and
        writes the result to validated_input. Prefer that; fall back to
        user_input if validated_input is absent (e.g. in a bare unit test).

        This is also the last point that still sees the outer state before the
        framework invokes the subgraph WITHOUT forwarding input_context, so the
        validated caller contract is stashed on the bridge here. Callers with no
        structured channel may still pass the same parameters as a JSON envelope
        inside the question text; InputValidateNode parses that form back, and
        the structured channel wins when both are present.
        """
        set_caller_context(from_json(state.get("caller_context"), {}))
        text: str = state.get("validated_input") or state.get("user_input", "") or ""
        return text

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map inner graph sub_result back into the outer state delta.

        sub_result is the dict returned by DomainWorkflowGraph.get_output().
        Returns ONLY changed keys - never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          Inner get_output() emits  -> "formatted_answer", "citations",
                                       "escalation", "status", ...
          This merge_output() reads -> sub_result.get("formatted_answer"),
                                       sub_result.get("citations"),
                                       sub_result.get("escalation"),
                                       sub_result.get("status")

        ops_answer (str | None): final rendered ops answer; written by
          OutputFormatNode inside the inner graph.
        result: PostProcessNode (outer post_process slot) reads
          state.get("result") - the inner graph emits the rendered answer
          under "formatted_answer", so map it to "result" as well; otherwise
          the final output surfaced by PostProcessNode (and the output gate) is
          always empty.
        citations / escalation: carried into the outer state so
          get_output() (below) can surface them as structured keys
          without reaching back into the inner graph.
        status (str | None): terminal AgentStatus value from the inner graph run.
        """
        return {
            # Outer reason wins: a reason settled before the inner run is the real
            # one, and a plain sub_result.get() would erase it.
            "error_code": state.get("error_code") or sub_result.get("error_code", ""),
            "ops_answer": sub_result.get("formatted_answer"),
            "result": sub_result.get("formatted_answer"),
            "citations": sub_result.get("citations"),
            "escalation": sub_result.get("escalation"),
            "status": sub_result.get("status"),
        }


class ConvenienceStoreNightShiftOpsAgent(AgentBaseGraph):
    """Outer graph for RET-C2-668 (Cat 2 RAG + safety-escalation boundary).

    Inherits AgentBaseGraph directly (L1 Base). Domain logic — including the
    safety-escalation decision — is fully encapsulated in
    NightShiftOpsGraphNode (main slot), which delegates to DomainWorkflowGraph
    (inner BaseGraph).

    Backbone (fixed - identical to Cat 1):
        START -> initialize -> pre_process -> main -> post_process -> finalize -> END

    register_nodes() is the ONLY override:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process:  PreProcessNode (caller contract + identifier strip)
      - main:         NightShiftOpsGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode (output gate)

    add_edges() is NOT overridden - backbone wiring belongs to the framework.

    get_output() EXTENDS the base envelope (never replaces it) to surface the
    structured `escalation` / `citations` keys this template's product
    requires (DOMAIN_BRIEF: "answer + cited passages + escalation routing
    fields") - SUCCESS only, fail-closed.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with AgentRegistry."""
        return "ConvenienceStoreNightShiftOpsAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first - it injects the
        framework's default InitializeNode (sets schema_version, session_id,
        trust_level) and FinalizeNode (builds response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = NightShiftOpsGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden - backbone wiring belongs to the framework.

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Extend the base envelope with whitelisted structured keys.

        `super().get_output(state)` returns the framework base envelope
        {output, status, trace_id, correlation_id, node_history} - kept
        AS-IS (never replaced; callers + audit rely on it).

        On top of that, this override surfaces two additional, EXPLICITLY
        whitelisted, SCALAR-only structures - never a raw request or
        response mapping verbatim:
          - escalation: {required: bool, category: str|None, channel: str|None,
                         reason: str|None}
          - citations:  [{ref: int|None, id: str, title: str, source: str}, ...]

        Fail-closed: structured fields are surfaced ONLY when
        status == AgentStatus.SUCCESS.value (PostProcessNode's gates on
        `result` already ran by this point, and a block sets status to
        ERROR), AND only after this override independently RE-SCANS the
        whitelisted payload with the same recursive `security_gate_output()`
        PostProcessNode uses. Either check failing withholds the structured
        fields entirely - the base envelope is still returned, never the
        pre-gate raw state[...].
        """
        base: Dict[str, Any] = super().get_output(state)
        # A run that completed WITHOUT carrying out the request holds the
        # sentence saying what to correct, not a product: none of the
        # structured fields below were produced, so none is released.
        if state.get("error_code"):
            return base

        if base.get("status") != AgentStatus.SUCCESS.value:
            return base

        escalation_raw: Dict[str, Any] = from_json(state.get("escalation"), {}) or {}
        citations_raw: List[Any] = from_json(state.get("citations"), []) or []

        escalation_clean: Dict[str, Any] = {
            "required": bool(escalation_raw.get("required", False)),
            "category": escalation_raw.get("category") if isinstance(escalation_raw.get("category"), str) else None,
            "channel": escalation_raw.get("channel") if isinstance(escalation_raw.get("channel"), str) else None,
            "reason": escalation_raw.get("reason") if isinstance(escalation_raw.get("reason"), str) else None,
        }
        citations_clean: List[Dict[str, Any]] = [
            {
                "ref": c.get("ref") if isinstance(c.get("ref"), int) else None,
                "id": str(c.get("id", "")),
                "title": str(c.get("title", "")),
                "source": str(c.get("source", "")),
            }
            for c in citations_raw
            if isinstance(c, dict)
        ]

        # Fail-closed: re-scan the whitelisted payload before exposing it.
        if security_gate_output(escalation_clean) or security_gate_output(citations_clean):
            return base

        base["escalation"] = escalation_clean
        base["citations"] = citations_clean
        return base


# Back-compat alias - config/agent.yaml declares class: "ConvenienceStoreNightShiftOpsAgent",
# and src/api/server.py imports the class directly. Keep both names pointing at the agent.
Graph = ConvenienceStoreNightShiftOpsAgent
