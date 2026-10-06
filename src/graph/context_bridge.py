"""AgentCore Platform v1.0"""

# RET-C2-668 - caller-context bridge across the nested-graph boundary.
#
# Why this exists: the framework invokes an inner graph as
# `subgraph.invoke(user_input, session_id=..., ctx=...)` and does NOT forward
# the outer state's input_context. An inner node reading
# state["input_context"] therefore always sees {} - the structured caller
# channel simply does not cross the boundary. The two sanctioned subclass hooks
# bridge it:
#
#   NightShiftOpsGraphNode.extract_input(state)   [runs BEFORE subgraph.invoke]
#       -> set_caller_context(state["caller_context"])
#   DomainWorkflowGraph._extra_initial_state()    [runs INSIDE subgraph.invoke]
#       -> returns {"caller_context": get_caller_context()}
#
# What crosses is the VALIDATED contract produced by PreProcessNode, never the
# raw request body, so the inner graph is only ever handed values that already
# passed their type, range and alphabet bounds.
#
# Smuggling the values inside the question string is not a usable alternative:
# that field is masked at every node boundary, so its contents can be rewritten
# between hops. This channel is not masked - which is exactly why PreProcessNode
# screens it before anything enters here.
#
# A ContextVar keeps the hand-off correct per thread and per task, so concurrent
# invocations in one process cannot see each other's context.

from contextvars import ContextVar
from typing import Any, Dict, Optional

_CALLER_CONTEXT: ContextVar[Optional[Dict[str, Any]]] = ContextVar("ret_c2_668_caller_context", default=None)


def set_caller_context(caller_context: Optional[Dict[str, Any]]) -> None:
    """Stash the validated caller contract for the imminent inner-graph invoke."""
    _CALLER_CONTEXT.set(dict(caller_context) if caller_context else {})


def get_caller_context() -> Dict[str, Any]:
    """Read (without consuming) the stashed contract; {} when none was set."""
    return _CALLER_CONTEXT.get() or {}
