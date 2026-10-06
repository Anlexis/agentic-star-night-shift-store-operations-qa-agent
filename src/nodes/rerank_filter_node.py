"""AgentCore Platform v1.0"""

# RET-C2-668 - RerankFilterNode
# Domain node 3: rerank the retrieval candidates and enforce the relevance
# floor. Deterministic: a small category-match boost on top of the retrieval
# score, drop everything below `score_threshold`, cap the survivors at
# `top_k`.
#
# Config: reads `top_k` / `score_threshold` from the state-seeded
# `retrieval_config` field (JSON), falling back to module defaults that mirror
# config/config.yaml. No `config` parameter on execute() - runtime tuning
# arrives via State only.
#
# A caller-supplied top_k or score_threshold wins ONLY when it is STRICTER than
# the deployment's configured value. That asymmetry is deliberate: a caller may
# ask for a narrower, better-grounded answer, but may not talk the agent into
# lowering the relevance floor that its grounding guarantee rests on.
#
# This node is intentionally domain-generic (a domain-agnostic algorithm
# golden reference) - it is NOT special-cased to exempt the three emergency
# categories from score_threshold. GenerateAnswerNode's independent keyword
# safety net (docs/02_design.md "THE SAFETY BOUNDARY") is the deliberate
# second layer that covers a retrieval/rerank miss on an emergency query.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.schemas.agent_status import AgentStatus
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json

# Defaults mirror the `retrieval` block in config/config.yaml.
_DEFAULT_RETRIEVAL: Dict[str, Any] = {
    "top_k": 5,
    "score_threshold": 0.5,
}

# Boost applied when a candidate's category matches the caller's filter.
_CATEGORY_BOOST = 0.1


def _resolve_retrieval_config(state: AgentState) -> Dict[str, Any]:
    """Effective retrieval config: state-seeded retrieval_config > defaults."""
    effective = dict(_DEFAULT_RETRIEVAL)  # local copy - never mutate the module default
    from_state = from_json(state.get("retrieval_config"), None)
    if isinstance(from_state, dict):
        effective.update(from_state)
    return effective


class RerankFilterNode(FunctionNode):
    """Rerank candidates, apply the score threshold, cap at top_k.

    Input state keys:
        retrieved_documents: JSON list of scored candidates (from RetrieveNode)
        query_filters:       JSON dict with optional category / top_k /
                             score_threshold override
        retrieval_config:    forwarded retrieval block (JSON)

    Output state keys (partial dict):
        ranked_documents: JSON list of surviving passages (score desc, <= top_k)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        candidates: List[Dict[str, Any]] = from_json(state.get("retrieved_documents"), []) or []
        filters = from_json(state.get("query_filters"), {}) or {}
        retrieval_cfg = _resolve_retrieval_config(state)

        try:
            top_k = int(retrieval_cfg.get("top_k", _DEFAULT_RETRIEVAL["top_k"]))
        except (TypeError, ValueError):
            top_k = int(_DEFAULT_RETRIEVAL["top_k"])
        top_k = max(1, min(20, top_k))
        # A stricter caller override (validated by InputValidateNode) wins.
        caller_top_k = filters.get("top_k")
        if isinstance(caller_top_k, int) and 1 <= caller_top_k < top_k:
            top_k = caller_top_k

        try:
            score_threshold = float(retrieval_cfg.get("score_threshold", _DEFAULT_RETRIEVAL["score_threshold"]))
        except (TypeError, ValueError):
            score_threshold = float(_DEFAULT_RETRIEVAL["score_threshold"])
        score_threshold = max(0.0, min(1.0, score_threshold))
        # A stricter caller floor wins; a looser one is ignored.
        caller_threshold = filters.get("score_threshold")
        if isinstance(caller_threshold, (int, float)) and not isinstance(caller_threshold, bool):
            if 0.0 <= float(caller_threshold) <= 1.0 and float(caller_threshold) > score_threshold:
                score_threshold = float(caller_threshold)

        category = filters.get("category")

        reranked: List[Dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            entry = dict(candidate)  # local copy - inputs stay immutable
            try:
                score = float(entry.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            if category and str(entry.get("category", "")).lower() == str(category).lower():
                score = min(1.0, score + _CATEGORY_BOOST)
            entry["score"] = round(score, 4)
            reranked.append(entry)

        # Deterministic ordering: score desc, then id asc for stable ties.
        reranked.sort(key=lambda c: (-c.get("score", 0.0), str(c.get("id", ""))))

        kept = [c for c in reranked if c.get("score", 0.0) >= score_threshold][:top_k]
        dropped = len(reranked) - len(kept)

        # Audit: rerank + relevance floor applied.
        emit_trace_event(
            "rerank_filter_complete",
            {
                "kept": len(kept),
                "dropped": dropped,
                "score_threshold": score_threshold,
                "top_k": top_k,
            },
            state,
        )

        return {"ranked_documents": to_json(kept)}
