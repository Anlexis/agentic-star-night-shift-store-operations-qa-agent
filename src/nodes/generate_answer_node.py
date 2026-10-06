"""AgentCore Platform v1.0"""

# RET-C2-668 - GenerateAnswerNode
# Domain node 4: assemble the grounded answer from the ranked KB passages -
# OR, THE SAFETY BOUNDARY: for a safety,
# security, or medical/disaster emergency class, do NOT assemble or
# improvise an operational answer at all. Return a FIXED, non-improvised
# escalation instruction from a constant lookup table and set the
# structured `escalation` field.
#
# Detection uses TWO INDEPENDENT signals - either one alone is sufficient
# (fail-safe-leaning by design; never fail-open on a possible emergency):
#   1. KB-category signal: the top-ranked passage's category is one of the
#      three escalation categories seeded in config/kb/cvs_night_ops_kb.json
#      (emergency_security / emergency_medical / emergency_disaster).
#   2. Keyword safety net: an independent scan of the raw search_query
#      against a fixed EN + JP emergency-phrase list, so a retrieval /
#      rerank miss (the protocol entry did not clear score_threshold, or
#      ranked below the entries RerankFilterNode kept) cannot suppress
#      escalation.
#
# The node is otherwise DETERMINISTIC (no model call): on the ordinary path
# the answer is rule-assembled from the ranked passages only - a lead sentence
# plus one cited point per passage, each carrying a numbered citation marker
# [n]. The answer-synthesis upgrade seam covers the ORDINARY path only - the
# escalation branch stays a fixed lookup afterwards too - and is documented in
# docs/02_design.md ("Answer synthesis") and
# config/prompts/answer_synthesis_prompt.md.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List, Optional

from framework.schemas.agent_status import AgentStatus
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json
from src.services.escalation_policy import (
    DISASTER,
    ESCALATION_CATEGORIES,
    MEDICAL,
    ORDINARY_ANSWER_LEAD,
    SECURITY,
    channel_for,
    instruction_for,
)

# Independent keyword safety net. English phrases are matched against the
# lowercased query; Japanese phrases are matched against the raw query (Japanese
# has no letter case). Catches a genuine emergency even when retrieval does not
# rank a matching protocol passage into ranked_documents at all.
_EMERGENCY_KEYWORDS_EN: Dict[str, List[str]] = {
    SECURITY: [
        "robbery",
        "robber",
        "weapon",
        "gun",
        "knife",
        "holdup",
        "hold up",
        "threatening me",
        "someone is threatening",
    ],
    MEDICAL: [
        "collapsed",
        "unconscious",
        "not breathing",
        "chest pain",
        "seizure",
        "heavy bleeding",
        "bleeding heavily",
    ],
    DISASTER: [
        "there's a fire",
        "there is a fire",
        "smoke everywhere",
        "earthquake",
        "need to evacuate",
        "building is shaking",
    ],
}
_EMERGENCY_KEYWORDS_JP: Dict[str, List[str]] = {
    SECURITY: ["強盗", "刃物", "凶器", "脅され"],
    MEDICAL: ["急病", "倒れ", "意識がない", "呼吸してい"],
    DISASTER: ["災害", "火事", "地震", "避難"],
}


# Answer body used when no passage cleared the relevance threshold AND no
# emergency signal fired.
_NO_COVERAGE_ANSWER = (
    "The store operations manual does not contain sufficient coverage to "
    "answer this question. Rephrase the query with more specific terms, or "
    "ask your shift supervisor for a manual review."
)

# Cited excerpt length per passage inside the answer body.
_POINT_EXCERPT_CHARS = 240


def _first_sentences(text: str, limit: int) -> str:
    """Trim an excerpt at a sentence boundary where possible, else hard-cap."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    period = cut.rfind(". ")
    if period > limit // 2:
        return cut[: period + 1]
    return cut.rstrip() + "..."


def _keyword_emergency_category(query: str) -> Optional[str]:
    """Independent keyword safety net over the raw query text (EN + JP)."""
    lowered = query.lower()
    for category, phrases in _EMERGENCY_KEYWORDS_EN.items():
        for phrase in phrases:
            if phrase in lowered:
                return category
    for category, phrases in _EMERGENCY_KEYWORDS_JP.items():
        for phrase in phrases:
            if phrase in query:
                return category
    return None


class GenerateAnswerNode(FunctionNode):
    """Rule-based grounded answer assembly - OR fixed escalation routing.

    Input state keys:
        ranked_documents: JSON list of surviving passages (from RerankFilterNode)
        search_query:     normalised query (for the lead sentence / keyword net)

    Output state keys (partial dict):
        grounded_answer: answer body with [n] citation markers, OR a FIXED
                         escalation instruction (never both / never a blend)
        citations:       JSON list [{ref, id, title, source}]
        escalation:      JSON {"required": bool, "category": str|None,
                               "channel": str|None, "reason": str|None}
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        ranked: List[Dict[str, Any]] = from_json(state.get("ranked_documents"), []) or []
        query = state.get("search_query") or ""
        if not isinstance(query, str):
            query = ""

        top_category: Optional[str] = None
        if ranked and isinstance(ranked[0], dict):
            candidate_category = str(ranked[0].get("category", "")).strip()
            top_category = candidate_category or None

        keyword_category = _keyword_emergency_category(query)

        escalation_category: Optional[str] = None
        reason: Optional[str] = None
        if top_category in ESCALATION_CATEGORIES:
            escalation_category = top_category
            reason = "top-ranked passage is an emergency protocol entry"
        elif keyword_category is not None:
            escalation_category = keyword_category
            reason = "query matched the emergency keyword safety net"

        citations: List[Dict[str, Any]] = []
        for ref, doc in enumerate(ranked, start=1):
            if not isinstance(doc, dict):
                continue
            citations.append(
                {
                    "ref": ref,
                    "id": str(doc.get("id", "")),
                    "title": str(doc.get("title", "")).strip(),
                    "source": str(doc.get("source", "")),
                }
            )

        if escalation_category is not None:
            # THE SAFETY BOUNDARY: a fixed, looked-up instruction - never
            # assembled from knowledge-base content or from the question.
            # Citations (if any) still point the clerk at the full written
            # protocol.
            grounded_answer = instruction_for(escalation_category)
            escalation: Dict[str, Any] = {
                "required": True,
                "category": escalation_category,
                "channel": channel_for(escalation_category),
                "reason": reason,
            }
        elif not ranked:
            grounded_answer = _NO_COVERAGE_ANSWER
            citations = []
            escalation = {"required": False, "category": None, "channel": None, "reason": None}
        else:
            lines: List[str] = []
            if query:
                lines.append(f"{ORDINARY_ANSWER_LEAD}, the following " f'passages answer the question: "{query}"')
            else:
                lines.append(f"{ORDINARY_ANSWER_LEAD}, the most relevant passages are:")
            lines.append("")
            for citation, doc in zip(citations, ranked):
                if not isinstance(doc, dict):
                    continue
                excerpt = _first_sentences(str(doc.get("excerpt", "")), _POINT_EXCERPT_CHARS)
                lines.append(f"[{citation['ref']}] {citation['title']}: {excerpt}")
            grounded_answer = "\n".join(lines)
            escalation = {"required": False, "category": None, "channel": None, "reason": None}

        # Audit: an answer (or an escalation) was assembled.
        emit_trace_event(
            "generate_answer_complete",
            {
                "citation_count": len(citations),
                "answer_chars": len(grounded_answer),
                "no_coverage": not ranked and escalation_category is None,
                "escalation_required": escalation_category is not None,
            },
            state,
        )

        return {
            "grounded_answer": grounded_answer,
            "citations": to_json(citations),
            "escalation": to_json(escalation),
        }
