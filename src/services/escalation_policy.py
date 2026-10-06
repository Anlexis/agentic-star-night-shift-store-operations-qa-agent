"""AgentCore Platform v1.0"""

# RET-C2-668 - the escalation policy: fixed instruction text and routing.
#
# This module is the single source of truth for what the agent says when a
# safety, security, medical or disaster signal fires. It lives OUTSIDE the node
# that generates the answer on purpose: the output boundary re-checks the
# rendered answer against these same constants, and a check that read the
# generating node's own working state would only be able to confirm that the
# node agreed with itself.
#
# The whole safety property of this template rests on these strings being DATA.
# They are looked up by category, never assembled from knowledge-base content
# and never from the caller's question, so no phrasing of a question can change
# what the clerk is told to do in an emergency. Any future model-backed answer
# synthesis covers the ordinary-question branch only; this branch stays a
# lookup.

from typing import Dict, Final

SECURITY: Final = "emergency_security"
MEDICAL: Final = "emergency_medical"
DISASTER: Final = "emergency_disaster"
UNSPECIFIED: Final = "emergency_unspecified"

# Knowledge-base categories that always route to escalation, never to answer
# assembly.
ESCALATION_CATEGORIES: Final = frozenset({SECURITY, MEDICAL, DISASTER})

ESCALATION_MESSAGES: Final[Dict[str, str]] = {
    SECURITY: (
        "This sounds like a security threat (robbery or a threatening "
        "customer). Do not confront the person or try to resolve this "
        "yourself. Comply, move to safety, then call the police at 110 and "
        "notify your shift supervisor immediately."
    ),
    MEDICAL: (
        "This sounds like a medical emergency. Do not attempt medical "
        "treatment beyond basic first aid you are certified for. Call an "
        "ambulance at 119 immediately, then notify your shift supervisor."
    ),
    DISASTER: (
        "This sounds like a fire, earthquake, or other disaster. Prioritize "
        "customer and staff safety over the store or its inventory - follow "
        "the posted evacuation route. Call 119 if there is a fire, then "
        "notify your shift supervisor once you are safe."
    ),
    UNSPECIFIED: (
        "This may be a safety emergency. Do not attempt to resolve this "
        "yourself. Notify your shift supervisor immediately, and call the "
        "police (110) or an ambulance (119) if anyone is in danger."
    ),
}

ESCALATION_CHANNEL: Final[Dict[str, str]] = {
    SECURITY: "police_110",
    MEDICAL: "ambulance_119",
    DISASTER: "fire_119_and_evacuate",
    UNSPECIFIED: "supervisor",
}


# Opening of every ORDINARY grounded answer - and the only place the caller's
# own question is echoed back into the output. The escalation path never uses
# it, so the output boundary can tell the two paths apart by this string alone.
ORDINARY_ANSWER_LEAD: Final = "Based on the store operations manual"


def instruction_for(category: str) -> str:
    """The fixed instruction for a category, falling back to the general one."""
    return ESCALATION_MESSAGES.get(category, ESCALATION_MESSAGES[UNSPECIFIED])


def channel_for(category: str) -> str:
    """The routing channel for a category, falling back to the supervisor."""
    return ESCALATION_CHANNEL.get(category, ESCALATION_CHANNEL[UNSPECIFIED])
