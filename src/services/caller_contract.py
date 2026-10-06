"""AgentCore Platform v1.0"""

# RET-C2-668 - the caller-data contract.
#
# Every value a caller can supply passes through this module, from either
# channel:
#
#   input_context   the structured channel on the invoke call (validated by
#                   PreProcessNode, carried across the graph boundary by
#                   src/graph/context_bridge.py)
#   the question    a JSON envelope in the question text, kept for callers
#                   that have no structured channel (parsed by
#                   InputValidateNode)
#
# One module for both so a rule cannot be enforced on one channel and forgotten
# on the other.
#
# Rules applied to all of them:
#   - values must be the declared TYPE, `bool` explicitly excluded. A number,
#     boolean, mapping or list where a string is required is refused outright
#     and never coerced: str(float("nan")) is "nan", which satisfies a shape
#     check, so coercion is how a non-finite value gets a foothold.
#   - every number goes through _finite_in_range: NaN and +/-Infinity parse
#     fine through float() AND arrive intact in a raw JSON body, and every
#     comparison against NaN is False - so an unchecked non-finite threshold
#     silently disables the relevance floor this agent's grounding depends on.
#     Rejected, not clamped, so the failure is loud.
#   - strings that reach a filter or the rendered answer are locked to an inert
#     identifier alphabet; free text there is caller-controlled output.
#   - a refusal names the FIELD and never echoes the value; an unrecognised
#     field NAME is masked rather than echoed back.
#   - fail CLOSED: a breach stops the request, it does not fall back to a
#     default.

import math
from typing import Any, Dict, Optional

from src.services.security import contains_instruction_override

# Retrieval depth a caller may ask for.
TOP_K_MIN = 1
TOP_K_MAX = 20

# Relevance floor a caller may ask for. A caller may only make the floor
# STRICTER than the deployment's configured value (enforced by the consuming
# node) - loosening it would let a caller talk the agent out of its own
# grounding guarantee.
SCORE_THRESHOLD_MIN = 0.0
SCORE_THRESHOLD_MAX = 1.0

# Caller strings that select a knowledge-base category or name the calling
# channel are locked to an inert alphabet: they are matched against KB data and
# carried into audit events, so free text there is caller-controlled content in
# somebody else's field.
# Hyphen is in the alphabet alongside underscore: a channel label is naturally
# written `pos-terminal`, and refusing a whole night-shift question over a
# cosmetic separator is the fail-closed direction that actually blocks work.
# It stays inert - no whitespace, quotes, markup or control characters - and
# the seeded knowledge-base categories contain no hyphen, so category matching
# is unaffected.
_INERT_IDENTIFIER_MAX = 32
_INERT_IDENTIFIER_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789_-"

# The declared contract. Any other key in input_context is caller-controlled
# text, so its name is masked wherever a refusal has to mention it.
KNOWN_CONTEXT_FIELDS = frozenset({"category", "top_k", "score_threshold", "channel"})
MASKED_FIELD_NAME = "<unrecognised-field>"

# Structural limits on the caller payload itself. The screen walks EVERY value a
# caller sends, declared or not, so without these a deeply nested or very wide
# payload turns the screen into the cost. The contract has four fields; anything
# past this is not a real request.
MAX_CONTEXT_ENTRIES = 32
MAX_CONTEXT_DEPTH = 6


class CallerFieldError(ValueError):
    """A caller-supplied field failed its contract. Carries the field name only."""


def _finite_in_range(
    value: Any,
    field: str,
    minimum: float,
    maximum: float,
    integer: bool = False,
) -> float:
    """Parse a caller number, or raise naming the field (never the value).

    Rejects, in order: bool (isinstance(True, int) is True in Python, so a
    bare int check lets `true` through as 1), non-numeric types and strings,
    NaN / +Infinity / -Infinity, and out-of-range magnitudes.
    """
    if isinstance(value, bool):
        raise CallerFieldError(f"'{field}' must be a number")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except (TypeError, ValueError):
            raise CallerFieldError(f"'{field}' must be a number") from None
    else:
        raise CallerFieldError(f"'{field}' must be a number")

    if not math.isfinite(number):
        raise CallerFieldError(f"'{field}' must be a finite number")
    if number < minimum or number > maximum:
        raise CallerFieldError(f"'{field}' must be between {minimum} and {maximum}")
    if integer:
        if number != int(number):
            raise CallerFieldError(f"'{field}' must be a whole number")
        return float(int(number))
    return number


def require_top_k(value: Any, field: str = "top_k") -> int:
    """Validated retrieval depth."""
    return int(_finite_in_range(value, field, TOP_K_MIN, TOP_K_MAX, integer=True))


def require_score_threshold(value: Any, field: str = "score_threshold") -> float:
    """Validated relevance floor."""
    return _finite_in_range(value, field, SCORE_THRESHOLD_MIN, SCORE_THRESHOLD_MAX)


def require_inert_identifier(value: Any, field: str) -> str:
    """Validated inert identifier: lowercase ASCII, digits, underscore, <= 32.

    Case is normalised before the check so a caller may write `Age_Verification`;
    everything outside the alphabet is a refusal rather than a strip, because a
    strip would silently turn one identifier into a different valid one.
    """
    if isinstance(value, bool) or not isinstance(value, str):
        raise CallerFieldError(f"'{field}' must be a string")
    text: str = value.strip().lower()
    if not text:
        raise CallerFieldError(f"'{field}' must not be empty")
    if len(text) > _INERT_IDENTIFIER_MAX:
        raise CallerFieldError(f"'{field}' exceeds the {_INERT_IDENTIFIER_MAX}-character limit")
    if any(character not in _INERT_IDENTIFIER_ALPHABET for character in text):
        raise CallerFieldError(f"'{field}' must use lowercase letters, digits, underscores and hyphens only")
    return text


def safe_field_name(key: Any) -> str:
    """The name to use when reporting on a caller key - masked unless declared."""
    if isinstance(key, str) and key in KNOWN_CONTEXT_FIELDS:
        return key
    return MASKED_FIELD_NAME


def find_instruction_override(value: Any, path: str = "input_context", depth: int = 0) -> Optional[str]:
    """Depth-first scan of every decoded string in the payload - KEYS included.

    Runs on the PARSED structure, so \\u-escapes in the request body cannot
    smuggle a directive past it, and it covers undeclared keys too: the screen
    must hold on what the caller SENT, not only on what the contract keeps.
    Every path component outside the declared contract is masked, so the
    returned path is always safe to put in an error message.

    The structural limits are enforced HERE rather than only in field
    validation, because this walk runs first and over the whole payload. Past
    the depth or entry cap the payload is a violation in its own right - a
    caller who sends something that shape is not making a request.
    """
    if depth > MAX_CONTEXT_DEPTH:
        return f"{path} (nested deeper than {MAX_CONTEXT_DEPTH} levels)"
    if isinstance(value, str):
        return path if contains_instruction_override(value) else None
    if isinstance(value, dict):
        if len(value) > MAX_CONTEXT_ENTRIES:
            return f"{path} (more than {MAX_CONTEXT_ENTRIES} entries)"
        for key, item in value.items():
            key_path = f"{path}.{safe_field_name(key)}"
            if isinstance(key, str) and contains_instruction_override(key):
                return key_path
            found = find_instruction_override(item, key_path, depth + 1)
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_CONTEXT_ENTRIES:
            return f"{path} (more than {MAX_CONTEXT_ENTRIES} entries)"
        for index, item in enumerate(value):
            found = find_instruction_override(item, f"{path}[{index}]", depth + 1)
            if found is not None:
                return found
        return None
    return None


def validate_caller_context(input_context: Any) -> Dict[str, Any]:
    """Validate the structured caller channel. Raises CallerFieldError on breach.

    Absent or empty context is not an error - the agent answers from the
    question alone. A field that is PRESENT and invalid is always a refusal.
    """
    if input_context is None or input_context == {}:
        return {}
    if not isinstance(input_context, dict):
        raise CallerFieldError("'input_context' must be a mapping")

    fields: Dict[str, Any] = {}
    if input_context.get("category") is not None:
        fields["category"] = require_inert_identifier(input_context["category"], "category")
    if input_context.get("top_k") is not None:
        fields["top_k"] = require_top_k(input_context["top_k"])
    if input_context.get("score_threshold") is not None:
        fields["score_threshold"] = require_score_threshold(input_context["score_threshold"])
    if input_context.get("channel") is not None:
        fields["channel"] = require_inert_identifier(input_context["channel"], "channel")
    return fields
