"""Deterministic, read-only tool layer for the AI WhatsApp Support Agent.

- No network calls. No database. No LLM calls.
- Pure Python matching against the validated clinic data from ``app.knowledge``.
- Tool handlers never raise; malformed input always yields a structured
  ``invalid_input`` result instead of an exception.
"""

from app.tools.clinic_tool import clinic_faq_lookup
from app.tools.registry import ToolRegistry, ToolSpec
from app.tools.schemas import ClinicFaqLookupInput

CLINIC_FAQ_LOOKUP_TOOL = ToolSpec(
    name="clinic_faq_lookup",
    description=(
        "Look up SmileCare Dental facts: service price ranges (consultation, cleaning, root canal/RCT, "
        "braces, whitening), timings, address and phone, dentists, and FAQs such as payment, insurance "
        "and booking. Accepts English, Hindi or Hinglish queries. Read-only; returns only real clinic data."
    ),
    input_model=ClinicFaqLookupInput,
    handler=clinic_faq_lookup,
)


def build_default_registry() -> ToolRegistry:
    """Build a fresh registry with every clinic tool registered."""
    registry = ToolRegistry()
    registry.register(CLINIC_FAQ_LOOKUP_TOOL)
    return registry


default_registry = build_default_registry()

__all__ = [
    "CLINIC_FAQ_LOOKUP_TOOL",
    "ToolRegistry",
    "ToolSpec",
    "build_default_registry",
    "default_registry",
]
