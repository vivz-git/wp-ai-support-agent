"""Deterministic, read-only tool layer for the AI WhatsApp Support Agent.

Design constraints (Milestone 2, Slice 2):
- No network calls. No database. No LLM calls.
- Pure Python matching against the validated catalog from ``app.knowledge``.
- Tool handlers never raise; malformed input always yields a structured
  ``invalid_input`` result instead of an exception.
- Independent of the running application: nothing here is wired into
  ``app.main`` or the Groq provider yet. That is future-slice work.
"""

from app.tools.catalog_tool import product_lookup
from app.tools.registry import ToolRegistry, ToolSpec
from app.tools.schemas import ProductLookupInput

PRODUCT_LOOKUP_TOOL = ToolSpec(
    name="product_lookup",
    description=(
        "Look up Kettle & Bloom Coffee Roasters products by query text, "
        "category, roast level, brew method, attributes, and/or max price. "
        "Read-only and deterministic; returns only real catalog data."
    ),
    input_model=ProductLookupInput,
    handler=product_lookup,
)


def build_default_registry() -> ToolRegistry:
    """Build a fresh registry with all Slice 2 tools registered."""
    registry = ToolRegistry()
    registry.register(PRODUCT_LOOKUP_TOOL)
    return registry


default_registry = build_default_registry()

__all__ = [
    "ToolRegistry",
    "ToolSpec",
    "PRODUCT_LOOKUP_TOOL",
    "build_default_registry",
    "default_registry",
]
