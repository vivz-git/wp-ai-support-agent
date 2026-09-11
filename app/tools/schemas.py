"""Pydantic schemas for the ``product_lookup`` tool.

These models define the tool's public contract: what a caller (eventually an
LLM via Groq's native function calling) may send as arguments, and exactly
what shape the handler returns. They are intentionally decoupled from the
catalog models in ``app.knowledge`` — the tool layer only reuses the shared
enums (``ProductCategory``, ``RoastLevel``, ``BrewMethod``) so the tool
contract can evolve independently of the on-disk catalog schema.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.knowledge import BrewMethod, ProductCategory, RoastLevel, StockLevel

# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------


class ProductAttribute(str, Enum):
    """Filterable product attributes.

    Mirrors ``app.knowledge.ProductAttribute`` plus ``wholesale_available``,
    which is a top-level boolean field on ``Product`` rather than an entry in
    its ``attributes`` list. The matching layer (``catalog_tool.py``) handles
    that distinction; callers of this tool see one flat attribute vocabulary.
    """

    ORGANIC = "organic"
    FAIRTRADE = "fairtrade"
    SINGLE_ESTATE = "single_estate"
    LOW_ACID = "low_acid"
    HIGH_CAFFEINE = "high_caffeine"
    DECAF = "decaf"
    GIFT_READY = "gift_ready"
    WHOLESALE_AVAILABLE = "wholesale_available"


class ProductLookupStatus(str, Enum):
    OK = "ok"
    NO_MATCH = "no_match"
    INVALID_INPUT = "invalid_input"
    UNAVAILABLE = "unavailable"


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class ProductLookupInput(BaseModel):
    """Validated arguments for the ``product_lookup`` tool.

    All fields are optional individually, but at least one field that can
    meaningfully constrain the search (``query``, ``category``,
    ``roast_level``, ``brew_method``, ``attributes``, or ``max_price_inr``)
    must be provided and non-empty — an all-defaults call (e.g. only
    ``in_stock_only``/``limit``) is rejected as an empty filter set.
    """

    model_config = ConfigDict(extra="forbid")

    query: Optional[str] = Field(None, max_length=120)
    category: Optional[ProductCategory] = None
    roast_level: Optional[RoastLevel] = None
    brew_method: Optional[BrewMethod] = None
    attributes: Optional[List[ProductAttribute]] = Field(None, max_length=5)
    max_price_inr: Optional[float] = Field(None, ge=0, le=100000)
    in_stock_only: bool = True
    limit: int = Field(3, ge=1, le=5)

    @field_validator("query")
    @classmethod
    def _normalize_query_whitespace(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("attributes")
    @classmethod
    def _dedupe_attributes(cls, value: Optional[List[ProductAttribute]]) -> Optional[List[ProductAttribute]]:
        if value is None:
            return None
        # Preserve first-seen order while removing duplicates so a caller
        # sending the same attribute twice can't be used to smuggle in a
        # 6th distinct entry under the max_length=5 check.
        deduped: List[ProductAttribute] = []
        for attr in value:
            if attr not in deduped:
                deduped.append(attr)
        return deduped or None

    @model_validator(mode="after")
    def _reject_empty_filter_set(self) -> "ProductLookupInput":
        has_constraint = any(
            [
                self.query is not None,
                self.category is not None,
                self.roast_level is not None,
                self.brew_method is not None,
                bool(self.attributes),
                self.max_price_inr is not None,
            ]
        )
        if not has_constraint:
            raise ValueError(
                "At least one of query, category, roast_level, brew_method, "
                "attributes, or max_price_inr must be provided"
            )
        return self

    @model_validator(mode="after")
    def _enforce_server_side_limit(self) -> "ProductLookupInput":
        # Belt-and-braces: Field(le=5) already rejects a too-large limit, but
        # a future relaxation of that constraint must not silently widen the
        # server-side cap without a deliberate change here too.
        if self.limit > 5:
            object.__setattr__(self, "limit", 5)
        return self


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class ProductResult(BaseModel):
    """A single catalog product as returned by the tool.

    A trimmed, tool-facing view of ``app.knowledge.Product`` — only fields a
    customer-facing answer would need.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sku: str
    name: str
    category: ProductCategory
    roast_level: Optional[RoastLevel] = None
    brew_methods: List[BrewMethod] = Field(default_factory=list)
    price_inr: float
    size_g: Optional[int] = None
    in_stock: bool
    stock_level: StockLevel
    attributes: List[str] = Field(default_factory=list)
    subscription_available: bool
    wholesale_available: bool
    product_url: str


class ProductLookupErrorInfo(BaseModel):
    """Structured, safe-to-display error detail.

    Never contains raw exception text or tracebacks — only a stable code and
    a short, generic, customer-safe message.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    message: str
    fields: List[str] = Field(default_factory=list)


class ProductLookupOutput(BaseModel):
    """The complete, validated result of a ``product_lookup`` call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: ProductLookupStatus
    catalog_version: Optional[str] = None
    normalized_query: Optional[str] = None
    result_count: int = 0
    results: List[ProductResult] = Field(default_factory=list)
    suggestions: List[ProductResult] = Field(default_factory=list)
    error: Optional[ProductLookupErrorInfo] = None
