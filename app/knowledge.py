"""Business knowledge layer for the AI WhatsApp Support Agent.

This module loads and validates the two publishable data files that describe
the fictional business ("Kettle & Bloom Coffee Roasters") the agent supports:

- ``data/business.json``: hours, shipping/return policy, wholesale info, FAQs.
- ``data/catalog.json``: the product catalog (single origins, blends, decaf,
  equipment, subscriptions, accessories).

Design constraints (Milestone 2, Slice 1):
- No network calls. No LLM calls. No product-lookup tool. No agent logic.
- Explicit Pydantic models, not loose dictionaries.
- Fails loudly (raises ``KnowledgeError``) on a missing or malformed file so a
  bad data file is caught at load time, not silently ignored at runtime.
- Deterministic: loading the same files twice yields identical, immutable
  data and identical digest text.
"""

import json
import logging
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Default locations, relative to the project root (two levels up from this file).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUSINESS_PROFILE_PATH = _PROJECT_ROOT / "data" / "business.json"
DEFAULT_CATALOG_PATH = _PROJECT_ROOT / "data" / "catalog.json"


class KnowledgeError(Exception):
    """Raised when business/catalog knowledge data is missing or malformed.

    This is intentionally a single, clearly-named exception type so a caller
    (or a startup check) can fail loudly with one obvious cause: the on-disk
    knowledge data is not usable.
    """


# ---------------------------------------------------------------------------
# Catalog models
# ---------------------------------------------------------------------------


class ProductCategory(str, Enum):
    SINGLE_ORIGIN = "single_origin"
    BLEND = "blend"
    DECAF = "decaf"
    EQUIPMENT = "equipment"
    SUBSCRIPTION = "subscription"
    ACCESSORY = "accessory"


class RoastLevel(str, Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    MEDIUM_DARK = "medium_dark"
    DARK = "dark"


class BrewMethod(str, Enum):
    ESPRESSO = "espresso"
    FILTER = "filter"
    POUROVER = "pourover"
    FRENCHPRESS = "frenchpress"
    MOKAPOT = "mokapot"
    COLDBREW = "coldbrew"
    AEROPRESS = "aeropress"


class StockLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ProductAttribute(str, Enum):
    ORGANIC = "organic"
    FAIRTRADE = "fairtrade"
    SINGLE_ESTATE = "single_estate"
    LOW_ACID = "low_acid"
    HIGH_CAFFEINE = "high_caffeine"
    DECAF = "decaf"
    GIFT_READY = "gift_ready"

# Categories that describe a roasted coffee (as opposed to equipment,
# subscriptions, or accessories) and therefore must carry brew-relevant facts.
_ROASTED_COFFEE_CATEGORIES = {
    ProductCategory.SINGLE_ORIGIN,
    ProductCategory.BLEND,
    ProductCategory.DECAF,
}


class Product(BaseModel):
    """A single catalog SKU.

    Field set matches what the future ``product_lookup`` tool will need to
    return, so no schema change should be required when that tool is built
    in a later slice.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sku: str = Field(..., pattern=r"^KB-[A-Z0-9]+(-[A-Z0-9]+)*$")
    name: str = Field(..., min_length=1, max_length=120)
    category: ProductCategory
    origin: Optional[str] = Field(None, max_length=120)
    process: Optional[str] = Field(None, max_length=120)
    roast_level: Optional[RoastLevel] = None
    tasting_notes: List[str] = Field(default_factory=list)
    brew_methods: List[BrewMethod] = Field(default_factory=list)
    price_inr: float = Field(..., gt=0, le=100000)
    size_g: Optional[int] = Field(None, gt=0, le=20000)
    in_stock: bool
    stock_level: StockLevel
    attributes: List[ProductAttribute] = Field(default_factory=list)
    subscription_available: bool
    wholesale_available: bool
    dispatch_days: int = Field(..., ge=1, le=30)
    product_url: str = Field(..., max_length=500)

    @field_validator("tasting_notes")
    @classmethod
    def _validate_tasting_notes(cls, value: List[str]) -> List[str]:
        for note in value:
            if not note or not note.strip():
                raise ValueError("tasting_notes entries must be non-empty strings")
        return value

    @field_validator("product_url")
    @classmethod
    def _validate_product_url(cls, value: str) -> str:
        if not (value.startswith("https://") or value.startswith("http://")):
            raise ValueError("product_url must be an http(s) URL")
        return value

    @model_validator(mode="after")
    def _validate_stock_consistency(self) -> "Product":
        if not self.in_stock and self.stock_level != StockLevel.NONE:
            raise ValueError(
                f"{self.sku}: in_stock is False but stock_level is "
                f"'{self.stock_level.value}' (expected 'none')"
            )
        if self.in_stock and self.stock_level == StockLevel.NONE:
            raise ValueError(
                f"{self.sku}: in_stock is True but stock_level is 'none'"
            )
        return self

    @model_validator(mode="after")
    def _validate_roasted_coffee_fields(self) -> "Product":
        if self.category in _ROASTED_COFFEE_CATEGORIES:
            if not self.roast_level:
                raise ValueError(
                    f"{self.sku}: category '{self.category.value}' requires roast_level"
                )
            if not self.brew_methods:
                raise ValueError(
                    f"{self.sku}: category '{self.category.value}' requires at least one brew_method"
                )
        if self.category == ProductCategory.DECAF and ProductAttribute.DECAF not in self.attributes:
            raise ValueError(f"{self.sku}: category 'decaf' requires the 'decaf' attribute")
        return self


class Catalog(BaseModel):
    """The full validated product catalog."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    catalog_version: str = Field(..., min_length=1, max_length=40)
    products: List[Product] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_unique_skus(self) -> "Catalog":
        skus = [p.sku for p in self.products]
        duplicates = {sku for sku in skus if skus.count(sku) > 1}
        if duplicates:
            raise ValueError(f"Duplicate SKUs in catalog: {sorted(duplicates)}")
        return self


# ---------------------------------------------------------------------------
# Business profile models
# ---------------------------------------------------------------------------


class BusinessLocation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    roastery_address: str
    city: str
    state: str
    country: str
    serves_wholesale_locally: bool
    ships_pan_india: bool


class BusinessContact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    whatsapp_support_hours: str
    support_email: str
    website: str


class BusinessHours(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    day: str
    open: Optional[str] = None
    close: Optional[str] = None
    closed: bool = False

    @model_validator(mode="after")
    def _validate_hours_consistency(self) -> "BusinessHours":
        if not self.closed and (not self.open or not self.close):
            raise ValueError(f"{self.day}: open/close required unless closed=true")
        return self


class ShippingZone(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    zone: str
    typical_delivery_days: str


class ShippingPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    summary: str
    domestic_zones: List[ShippingZone]
    shipping_fee_note: str
    international_shipping: bool


class ReturnPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    summary: str
    window_days: int = Field(..., ge=0, le=90)
    conditions: List[str]
    refund_or_replacement_handled_by: str


class WholesaleInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    summary: str
    typical_minimum_order_kg: float = Field(..., ge=0)
    customer_types: List[str]
    process_note: str
    pricing_disclosed_by_agent: bool

    @field_validator("pricing_disclosed_by_agent")
    @classmethod
    def _agent_must_not_disclose_pricing(cls, value: bool) -> bool:
        # This is a hard business rule, not just a default: the agent has no
        # pricing authority, so this file must never flip it to True.
        if value is not False:
            raise ValueError(
                "wholesale_info.pricing_disclosed_by_agent must be false — "
                "the agent must never be configured to quote wholesale pricing"
            )
        return value


class FAQItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question: str
    answer: str


class BusinessProfile(BaseModel):
    """Publishable facts about the business the agent may safely state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_fictional: bool
    fictional_notice: str
    name: str
    short_name: str
    founded_year: int = Field(..., ge=1900, le=2100)
    description: str
    location: BusinessLocation
    contact: BusinessContact
    hours: List[BusinessHours]
    shipping_policy: ShippingPolicy
    return_policy: ReturnPolicy
    wholesale_info: WholesaleInfo
    supported_customer_types: List[str]
    payment_methods_note: str
    faqs: List[FAQItem] = Field(..., min_length=1)

    @field_validator("is_fictional")
    @classmethod
    def _must_be_marked_fictional(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError(
                "business.json must have is_fictional=true — this project must "
                "never present a real business's data as a fictional demo"
            )
        return value

    @model_validator(mode="after")
    def _validate_unique_faq_ids(self) -> "BusinessProfile":
        ids = [f.id for f in self.faqs]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"Duplicate FAQ ids: {sorted(duplicates)}")
        return self


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_json_file(path: Path, label: str) -> dict:
    if not path.exists():
        raise KnowledgeError(f"{label} file not found: {path}")
    if not path.is_file():
        raise KnowledgeError(f"{label} path is not a file: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise KnowledgeError(f"Failed to read {label} file at {path}: {exc}") from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise KnowledgeError(f"{label} file at {path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise KnowledgeError(f"{label} file at {path} must contain a JSON object at the top level")

    return data


def load_business_profile(path: Optional[Path] = None) -> BusinessProfile:
    """Load and validate the business profile from ``business.json``.

    Args:
        path: Optional override path (used by tests for fixture files).

    Returns:
        A validated, immutable ``BusinessProfile``.

    Raises:
        KnowledgeError: If the file is missing, unreadable, not valid JSON,
            or fails schema validation.
    """
    resolved_path = Path(path) if path is not None else DEFAULT_BUSINESS_PROFILE_PATH
    data = _read_json_file(resolved_path, "Business profile")

    try:
        return BusinessProfile.model_validate(data)
    except Exception as exc:
        raise KnowledgeError(
            f"Business profile at {resolved_path} failed validation: {exc}"
        ) from exc


def load_catalog(path: Optional[Path] = None) -> Catalog:
    """Load and validate the product catalog from ``catalog.json``.

    Args:
        path: Optional override path (used by tests for fixture files).

    Returns:
        A validated, immutable ``Catalog``.

    Raises:
        KnowledgeError: If the file is missing, unreadable, not valid JSON,
            or fails schema validation.
    """
    resolved_path = Path(path) if path is not None else DEFAULT_CATALOG_PATH
    data = _read_json_file(resolved_path, "Catalog")

    try:
        return Catalog.model_validate(data)
    except Exception as exc:
        raise KnowledgeError(f"Catalog at {resolved_path} failed validation: {exc}") from exc


class KnowledgeBase(BaseModel):
    """Immutable, validated bundle of business + catalog knowledge.

    This is the object later slices (prompt builder, product-lookup tool)
    are expected to consume. It performs no network or LLM calls; it is a
    pure in-memory data container built once from validated JSON.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    business: BusinessProfile
    catalog: Catalog

    @classmethod
    def load(
        cls,
        business_path: Optional[Path] = None,
        catalog_path: Optional[Path] = None,
    ) -> "KnowledgeBase":
        """Load both the business profile and catalog and bundle them.

        Raises:
            KnowledgeError: If either file is missing or fails validation.
        """
        business = load_business_profile(business_path)
        catalog = load_catalog(catalog_path)
        logger.info(
            "Loaded knowledge base: business=%s, catalog_version=%s, products=%d",
            business.short_name,
            catalog.catalog_version,
            len(catalog.products),
        )
        return cls(business=business, catalog=catalog)


@lru_cache()
def get_knowledge_base() -> KnowledgeBase:
    """Return a cached ``KnowledgeBase`` loaded from the default file paths.

    Not wired into the running application in this slice. Provided for later
    use by the prompt builder / orchestrator, and to give tests a single,
    consistent entry point that mirrors ``app.config.get_settings``.
    """
    return KnowledgeBase.load()


# ---------------------------------------------------------------------------
# Digest builders
# ---------------------------------------------------------------------------


def build_business_digest(knowledge: KnowledgeBase) -> str:
    """Build a compact, deterministic text digest of business facts.

    Intended for a future prompt builder to embed directly into the system
    prompt as grounding context. Output is plain text (no markdown tables,
    consistent with the WhatsApp-appropriate style already used elsewhere in
    this project), and is deterministic for a given ``KnowledgeBase``.
    """
    b = knowledge.business
    lines: List[str] = []

    lines.append(f"Business: {b.name} ({b.short_name}), founded {b.founded_year}.")
    lines.append(b.description)
    lines.append(
        f"Location: {b.location.roastery_address}. Ships pan-India: "
        f"{'yes' if b.location.ships_pan_india else 'no'}."
    )

    hours_by_span: dict = {}
    for h in b.hours:
        span = "closed" if h.closed else f"{h.open}-{h.close}"
        hours_by_span.setdefault(span, []).append(h.day)
    hours_text = "; ".join(
        f"{', '.join(days)}: {span}" for span, days in sorted(hours_by_span.items())
    )
    lines.append(f"Hours: {hours_text}")

    lines.append(f"Shipping: {b.shipping_policy.summary} {b.shipping_policy.shipping_fee_note}")
    for zone in b.shipping_policy.domestic_zones:
        lines.append(f"  - {zone.zone}: {zone.typical_delivery_days} days")

    lines.append(
        f"Returns: {b.return_policy.summary} "
        f"(window: {b.return_policy.window_days} days)"
    )

    lines.append(
        f"Wholesale: {b.wholesale_info.summary} "
        f"Typical minimum order: {b.wholesale_info.typical_minimum_order_kg}kg. "
        f"Customer types: {', '.join(b.wholesale_info.customer_types)}."
    )

    lines.append(f"Payments: {b.payment_methods_note}")

    lines.append("FAQs:")
    for faq in b.faqs:
        lines.append(f"  Q: {faq.question}")
        lines.append(f"  A: {faq.answer}")

    return "\n".join(lines)


def build_catalog_digest(knowledge: KnowledgeBase, max_items: Optional[int] = None) -> str:
    """Build a compact, deterministic text digest of the product catalog.

    Products are sorted by SKU for determinism. Intended as grounding
    context for a future prompt builder; it is NOT the mechanism a future
    ``product_lookup`` tool will use to answer specific queries.

    Args:
        knowledge: The loaded knowledge base.
        max_items: Optional cap on the number of products included.
    """
    products = sorted(knowledge.catalog.products, key=lambda p: p.sku)
    if max_items is not None:
        products = products[:max_items]

    lines: List[str] = [
        f"Catalog (version {knowledge.catalog.catalog_version}, "
        f"{len(knowledge.catalog.products)} products):"
    ]
    for p in products:
        detail_bits = [f"{p.category.value}", f"INR {p.price_inr:g}"]
        if p.size_g:
            detail_bits.append(f"{p.size_g}g")
        if p.roast_level:
            detail_bits.append(f"{p.roast_level.value} roast")
        detail_bits.append("in stock" if p.in_stock else "out of stock")
        lines.append(f"  - {p.sku} | {p.name} | {', '.join(detail_bits)}")

    return "\n".join(lines)
