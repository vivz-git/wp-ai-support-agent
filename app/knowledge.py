"""Clinic knowledge layer for the AI WhatsApp Support Agent.

Loads and validates ``data/clinic_info.json``, which describes the fictional
clinic ("SmileCare Dental") the agent supports: address, timings, phone,
dentists, services with rough price ranges, and FAQs.

Design constraints:
- No network calls. No LLM calls. No agent logic.
- Explicit Pydantic models, not loose dictionaries.
- Fails loudly (raises ``KnowledgeError``) on a missing or malformed file so a
  bad data file is caught at load time, not silently ignored at runtime.
- Deterministic: loading the same file twice yields identical, immutable
  data and identical digest text.
"""

import json
import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLINIC_INFO_PATH = _PROJECT_ROOT / "data" / "clinic_info.json"

WEEKDAYS: Tuple[str, ...] = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

_ZERO_WIDTH_RE = re.compile("[​-‏⁠﻿]")
_WHITESPACE_RE = re.compile(r"\s+")


class KnowledgeError(Exception):
    """Raised when clinic knowledge data is missing or malformed."""


def strip_punctuation(text: str) -> str:
    """Replace punctuation and symbols with spaces and collapse whitespace.

    Category-based rather than ``[^\\w\\s]``: Python's ``\\w`` does not match
    Devanagari vowel signs, so a regex strip would break Hindi words apart.
    """
    cleaned = "".join(" " if unicodedata.category(c)[0] in "PSC" and not c.isspace() else c for c in text)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def normalize_match_text(text: str) -> str:
    """NFKC, lower-cased, punctuation-free text used for alias/keyword matching."""
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    return strip_punctuation(normalized.lower())


def format_inr(amount: float) -> str:
    return f"₹{amount:,.0f}"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ClinicHours(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    day: str
    open: Optional[str] = None
    close: Optional[str] = None
    closed: bool = False

    @field_validator("day")
    @classmethod
    def _known_day(cls, value: str) -> str:
        if value not in WEEKDAYS:
            raise ValueError(f"unknown day '{value}'")
        return value

    @model_validator(mode="after")
    def _validate_hours_consistency(self) -> "ClinicHours":
        if not self.closed and (not self.open or not self.close):
            raise ValueError(f"{self.day}: open/close required unless closed=true")
        return self


class Dentist(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., min_length=1, max_length=40)
    name: str = Field(..., pattern=r"^Dr\. [A-Z][\w'-]+( [A-Z][\w'-]+)+$")
    qualification: str = Field(..., min_length=1, max_length=80)
    focus: str = Field(..., min_length=1, max_length=120)


class Service(BaseModel):
    """One clinic service with a rough price range in INR."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., pattern=r"^svc-[a-z0-9]+(-[a-z0-9]+)*$")
    name: str = Field(..., min_length=1, max_length=80)
    aliases: List[str] = Field(..., min_length=1)
    description: str = Field(..., min_length=1, max_length=300)
    price_min_inr: float = Field(..., gt=0, le=500000)
    price_max_inr: float = Field(..., gt=0, le=500000)
    price_note: str = Field(..., min_length=1, max_length=200)

    @field_validator("aliases")
    @classmethod
    def _aliases_not_blank(cls, value: List[str]) -> List[str]:
        for alias in value:
            if not normalize_match_text(alias):
                raise ValueError("aliases must be non-empty after normalization")
        return value

    @model_validator(mode="after")
    def _range_ordered(self) -> "Service":
        if self.price_min_inr > self.price_max_inr:
            raise ValueError(f"{self.id}: price_min_inr must not exceed price_max_inr")
        return self

    def match_aliases(self) -> List[str]:
        """Normalized name + aliases, longest first, deduplicated."""
        terms = {normalize_match_text(self.name), *(normalize_match_text(a) for a in self.aliases)}
        return sorted((t for t in terms if t), key=lambda t: (-len(t), t))

    def price_range_text(self) -> str:
        return f"{format_inr(self.price_min_inr)}–{format_inr(self.price_max_inr)}"


class FAQItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question: str
    answer: str
    keywords: List[str] = Field(default_factory=list)


class ClinicInfo(BaseModel):
    """Publishable facts about the clinic the agent may safely state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_fictional: bool
    fictional_notice: str
    name: str
    short_name: str
    description: str
    address: str
    city: str
    phone: str = Field(..., min_length=6, max_length=30)
    email: str
    hours: List[ClinicHours] = Field(..., min_length=7, max_length=7)
    dentists: List[Dentist] = Field(..., min_length=1)
    services: List[Service] = Field(..., min_length=1)
    booking_note: str
    payment_methods_note: str
    faqs: List[FAQItem] = Field(..., min_length=1)

    @field_validator("is_fictional")
    @classmethod
    def _must_be_marked_fictional(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError(
                "clinic_info.json must have is_fictional=true — this project must "
                "never present a real clinic's data as a fictional demo"
            )
        return value

    @model_validator(mode="after")
    def _validate_unique_ids_and_days(self) -> "ClinicInfo":
        for label, ids in (
            ("service", [s.id for s in self.services]),
            ("dentist", [d.id for d in self.dentists]),
            ("FAQ", [f.id for f in self.faqs]),
        ):
            duplicates = {i for i in ids if ids.count(i) > 1}
            if duplicates:
                raise ValueError(f"Duplicate {label} ids: {sorted(duplicates)}")
        if sorted(h.day for h in self.hours) != sorted(WEEKDAYS):
            raise ValueError("hours must list each weekday exactly once")
        return self

    def hours_summary(self) -> str:
        """Deterministic one-line timings, grouping days with the same span."""
        spans: dict = {}
        for day in WEEKDAYS:
            entry = next(h for h in self.hours if h.day == day)
            span = "closed" if entry.closed else f"{entry.open}-{entry.close}"
            spans.setdefault(span, []).append(day.capitalize())
        return "; ".join(f"{', '.join(days)}: {span}" for span, days in spans.items())


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


def load_clinic_info(path: Optional[Path] = None) -> ClinicInfo:
    """Load and validate ``clinic_info.json``.

    Raises:
        KnowledgeError: If the file is missing, unreadable, not valid JSON,
            or fails schema validation.
    """
    resolved_path = Path(path) if path is not None else DEFAULT_CLINIC_INFO_PATH
    data = _read_json_file(resolved_path, "Clinic info")

    try:
        return ClinicInfo.model_validate(data)
    except Exception as exc:
        raise KnowledgeError(f"Clinic info at {resolved_path} failed validation: {exc}") from exc


class KnowledgeBase(BaseModel):
    """Immutable, validated clinic knowledge consumed by prompts, tools and guardrails."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    clinic: ClinicInfo

    @classmethod
    def load(cls, clinic_path: Optional[Path] = None) -> "KnowledgeBase":
        clinic = load_clinic_info(clinic_path)
        logger.info(
            "Loaded knowledge base: clinic=%s, services=%d, dentists=%d",
            clinic.short_name,
            len(clinic.services),
            len(clinic.dentists),
        )
        return cls(clinic=clinic)


@lru_cache()
def get_knowledge_base() -> KnowledgeBase:
    """Return a cached ``KnowledgeBase`` loaded from the default file path."""
    return KnowledgeBase.load()


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------


def build_business_digest(knowledge: KnowledgeBase) -> str:
    """Compact, deterministic text digest of clinic facts for the system prompt.

    Service *prices* are deliberately left out: the model must fetch them
    through ``clinic_faq_lookup`` so every quoted range comes from a tool
    result the grounding validator can check against.
    """
    c = knowledge.clinic
    lines: List[str] = [
        f"Clinic: {c.name} ({c.short_name}). {c.description}",
        f"Address: {c.address}.",
        f"Phone: {c.phone}. Email: {c.email}.",
        f"Timings: {c.hours_summary()}.",
        "Dentists:",
    ]
    for dentist in c.dentists:
        lines.append(f"  - {dentist.name} ({dentist.qualification}): {dentist.focus}")
    lines.append("Services offered (call clinic_faq_lookup for price ranges):")
    for service in c.services:
        lines.append(f"  - {service.name}: {service.description}")
    lines.append(f"Booking: {c.booking_note}")
    lines.append(f"Payments: {c.payment_methods_note}")
    lines.append("FAQs:")
    for faq in c.faqs:
        lines.append(f"  Q: {faq.question}")
        lines.append(f"  A: {faq.answer}")
    return "\n".join(lines)
