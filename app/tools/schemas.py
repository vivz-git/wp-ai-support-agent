"""Pydantic schemas for the ``clinic_faq_lookup`` tool.

These models define the tool's public contract: what the LLM (via Groq's
native function calling) may send as arguments, and exactly what shape the
handler returns. They are decoupled from the on-disk ``ClinicInfo`` models in
``app.knowledge`` so the tool contract can evolve independently.
"""

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ClinicTopic(str, Enum):
    """Whole categories the model can ask for without a free-text query."""

    SERVICES = "services"
    HOURS = "hours"
    LOCATION = "location"
    DENTISTS = "dentists"


class LookupStatus(str, Enum):
    OK = "ok"
    NO_MATCH = "no_match"
    INVALID_INPUT = "invalid_input"
    UNAVAILABLE = "unavailable"


class ClinicFaqLookupInput(BaseModel):
    """Validated arguments for ``clinic_faq_lookup``.

    At least one of ``query`` or ``topic`` is required. ``query`` may be in
    English, Hindi (Devanagari) or Hinglish; matching is keyword/alias based.
    """

    model_config = ConfigDict(extra="forbid")

    query: Optional[str] = Field(
        None,
        max_length=200,
        description="The patient's question or key words, e.g. 'RCT price', 'sunday open?', 'braces kitna'.",
    )
    topic: Optional[ClinicTopic] = Field(None, description="Return a whole category instead of (or as well as) matching a query.")
    limit: int = Field(5, ge=1, le=5)

    @field_validator("query")
    @classmethod
    def _normalize_query_whitespace(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def _require_query_or_topic(self) -> "ClinicFaqLookupInput":
        if self.query is None and self.topic is None:
            raise ValueError("At least one of query or topic must be provided")
        return self


FactKind = Literal["service", "faq", "hours", "contact", "dentist"]


class ClinicFact(BaseModel):
    """One piece of clinic knowledge returned by the tool.

    Service facts carry their price range both as numbers (for grounding)
    and as display text (so the model quotes it verbatim).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: FactKind
    id: str
    title: str
    details: str
    price_min_inr: Optional[float] = None
    price_max_inr: Optional[float] = None
    price_range: Optional[str] = None
    price_note: Optional[str] = None


class LookupErrorInfo(BaseModel):
    """Structured, safe-to-display error detail. Never raw exception text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    message: str
    fields: List[str] = Field(default_factory=list)


class ClinicFaqLookupOutput(BaseModel):
    """The complete, validated result of a ``clinic_faq_lookup`` call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: LookupStatus
    normalized_query: Optional[str] = None
    result_count: int = 0
    results: List[ClinicFact] = Field(default_factory=list)
    suggestions: List[ClinicFact] = Field(default_factory=list)
    error: Optional[LookupErrorInfo] = None
