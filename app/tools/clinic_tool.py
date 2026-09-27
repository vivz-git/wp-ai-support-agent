"""Deterministic ``clinic_faq_lookup`` tool handler.

Matching semantics
------------------
The query is normalized with ``normalize_match_text`` (NFKC, lower-cased,
punctuation replaced by spaces), so English, Hindi (Devanagari) and Hinglish
all go through the same path. A term matches when its normalized token
sequence appears in the query as whole tokens; there is no fuzzy matching.

- Services match on their name and ``aliases`` (e.g. "rct", "root canal",
  "रूट कैनाल"). A longer matching alias scores higher.
- FAQs match on their ``keywords``; each keyword hit adds to the score.
- Timings, address/phone and the dentist list match on fixed keyword sets
  (plus each dentist's first and last name).
- A price question that names no service ("kitna lagega?", "your prices")
  returns every service, so the model can quote ranges.
- ``topic`` adds a whole category regardless of the query.

Results are ordered by score (descending), then a fixed category order,
then ID, and truncated to ``limit``. On ``no_match`` the service list is
returned as ``suggestions`` so the model can say what the clinic offers
without inventing anything.
"""

from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import ValidationError

from app.knowledge import (
    ClinicInfo,
    KnowledgeError,
    Service,
    format_inr,
    get_knowledge_base,
    normalize_match_text,
)
from app.tools.schemas import (
    ClinicFact,
    ClinicFaqLookupInput,
    ClinicFaqLookupOutput,
    ClinicTopic,
    LookupErrorInfo,
    LookupStatus,
)


def _normalized(terms: Iterable[str]) -> Tuple[str, ...]:
    return tuple(t for t in (normalize_match_text(term) for term in terms) if t)


_HOURS_TERMS = _normalized(
    [
        "timing", "timings", "time", "hours", "open", "opening", "close", "closing", "closed", "sunday",
        "holiday", "today", "kab", "khula", "khule", "khulta", "band", "samay", "baje",
        "समय", "टाइमिंग", "खुला", "खुले", "खुलता", "बंद", "रविवार", "कब",
    ]
)
_LOCATION_TERMS = _normalized(
    [
        "address", "location", "where", "located", "directions", "map", "reach", "landmark", "phone", "number",
        "call", "contact", "email", "kahan", "kaha", "kidhar", "pata", "पता", "कहाँ", "कहां", "एड्रेस", "फोन", "नंबर",
    ]
)
_DENTIST_TERMS = _normalized(
    ["doctor", "doctors", "dentist", "dentists", "dr", "specialist", "daktar", "doctor sahab", "डॉक्टर", "डेंटिस्ट"]
)
_PRICE_TERMS = _normalized(
    [
        "price", "prices", "pricing", "cost", "costs", "charge", "charges", "fee", "fees", "rate", "rates",
        "how much", "kitna", "kitne", "kitni", "lagega", "lagta", "lagti", "kharcha", "kharch", "paisa", "paise",
        "कितना", "कितने", "कितनी", "खर्च", "खर्चा", "दाम", "फीस", "कीमत", "लगेगा",
    ]
)
_SERVICES_TERMS = _normalized(["services", "treatments", "treatment", "ilaj", "ilaaj", "इलाज", "सेवाएं", "सेवाएँ"])

_CATEGORY_ORDER = {"service": 0, "hours": 1, "contact": 2, "dentist": 3, "faq": 4}

_Scored = Tuple[float, ClinicFact]


def _contains(padded_query: str, term: str) -> bool:
    return f" {term} " in padded_query


def _hits(padded_query: str, terms: Iterable[str]) -> int:
    return sum(1 for term in terms if _contains(padded_query, term))


def service_fact(service: Service) -> ClinicFact:
    return ClinicFact(
        kind="service",
        id=service.id,
        title=service.name,
        details=service.description,
        price_min_inr=service.price_min_inr,
        price_max_inr=service.price_max_inr,
        price_range=service.price_range_text(),
        price_note=service.price_note,
    )


def _hours_fact(clinic: ClinicInfo) -> ClinicFact:
    return ClinicFact(kind="hours", id="hours", title="Clinic timings", details=clinic.hours_summary())


def _contact_fact(clinic: ClinicInfo) -> ClinicFact:
    return ClinicFact(
        kind="contact",
        id="contact",
        title="Address and phone",
        details=f"{clinic.address}. Phone: {clinic.phone}. Email: {clinic.email}.",
    )


def _dentist_facts(clinic: ClinicInfo) -> List[ClinicFact]:
    return [
        ClinicFact(kind="dentist", id=d.id, title=d.name, details=f"{d.qualification}. {d.focus}.")
        for d in clinic.dentists
    ]


def _score_query(clinic: ClinicInfo, normalized_query: str) -> List[_Scored]:
    padded = f" {normalized_query} "
    scored: List[_Scored] = []

    service_matched = False
    for service in clinic.services:
        matched = [alias for alias in service.match_aliases() if _contains(padded, alias)]
        if matched:
            service_matched = True
            scored.append((10.0 + len(matched[0]), service_fact(service)))

    if not service_matched and (_hits(padded, _PRICE_TERMS) or _hits(padded, _SERVICES_TERMS)):
        scored.extend((5.0, service_fact(service)) for service in clinic.services)

    hours_hits = _hits(padded, _HOURS_TERMS)
    if hours_hits:
        scored.append((float(hours_hits), _hours_fact(clinic)))

    location_hits = _hits(padded, _LOCATION_TERMS)
    if location_hits:
        scored.append((float(location_hits), _contact_fact(clinic)))

    generic_dentist_hits = _hits(padded, _DENTIST_TERMS)
    for dentist, fact in zip(clinic.dentists, _dentist_facts(clinic)):
        name_hits = _hits(padded, _normalized(dentist.name.split()[1:]))  # drop the "Dr." prefix
        if name_hits or generic_dentist_hits:
            scored.append((float(generic_dentist_hits + 3 * name_hits), fact))

    for faq in clinic.faqs:
        faq_hits = _hits(padded, _normalized(faq.keywords))
        if faq_hits:
            scored.append(
                (float(faq_hits), ClinicFact(kind="faq", id=faq.id, title=faq.question, details=faq.answer))
            )
    return scored


def _topic_facts(clinic: ClinicInfo, topic: ClinicTopic) -> List[ClinicFact]:
    if topic == ClinicTopic.SERVICES:
        return [service_fact(s) for s in clinic.services]
    if topic == ClinicTopic.HOURS:
        return [_hours_fact(clinic)]
    if topic == ClinicTopic.LOCATION:
        return [_contact_fact(clinic)]
    return _dentist_facts(clinic)


def _rank(scored: List[_Scored], limit: int) -> List[ClinicFact]:
    best: Dict[str, _Scored] = {}
    for score, fact in scored:
        key = f"{fact.kind}:{fact.id}"
        if key not in best or score > best[key][0]:
            best[key] = (score, fact)
    ordered = sorted(best.values(), key=lambda item: (-item[0], _CATEGORY_ORDER[item[1].kind], item[1].id))
    return [fact for _, fact in ordered[:limit]]


def _invalid_input(code: str, message: str, fields: Optional[List[str]] = None) -> dict:
    return ClinicFaqLookupOutput(
        status=LookupStatus.INVALID_INPUT,
        result_count=0,
        error=LookupErrorInfo(code=code, message=message, fields=fields or []),
    ).model_dump(mode="json")


def clinic_faq_lookup(arguments: dict) -> dict:
    """Handle a ``clinic_faq_lookup`` tool call.

    Never raises: bad input, an unavailable knowledge file, or an unexpected
    internal error all become a structured result with an appropriate status.
    """
    try:
        if not isinstance(arguments, dict):
            return _invalid_input("invalid_arguments_type", "Tool arguments must be a JSON object.")

        try:
            params = ClinicFaqLookupInput.model_validate(arguments)
        except ValidationError as exc:
            fields = sorted({".".join(str(part) for part in err["loc"]) for err in exc.errors()})
            return _invalid_input(
                "validation_failed",
                "One or more clinic_faq_lookup arguments were invalid.",
                fields=fields,
            )

        try:
            clinic = get_knowledge_base().clinic
        except KnowledgeError:
            return ClinicFaqLookupOutput(
                status=LookupStatus.UNAVAILABLE,
                error=LookupErrorInfo(
                    code="clinic_info_unavailable",
                    message="Clinic information is temporarily unavailable.",
                ),
            ).model_dump(mode="json")

        normalized_query = normalize_match_text(params.query) if params.query else None
        scored = _score_query(clinic, normalized_query) if normalized_query else []
        if params.topic is not None:
            scored.extend((1.0, fact) for fact in _topic_facts(clinic, params.topic))

        results = _rank(scored, params.limit)
        if results:
            return ClinicFaqLookupOutput(
                status=LookupStatus.OK,
                normalized_query=normalized_query,
                result_count=len(results),
                results=results,
            ).model_dump(mode="json")

        return ClinicFaqLookupOutput(
            status=LookupStatus.NO_MATCH,
            normalized_query=normalized_query,
            result_count=0,
            suggestions=[service_fact(s) for s in clinic.services][: params.limit],
        ).model_dump(mode="json")

    except Exception:
        return _invalid_input("internal_error", "clinic_faq_lookup could not process the request.")


__all__ = ["clinic_faq_lookup", "format_inr", "service_fact"]
