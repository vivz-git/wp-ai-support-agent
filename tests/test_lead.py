"""Tests for the lead qualification domain model (app/agent/lead.py).

Self-contained: does not depend on tests/conftest.py's WhatsApp/Groq mocks.
Everything here is pure data + deterministic rules; no network, no LLM.
"""

import pytest
from pydantic import ValidationError

from app.agent.lead import (
    CONSUMER_REQUIRED_FIELDS,
    MAX_MONTHLY_VOLUME_KG,
    WHOLESALE_REQUIRED_FIELDS,
    BudgetBand,
    BusinessType,
    LeadDelta,
    LeadProfile,
    LeadSource,
    LeadTrack,
    QualificationState,
    Timeline,
    evaluate_qualification,
    merge_lead_delta,
)
from app.knowledge import BrewMethod

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WHOLESALE_COMPLETE = {
    "track": "wholesale",
    "contact_name": "Asha Rao",
    "business_name": "Third Wave Cafe",
    "business_type": "cafe",
    "monthly_volume_kg": 25,
    "city": "Bengaluru",
    "timeline": "within_1_month",
}

CONSUMER_COMPLETE = {
    "track": "consumer",
    "contact_name": "Ravi",
    "brew_method": "pourover",
    "taste_preference": "fruity and bright",
}


def _wholesale_profile(**overrides) -> LeadProfile:
    data = {**WHOLESALE_COMPLETE, **overrides}
    return LeadProfile.model_validate(data)


def _consumer_profile(**overrides) -> LeadProfile:
    data = {**CONSUMER_COMPLETE, **overrides}
    return LeadProfile.model_validate(data)


# ---------------------------------------------------------------------------
# 7. LeadProfile defaults
# ---------------------------------------------------------------------------


def test_lead_profile_defaults_are_all_unknown():
    profile = LeadProfile()
    assert profile.track == LeadTrack.UNKNOWN
    assert profile.source == LeadSource.WHATSAPP
    assert profile.field_provenance == {}
    for name in (
        "contact_name",
        "whatsapp_number",
        "email",
        "city",
        "intent_summary",
        "brew_method",
        "taste_preference",
        "budget_band",
        "subscription_interest",
        "business_name",
        "business_type",
        "monthly_volume_kg",
        "timeline",
        "current_supplier",
    ):
        assert getattr(profile, name) is None, name
    assert profile.has_any_lead_data() is False
    assert profile.is_complete() is False
    assert profile.effective_track() == LeadTrack.UNKNOWN


def test_lead_profile_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        LeadProfile(qualification="qualified")


# ---------------------------------------------------------------------------
# 8. Valid wholesale profile
# ---------------------------------------------------------------------------


def test_valid_wholesale_profile():
    profile = _wholesale_profile(email="ASHA@ThirdWave.example", current_supplier="Local roaster")
    assert profile.track == LeadTrack.WHOLESALE
    assert profile.business_type == BusinessType.CAFE
    assert profile.timeline == Timeline.WITHIN_1_MONTH
    assert profile.monthly_volume_kg == 25.0
    assert profile.email == "asha@thirdwave.example"  # normalised to lowercase
    assert profile.missing_required_fields() == []
    assert profile.is_complete() is True


# ---------------------------------------------------------------------------
# 9. Valid consumer profile
# ---------------------------------------------------------------------------


def test_valid_consumer_profile():
    profile = _consumer_profile(budget_band="500_1000", subscription_interest=True)
    assert profile.track == LeadTrack.CONSUMER
    assert profile.brew_method == BrewMethod.POUROVER
    assert profile.budget_band == BudgetBand.FROM_500_TO_1000
    assert profile.subscription_interest is True
    assert profile.is_complete() is True


def test_effective_track_is_inferred_from_data_when_not_set():
    assert LeadProfile(business_name="Cafe X").effective_track() == LeadTrack.WHOLESALE
    assert LeadProfile(brew_method="espresso").effective_track() == LeadTrack.CONSUMER
    assert LeadProfile(contact_name="Only a name").effective_track() == LeadTrack.UNKNOWN
    # Explicit track wins over inference.
    assert LeadProfile(track="consumer", business_name="Cafe X").effective_track() == LeadTrack.CONSUMER


# ---------------------------------------------------------------------------
# 10. Invalid business_type
# ---------------------------------------------------------------------------


def test_invalid_business_type_rejected():
    with pytest.raises(ValidationError):
        _wholesale_profile(business_type="spaceport")
    with pytest.raises(ValidationError):
        LeadDelta(business_type="spaceport")


# ---------------------------------------------------------------------------
# 11. Invalid timeline
# ---------------------------------------------------------------------------


def test_invalid_timeline_rejected():
    with pytest.raises(ValidationError):
        _wholesale_profile(timeline="someday")
    with pytest.raises(ValidationError):
        LeadDelta(timeline="someday")


# ---------------------------------------------------------------------------
# 12. Invalid monthly volume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_volume", [-1, -0.5, MAX_MONTHLY_VOLUME_KG + 1, 1e9, "twenty"])
def test_invalid_monthly_volume_rejected(bad_volume):
    with pytest.raises(ValidationError):
        _wholesale_profile(monthly_volume_kg=bad_volume)
    with pytest.raises(ValidationError):
        LeadDelta(monthly_volume_kg=bad_volume)


def test_monthly_volume_bounds_are_inclusive():
    assert _wholesale_profile(monthly_volume_kg=0).monthly_volume_kg == 0
    assert _wholesale_profile(monthly_volume_kg=MAX_MONTHLY_VOLUME_KG).monthly_volume_kg == MAX_MONTHLY_VOLUME_KG


# ---------------------------------------------------------------------------
# 13. Invalid email
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_email", ["not-an-email", "a@b", "@example.com", "user@", "a b@example.com", "   "])
def test_invalid_email_rejected(bad_email):
    with pytest.raises(ValidationError):
        LeadProfile(email=bad_email)
    with pytest.raises(ValidationError):
        LeadDelta(email=bad_email)


def test_valid_email_accepted_and_normalised():
    assert LeadProfile(email="  Ravi@Example.com ").email == "ravi@example.com"


# ---------------------------------------------------------------------------
# 14. Whitespace-only names rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["contact_name", "business_name"])
@pytest.mark.parametrize("blank", ["", " ", "\t\n  "])
def test_whitespace_only_names_rejected(field, blank):
    with pytest.raises(ValidationError):
        LeadProfile(**{field: blank})
    with pytest.raises(ValidationError):
        LeadDelta(**{field: blank})


def test_names_are_trimmed_but_kept():
    assert LeadProfile(contact_name="  Asha  ").contact_name == "Asha"


def test_over_long_values_rejected():
    with pytest.raises(ValidationError):
        LeadProfile(contact_name="x" * 81)
    with pytest.raises(ValidationError):
        LeadProfile(intent_summary="x" * 301)
    with pytest.raises(ValidationError):
        LeadDelta(business_name="x" * 121)


def test_whatsapp_number_must_be_digits():
    assert LeadProfile(whatsapp_number="919876543210").whatsapp_number == "919876543210"
    with pytest.raises(ValidationError):
        LeadProfile(whatsapp_number="+91 98765")
    with pytest.raises(ValidationError):
        LeadProfile(whatsapp_number="12")


def test_whatsapp_number_is_not_a_delta_field():
    # It is webhook metadata: an extractor must not be able to set it.
    with pytest.raises(ValidationError):
        LeadDelta(whatsapp_number="919876543210")


# ---------------------------------------------------------------------------
# 15. LeadDelta partial update
# ---------------------------------------------------------------------------


def test_lead_delta_partial_update_only_needs_provided_fields():
    delta = LeadDelta(city="Mysuru")
    assert delta.provided_fields() == {"city": "Mysuru"}
    assert delta.is_empty() is False
    assert LeadDelta().is_empty() is True

    merged = merge_lead_delta(LeadProfile(), delta, turn=1)
    assert merged.city == "Mysuru"
    assert merged.contact_name is None
    assert merged.field_provenance == {"city": 1}


def test_lead_delta_has_no_qualification_field():
    with pytest.raises(ValidationError):
        LeadDelta(qualification="qualified")
    with pytest.raises(ValidationError):
        LeadDelta(qualification=QualificationState.QUALIFIED)


# ---------------------------------------------------------------------------
# 16. Merge preserves existing values
# ---------------------------------------------------------------------------


def test_merge_preserves_existing_values_and_input_profile():
    base = LeadProfile(contact_name="Asha", city="Bengaluru", field_provenance={"contact_name": 1, "city": 1})
    merged = merge_lead_delta(base, LeadDelta(business_name="Third Wave Cafe"), turn=2)

    assert merged.contact_name == "Asha"
    assert merged.city == "Bengaluru"
    assert merged.business_name == "Third Wave Cafe"
    # The original is untouched (value semantics).
    assert base.business_name is None
    assert base.field_provenance == {"contact_name": 1, "city": 1}


def test_merge_overwrites_with_new_non_null_value_and_reprovenances():
    base = LeadProfile(city="Bengaluru", field_provenance={"city": 1})
    merged = merge_lead_delta(base, LeadDelta(city="Mysuru"), turn=3)
    assert merged.city == "Mysuru"
    assert merged.field_provenance == {"city": 3}


# ---------------------------------------------------------------------------
# 17. Merge does not overwrite with None
# ---------------------------------------------------------------------------


def test_merge_never_overwrites_with_none():
    base = _wholesale_profile(email="asha@example.com")
    empty = LeadDelta()  # every field None
    merged = merge_lead_delta(base, empty, turn=5)
    assert merged == base

    partial = LeadDelta(contact_name=None, city=None, timeline="immediate")
    merged = merge_lead_delta(base, partial, turn=5)
    assert merged.contact_name == "Asha Rao"
    assert merged.city == "Bengaluru"
    assert merged.email == "asha@example.com"
    assert merged.timeline == Timeline.IMMEDIATE


def test_merge_ignores_unknown_track_in_delta():
    base = LeadProfile(track="wholesale")
    merged = merge_lead_delta(base, LeadDelta(track="unknown", city="Pune"), turn=2)
    assert merged.track == LeadTrack.WHOLESALE
    assert "track" not in merged.field_provenance
    assert merged.field_provenance == {"city": 2}


def test_merge_rejects_negative_turn():
    with pytest.raises(ValueError):
        merge_lead_delta(LeadProfile(), LeadDelta(city="Pune"), turn=-1)


# ---------------------------------------------------------------------------
# 18. Field provenance
# ---------------------------------------------------------------------------


def test_field_provenance_tracks_turn_per_field():
    profile = LeadProfile()
    profile = merge_lead_delta(profile, LeadDelta(contact_name="Asha"), turn=1)
    profile = merge_lead_delta(profile, LeadDelta(business_name="Cafe X", business_type="cafe"), turn=2)
    profile = merge_lead_delta(profile, LeadDelta(contact_name="Asha Rao"), turn=4)
    assert profile.field_provenance == {
        "contact_name": 4,
        "business_name": 2,
        "business_type": 2,
    }


def test_field_provenance_rejects_unknown_field():
    with pytest.raises(ValidationError):
        LeadProfile(contact_name="Asha", field_provenance={"nickname": 1})


def test_field_provenance_rejects_invalid_turn_numbers():
    with pytest.raises(ValidationError):
        LeadProfile(contact_name="Asha", field_provenance={"contact_name": -1})
    with pytest.raises(ValidationError):
        LeadProfile(contact_name="Asha", field_provenance={"contact_name": "one"})


def test_field_provenance_rejects_entry_for_unset_field():
    with pytest.raises(ValidationError):
        LeadProfile(field_provenance={"contact_name": 1})


def test_field_provenance_cannot_reference_whatsapp_number():
    # whatsapp_number is metadata, not an extracted lead field.
    with pytest.raises(ValidationError):
        LeadProfile(whatsapp_number="919876543210", field_provenance={"whatsapp_number": 0})


# ---------------------------------------------------------------------------
# 19. Wholesale qualification matrix
# ---------------------------------------------------------------------------


def test_wholesale_required_fields_are_exactly_the_spec():
    assert set(WHOLESALE_REQUIRED_FIELDS) == {
        "contact_name",
        "business_name",
        "business_type",
        "monthly_volume_kg",
        "city",
        "timeline",
    }


@pytest.mark.parametrize("missing", WHOLESALE_REQUIRED_FIELDS)
def test_wholesale_missing_any_required_field_is_not_qualified(missing):
    profile = _wholesale_profile(**{missing: None})
    assert profile.is_complete() is False
    assert profile.missing_required_fields() == [missing]
    assert (
        evaluate_qualification(profile, QualificationState.COLLECTING, turn_count=3)
        == QualificationState.COLLECTING
    )


def test_wholesale_complete_profile_is_qualified():
    profile = _wholesale_profile()
    assert evaluate_qualification(profile, QualificationState.COLLECTING, turn_count=3) == QualificationState.QUALIFIED
    # Optional wholesale fields are not required.
    assert profile.email is None and profile.current_supplier is None


def test_wholesale_qualification_does_not_need_consumer_fields():
    profile = _wholesale_profile()
    assert profile.brew_method is None and profile.taste_preference is None
    assert profile.is_complete() is True


# ---------------------------------------------------------------------------
# 20. Consumer qualification matrix
# ---------------------------------------------------------------------------


def test_consumer_required_fields_are_exactly_the_spec():
    assert set(CONSUMER_REQUIRED_FIELDS) == {"contact_name", "brew_method", "taste_preference"}


@pytest.mark.parametrize("missing", CONSUMER_REQUIRED_FIELDS)
def test_consumer_missing_any_required_field_is_not_qualified(missing):
    profile = _consumer_profile(**{missing: None})
    assert profile.is_complete() is False
    assert profile.missing_required_fields() == [missing]
    assert (
        evaluate_qualification(profile, QualificationState.COLLECTING, turn_count=3)
        == QualificationState.COLLECTING
    )


def test_consumer_complete_profile_is_qualified():
    profile = _consumer_profile()
    assert evaluate_qualification(profile, QualificationState.COLLECTING, turn_count=3) == QualificationState.QUALIFIED


def test_consumer_qualification_does_not_need_wholesale_fields():
    profile = _consumer_profile()
    assert profile.business_name is None and profile.monthly_volume_kg is None
    assert profile.is_complete() is True


# ---------------------------------------------------------------------------
# 21. Incomplete profile is not qualified
# ---------------------------------------------------------------------------


def test_incomplete_profile_is_never_qualified():
    assert evaluate_qualification(LeadProfile(), QualificationState.UNKNOWN, turn_count=0) == QualificationState.UNKNOWN
    assert evaluate_qualification(LeadProfile(), QualificationState.UNKNOWN, turn_count=1) == QualificationState.BROWSING

    only_name = LeadProfile(contact_name="Asha")
    assert only_name.effective_track() == LeadTrack.UNKNOWN
    assert evaluate_qualification(only_name, QualificationState.BROWSING, turn_count=2) == QualificationState.COLLECTING

    # A consumer-complete data set declared as wholesale is NOT wholesale-qualified.
    mislabelled = _consumer_profile(track="wholesale")
    assert mislabelled.is_complete() is False
    assert evaluate_qualification(mislabelled, QualificationState.COLLECTING, turn_count=3) == QualificationState.COLLECTING


def test_qualification_cannot_be_asserted_only_computed():
    # Even claiming to be qualified in `current` is recomputed from data.
    assert (
        evaluate_qualification(LeadProfile(contact_name="Asha"), QualificationState.QUALIFIED, turn_count=2)
        == QualificationState.COLLECTING
    )


def test_sticky_states_survive_reevaluation():
    complete = _wholesale_profile()
    for sticky in (QualificationState.HANDOFF_READY, QualificationState.DECLINED, QualificationState.ESCALATED):
        assert evaluate_qualification(complete, sticky, turn_count=9) == sticky
        assert evaluate_qualification(LeadProfile(), sticky, turn_count=0) == sticky


def test_evaluate_is_deterministic():
    profile = _wholesale_profile(monthly_volume_kg=5)
    results = {evaluate_qualification(profile, QualificationState.COLLECTING, 4) for _ in range(20)}
    assert results == {QualificationState.QUALIFIED}
