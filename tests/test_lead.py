"""Tests for the patient lead (booking request) model in app/agent/lead.py."""

import pytest
from pydantic import ValidationError

from app.agent.lead import (
    BOOKING_QUESTIONS,
    LEAD_DATA_FIELDS,
    QUESTION_ORDER,
    REQUIRED_FIELDS,
    LeadDelta,
    LeadProfile,
    LeadSource,
    QualificationState,
    STICKY_QUALIFICATION_STATES,
    evaluate_qualification,
    merge_lead_delta,
)


def _complete(**overrides) -> LeadProfile:
    data = dict(patient_name="Priya Sharma", concern="cleaning", preferred_day_time="Saturday 11am", whatsapp_number="919876543210")
    data.update(overrides)
    return LeadProfile(**data)


# ---------------------------------------------------------------------------
# 1. Defaults and shape
# ---------------------------------------------------------------------------


def test_lead_profile_defaults_are_all_unknown():
    profile = LeadProfile()
    for name in LEAD_DATA_FIELDS:
        assert getattr(profile, name) is None
    assert profile.whatsapp_number is None
    assert profile.source == LeadSource.WHATSAPP
    assert profile.field_provenance == {}
    assert not profile.has_any_lead_data()
    assert not profile.is_complete()


def test_lead_data_fields_are_the_four_booking_fields():
    assert LEAD_DATA_FIELDS == ("patient_name", "phone", "concern", "preferred_day_time")
    assert REQUIRED_FIELDS == LEAD_DATA_FIELDS
    assert set(QUESTION_ORDER) == set(LEAD_DATA_FIELDS)
    assert set(BOOKING_QUESTIONS) == set(LEAD_DATA_FIELDS)


def test_lead_profile_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        LeadProfile(brew_method="espresso")


# ---------------------------------------------------------------------------
# 2. Field validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_whitespace_only_name_rejected(blank):
    with pytest.raises(ValidationError):
        LeadProfile(patient_name=blank)
    with pytest.raises(ValidationError):
        LeadDelta(patient_name=blank)


def test_names_are_trimmed_but_kept():
    assert LeadProfile(patient_name="  Rahul  ").patient_name == "Rahul"


def test_free_text_blank_becomes_none():
    delta = LeadDelta(concern="   ", preferred_day_time="")
    assert delta.concern is None and delta.preferred_day_time is None
    assert delta.is_empty()


@pytest.mark.parametrize(
    "raw, normalized",
    [
        ("98765 43210", "9876543210"),
        ("+91-98765-43210", "919876543210"),
        ("(020) 5550 0142", "02055500142"),
        ("+91 98765.43210", "919876543210"),
    ],
)
def test_phone_is_normalized_to_digits(raw, normalized):
    assert LeadDelta(phone=raw).phone == normalized
    assert LeadProfile(phone=raw).phone == normalized


@pytest.mark.parametrize("bad", ["12345", "call me", "98765-4321x", "+" + "1" * 16, "98765 43210 ext 2"])
def test_invalid_phone_rejected(bad):
    with pytest.raises(ValidationError):
        LeadDelta(phone=bad)


def test_over_long_values_rejected():
    with pytest.raises(ValidationError):
        LeadProfile(patient_name="x" * 81)
    with pytest.raises(ValidationError):
        LeadProfile(concern="x" * 201)
    with pytest.raises(ValidationError):
        LeadProfile(preferred_day_time="x" * 81)


def test_whatsapp_number_must_be_digits():
    assert LeadProfile(whatsapp_number="919876543210").whatsapp_number == "919876543210"
    with pytest.raises(ValidationError):
        LeadProfile(whatsapp_number="+91 98765")


def test_whatsapp_number_is_not_a_delta_field():
    with pytest.raises(ValidationError):
        LeadDelta(whatsapp_number="919876543210")


def test_lead_delta_has_no_qualification_field():
    with pytest.raises(ValidationError):
        LeadDelta(qualification="qualified")
    assert "qualification" not in LeadDelta.model_fields


# ---------------------------------------------------------------------------
# 3. Callback phone defaults to WhatsApp
# ---------------------------------------------------------------------------


def test_callback_phone_defaults_to_whatsapp_number():
    profile = LeadProfile(whatsapp_number="919876543210")
    assert profile.phone is None
    assert profile.callback_phone() == "919876543210"
    assert "phone" not in profile.missing_required_fields()


def test_stated_phone_overrides_whatsapp_number():
    profile = LeadProfile(whatsapp_number="919876543210", phone="9123456789")
    assert profile.callback_phone() == "9123456789"


def test_phone_required_when_no_whatsapp_number():
    profile = LeadProfile(patient_name="A", concern="b", preferred_day_time="c")
    assert profile.missing_required_fields() == ["phone"]
    assert not profile.is_complete()


def test_whatsapp_number_alone_is_not_lead_data():
    assert not LeadProfile(whatsapp_number="919876543210").has_any_lead_data()


# ---------------------------------------------------------------------------
# 4. One missing field at a time
# ---------------------------------------------------------------------------


def test_next_missing_field_follows_question_order():
    profile = LeadProfile(whatsapp_number="919876543210")
    assert profile.next_missing_field() == "concern"
    profile = profile.model_copy(update={"concern": "braces"})
    assert profile.next_missing_field() == "patient_name"
    profile = profile.model_copy(update={"patient_name": "Asha"})
    assert profile.next_missing_field() == "preferred_day_time"
    profile = profile.model_copy(update={"preferred_day_time": "kal shaam"})
    assert profile.next_missing_field() is None


def test_next_missing_field_asks_for_phone_last_when_no_whatsapp():
    profile = LeadProfile(patient_name="A", concern="b", preferred_day_time="c")
    assert profile.next_missing_field() == "phone"


def test_booking_questions_are_single_questions():
    for question in BOOKING_QUESTIONS.values():
        assert question.count("?") == 1


# ---------------------------------------------------------------------------
# 5. Delta merge
# ---------------------------------------------------------------------------


def test_lead_delta_partial_update_only_needs_provided_fields():
    delta = LeadDelta(concern="daant mein dard")
    assert delta.provided_fields() == {"concern": "daant mein dard"}


def test_merge_preserves_existing_values_and_input_profile():
    profile = LeadProfile(patient_name="Asha")
    merged = merge_lead_delta(profile, LeadDelta(concern="braces"), turn=2)
    assert merged.patient_name == "Asha"
    assert merged.concern == "braces"
    assert profile.concern is None  # input untouched


def test_merge_overwrites_with_new_non_null_value_and_reprovenances():
    profile = merge_lead_delta(LeadProfile(), LeadDelta(preferred_day_time="Monday"), turn=1)
    merged = merge_lead_delta(profile, LeadDelta(preferred_day_time="Tuesday 5pm"), turn=3)
    assert merged.preferred_day_time == "Tuesday 5pm"
    assert merged.field_provenance == {"preferred_day_time": 3}


def test_merge_never_overwrites_with_none():
    profile = merge_lead_delta(LeadProfile(), LeadDelta(patient_name="Asha", concern="rct"), turn=1)
    merged = merge_lead_delta(profile, LeadDelta(), turn=2)
    assert (merged.patient_name, merged.concern) == ("Asha", "rct")
    assert merged.field_provenance == {"patient_name": 1, "concern": 1}


def test_merge_rejects_negative_turn():
    with pytest.raises(ValueError):
        merge_lead_delta(LeadProfile(), LeadDelta(concern="x"), turn=-1)


# ---------------------------------------------------------------------------
# 6. Provenance
# ---------------------------------------------------------------------------


def test_field_provenance_tracks_turn_per_field():
    profile = merge_lead_delta(LeadProfile(), LeadDelta(concern="cleaning"), turn=1)
    profile = merge_lead_delta(profile, LeadDelta(patient_name="Asha", phone="9123456789"), turn=2)
    assert profile.field_provenance == {"concern": 1, "patient_name": 2, "phone": 2}


def test_field_provenance_rejects_unknown_field():
    with pytest.raises(ValidationError):
        LeadProfile(field_provenance={"brew_method": 1})


@pytest.mark.parametrize("bad_turn", [-1, "first", 1.5])
def test_field_provenance_rejects_invalid_turn_numbers(bad_turn):
    with pytest.raises(ValidationError):
        LeadProfile(concern="x", field_provenance={"concern": bad_turn})


def test_field_provenance_rejects_entry_for_unset_field():
    with pytest.raises(ValidationError):
        LeadProfile(field_provenance={"concern": 1})


def test_field_provenance_cannot_reference_whatsapp_number():
    with pytest.raises(ValidationError):
        LeadProfile(whatsapp_number="919876543210", field_provenance={"whatsapp_number": 0})


# ---------------------------------------------------------------------------
# 7. Qualification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["patient_name", "concern", "preferred_day_time"])
def test_missing_any_required_field_is_not_qualified(missing):
    profile = _complete(**{missing: None})
    assert not profile.is_complete()
    assert evaluate_qualification(profile, QualificationState.COLLECTING, 3) == QualificationState.COLLECTING


def test_complete_profile_is_qualified():
    profile = _complete()
    assert profile.is_complete()
    assert evaluate_qualification(profile, QualificationState.COLLECTING, 4) == QualificationState.QUALIFIED


def test_complete_profile_without_whatsapp_needs_stated_phone():
    profile = _complete(whatsapp_number=None)
    assert evaluate_qualification(profile, QualificationState.COLLECTING, 4) == QualificationState.COLLECTING
    profile = _complete(whatsapp_number=None, phone="9123456789")
    assert evaluate_qualification(profile, QualificationState.COLLECTING, 4) == QualificationState.QUALIFIED


def test_no_data_is_browsing_after_first_turn_and_unknown_before():
    assert evaluate_qualification(LeadProfile(), QualificationState.UNKNOWN, 0) == QualificationState.UNKNOWN
    assert evaluate_qualification(LeadProfile(), QualificationState.UNKNOWN, 1) == QualificationState.BROWSING
    whatsapp_only = LeadProfile(whatsapp_number="919876543210")
    assert evaluate_qualification(whatsapp_only, QualificationState.UNKNOWN, 1) == QualificationState.BROWSING


def test_partial_data_is_collecting():
    assert evaluate_qualification(LeadProfile(concern="rct"), QualificationState.BROWSING, 2) == QualificationState.COLLECTING


@pytest.mark.parametrize("sticky", sorted(STICKY_QUALIFICATION_STATES, key=lambda s: s.value))
def test_sticky_states_survive_reevaluation(sticky):
    assert evaluate_qualification(_complete(), sticky, 5) == sticky
    assert evaluate_qualification(LeadProfile(), sticky, 5) == sticky


def test_evaluate_is_deterministic():
    profile = _complete(concern=None)
    results = {evaluate_qualification(profile, QualificationState.BROWSING, 3) for _ in range(5)}
    assert results == {QualificationState.COLLECTING}
