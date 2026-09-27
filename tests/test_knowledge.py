"""Tests for the clinic knowledge layer (app/knowledge.py).

Self-contained: the knowledge layer has no dependency on the WhatsApp/Groq
mocks. The "no secrets leaked" checks read the same mock credential values
conftest.py sets in the environment, so they stay meaningful.
"""

import json
import os
import socket

import pytest
from pydantic import ValidationError

from app.knowledge import (
    DEFAULT_CLINIC_INFO_PATH,
    ClinicInfo,
    KnowledgeBase,
    KnowledgeError,
    Service,
    build_business_digest,
    get_knowledge_base,
    load_clinic_info,
    normalize_match_text,
    strip_punctuation,
)


def _raw() -> dict:
    return json.loads(DEFAULT_CLINIC_INFO_PATH.read_text(encoding="utf-8"))


def _write(tmp_path, data) -> str:
    path = tmp_path / "clinic_info.json"
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. Valid clinic info loads
# ---------------------------------------------------------------------------


def test_valid_clinic_info_loads():
    clinic = load_clinic_info()
    assert isinstance(clinic, ClinicInfo)
    assert clinic.is_fictional is True
    assert clinic.name == "SmileCare Dental"
    assert clinic.city == "Pune"
    assert clinic.address
    assert clinic.phone
    assert 2 <= len(clinic.dentists) <= 3
    assert all(d.name.startswith("Dr. ") for d in clinic.dentists)


def test_default_clinic_info_path_resolves_to_data_dir():
    assert DEFAULT_CLINIC_INFO_PATH.name == "clinic_info.json"
    assert DEFAULT_CLINIC_INFO_PATH.exists()


def test_timings_are_mon_to_sat_10_to_8_and_sunday_closed():
    clinic = load_clinic_info()
    by_day = {h.day: h for h in clinic.hours}
    for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday"):
        assert (by_day[day].open, by_day[day].close, by_day[day].closed) == ("10:00", "20:00", False)
    assert by_day["sunday"].closed is True
    assert clinic.hours_summary() == "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday: 10:00-20:00; Sunday: closed"


def test_services_cover_the_five_required_treatments_with_price_ranges():
    clinic = load_clinic_info()
    ids = {s.id for s in clinic.services}
    assert ids == {"svc-consultation", "svc-cleaning", "svc-root-canal", "svc-braces", "svc-whitening"}
    for service in clinic.services:
        assert 0 < service.price_min_inr < service.price_max_inr
        assert service.price_note


def test_root_canal_aliases_include_rct_and_devanagari():
    service = next(s for s in load_clinic_info().services if s.id == "svc-root-canal")
    aliases = service.match_aliases()
    assert "rct" in aliases
    assert normalize_match_text("रूट कैनाल") in aliases
    assert aliases == sorted(aliases, key=lambda t: (-len(t), t))


def test_price_range_text_uses_rupee_formatting():
    service = next(s for s in load_clinic_info().services if s.id == "svc-braces")
    assert service.price_range_text() == "₹35,000–₹90,000"


def test_clinic_models_are_immutable():
    clinic = load_clinic_info()
    with pytest.raises(ValidationError):
        clinic.name = "Other Clinic"
    with pytest.raises(ValidationError):
        clinic.services[0].price_min_inr = 1


# ---------------------------------------------------------------------------
# 2. Malformed files fail loudly
# ---------------------------------------------------------------------------


def test_malformed_json_syntax_fails(tmp_path):
    with pytest.raises(KnowledgeError, match="not valid JSON"):
        load_clinic_info(_write(tmp_path, "{not json"))


def test_missing_required_field_fails(tmp_path):
    data = _raw()
    del data["phone"]
    with pytest.raises(KnowledgeError, match="failed validation"):
        load_clinic_info(_write(tmp_path, data))


def test_clinic_must_be_marked_fictional(tmp_path):
    data = _raw()
    data["is_fictional"] = False
    with pytest.raises(KnowledgeError, match="is_fictional"):
        load_clinic_info(_write(tmp_path, data))


def test_top_level_must_be_object(tmp_path):
    with pytest.raises(KnowledgeError, match="JSON object"):
        load_clinic_info(_write(tmp_path, "[]"))


def test_missing_file_fails_clearly_with_path(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(KnowledgeError, match="not found") as excinfo:
        load_clinic_info(missing)
    assert str(missing) in str(excinfo.value)


def test_duplicate_service_id_fails(tmp_path):
    data = _raw()
    data["services"].append(dict(data["services"][0]))
    with pytest.raises(KnowledgeError, match="Duplicate service ids"):
        load_clinic_info(_write(tmp_path, data))


def test_inverted_price_range_fails(tmp_path):
    data = _raw()
    data["services"][0]["price_min_inr"] = 9999
    data["services"][0]["price_max_inr"] = 100
    with pytest.raises(KnowledgeError, match="price_min_inr"):
        load_clinic_info(_write(tmp_path, data))


def test_negative_price_fails(tmp_path):
    data = _raw()
    data["services"][0]["price_min_inr"] = -5
    with pytest.raises(KnowledgeError):
        load_clinic_info(_write(tmp_path, data))


def test_invalid_service_id_pattern_fails(tmp_path):
    data = _raw()
    data["services"][0]["id"] = "Consultation"
    with pytest.raises(KnowledgeError):
        load_clinic_info(_write(tmp_path, data))


def test_open_day_without_hours_fails(tmp_path):
    data = _raw()
    data["hours"][0] = {"day": "monday", "closed": False}
    with pytest.raises(KnowledgeError, match="open/close required"):
        load_clinic_info(_write(tmp_path, data))


def test_hours_must_cover_each_weekday_once(tmp_path):
    data = _raw()
    data["hours"][6] = dict(data["hours"][5])  # saturday twice, sunday missing
    with pytest.raises(KnowledgeError, match="each weekday exactly once"):
        load_clinic_info(_write(tmp_path, data))


def test_dentist_name_must_carry_dr_prefix(tmp_path):
    data = _raw()
    data["dentists"][0]["name"] = "Ananya Kulkarni"
    with pytest.raises(KnowledgeError):
        load_clinic_info(_write(tmp_path, data))


def test_blank_alias_fails():
    with pytest.raises(ValidationError):
        Service(
            id="svc-x",
            name="X",
            aliases=["  ...  "],
            description="d",
            price_min_inr=1,
            price_max_inr=2,
            price_note="n",
        )


# ---------------------------------------------------------------------------
# 3. Text normalization (shared by the tool and guardrails)
# ---------------------------------------------------------------------------


def test_strip_punctuation_keeps_devanagari_words_intact():
    # ``[^\w\s]`` would split "दर्द" because \w does not match the virama/matras.
    assert strip_punctuation("दाँत में दर्द, बहुत!") == "दाँत में दर्द बहुत"


def test_normalize_match_text_lowercases_and_strips_punctuation():
    assert normalize_match_text("  R.C.T ka  KITNA lagega?? ") == "r c t ka kitna lagega"


def test_normalize_match_text_rejects_non_strings():
    assert normalize_match_text(None) == ""


# ---------------------------------------------------------------------------
# 4. Digest
# ---------------------------------------------------------------------------


def test_digest_is_deterministic_and_complete():
    kb = get_knowledge_base()
    digest = build_business_digest(kb)
    assert digest == build_business_digest(KnowledgeBase.load())
    for expected in ("SmileCare Dental", "Shivajinagar", "Sunday: closed", "Dr. Rohan Mehta", "Braces", "UPI"):
        assert expected in digest


def test_digest_leaves_prices_to_the_tool():
    digest = build_business_digest(get_knowledge_base())
    assert "₹" not in digest
    assert "clinic_faq_lookup" in digest


def test_get_knowledge_base_is_cached_and_consistent():
    assert get_knowledge_base() is get_knowledge_base()


# ---------------------------------------------------------------------------
# 5. No network, no secrets
# ---------------------------------------------------------------------------


def test_knowledge_loading_performs_no_network_access(monkeypatch):
    def _forbidden_socket(*args, **kwargs):
        raise AssertionError("Knowledge loader attempted to open a network socket")

    monkeypatch.setattr(socket, "socket", _forbidden_socket)
    kb = KnowledgeBase.load()
    assert kb.clinic.services
    build_business_digest(kb)


def test_no_configured_secret_values_leak_into_knowledge_data():
    dumped = KnowledgeBase.load().model_dump_json()
    for var_name in ("WHATSAPP_ACCESS_TOKEN", "WHATSAPP_VERIFY_TOKEN", "GROQ_API_KEY"):
        value = os.environ.get(var_name)
        if value:
            assert value not in dumped, f"{var_name} value leaked into knowledge data"


def test_knowledge_data_contains_no_secret_looking_field_names():
    dumped = json.loads(KnowledgeBase.load().model_dump_json())
    suspicious_markers = ["api_key", "apikey", "access_token", "secret", "password", "private_key"]

    def _walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                for marker in suspicious_markers:
                    assert marker not in key.lower(), f"Suspicious field name at {path}.{key}"
                _walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                _walk(item, f"{path}[{i}]")

    _walk(dumped)


def test_clinic_info_file_contains_no_bearer_or_key_patterns():
    text = DEFAULT_CLINIC_INFO_PATH.read_text(encoding="utf-8").lower()
    assert "bearer " not in text
    assert "-----begin" not in text
    assert "sk-" not in text


def test_clinic_contact_details_are_obviously_fictional():
    clinic = load_clinic_info()
    assert clinic.email.endswith(".example.invalid")
    # Indian mobile numbers never start with 0 after the country code.
    assert clinic.phone.replace(" ", "").startswith("+910")
