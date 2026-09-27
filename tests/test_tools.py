"""Tests for the deterministic ``clinic_faq_lookup`` tool and the tool registry."""

import json
import os
import socket

import pytest

from app.tools import CLINIC_FAQ_LOOKUP_TOOL, build_default_registry
from app.tools.clinic_tool import clinic_faq_lookup
from app.tools.registry import ToolRegistry, ToolSpec
from app.tools.schemas import ClinicFaqLookupInput


def _ids(result: dict) -> list:
    return [item["id"] for item in result["results"]]


# ---------------------------------------------------------------------------
# 1. Service price lookups (English, Hinglish, Hindi)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query, expected_id",
    [
        ("How much is a root canal?", "svc-root-canal"),
        ("RCT ka kitna lagega?", "svc-root-canal"),
        ("रूट कैनाल का कितना खर्च होगा", "svc-root-canal"),
        ("braces price", "svc-braces"),
        ("tedhe daant ke liye kya karein", "svc-braces"),
        ("teeth cleaning cost", "svc-cleaning"),
        ("daant saaf karwane ka kharcha", "svc-cleaning"),
        ("सफाई कितने की है", "svc-cleaning"),
        ("whitening charges", "svc-whitening"),
        ("consultation fees", "svc-consultation"),
        ("check-up kitne ka hai", "svc-consultation"),
    ],
)
def test_service_query_returns_that_service_first(query, expected_id):
    result = clinic_faq_lookup({"query": query})
    assert result["status"] == "ok"
    assert _ids(result)[0] == expected_id


def test_service_result_carries_numeric_range_and_display_text():
    result = clinic_faq_lookup({"query": "rct"})
    rct = result["results"][0]
    assert rct["kind"] == "service"
    assert (rct["price_min_inr"], rct["price_max_inr"]) == (3500, 8000)
    assert rct["price_range"] == "₹3,500–₹8,000"
    assert "crown" in rct["price_note"].lower()


def test_generic_price_question_returns_every_service():
    result = clinic_faq_lookup({"query": "what are your prices?"})
    assert result["status"] == "ok"
    assert sorted(_ids(result)) == sorted(
        ["svc-braces", "svc-cleaning", "svc-consultation", "svc-root-canal", "svc-whitening"]
    )


def test_two_services_in_one_query_both_returned():
    result = clinic_faq_lookup({"query": "teeth cleaning aur whitening"})
    assert set(_ids(result)) == {"svc-cleaning", "svc-whitening"}


def test_alias_must_match_whole_tokens():
    # "brace" is an alias, but "embraced" must not match it.
    result = clinic_faq_lookup({"query": "we embraced the idea"})
    assert "svc-braces" not in _ids(result)


# ---------------------------------------------------------------------------
# 2. Timings, location, dentists, FAQs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["Are you open on Sunday?", "clinic kab khula hai", "आपका समय क्या है"])
def test_hours_queries(query):
    result = clinic_faq_lookup({"query": query})
    assert _ids(result)[0] == "hours"
    assert "Sunday: closed" in result["results"][0]["details"]


@pytest.mark.parametrize("query", ["where is the clinic", "clinic kahan hai", "क्लिनिक का पता"])
def test_location_queries(query):
    result = clinic_faq_lookup({"query": query})
    assert "contact" in _ids(result)
    contact = next(r for r in result["results"] if r["id"] == "contact")
    assert "Shivajinagar" in contact["details"]


def test_named_dentist_ranks_first():
    result = clinic_faq_lookup({"query": "Is Dr Mehta available?"})
    assert _ids(result)[0] == "dr-mehta"


def test_generic_doctor_question_lists_all_dentists():
    result = clinic_faq_lookup({"query": "which doctors do you have"})
    assert set(_ids(result)) == {"dr-kulkarni", "dr-mehta", "dr-sheikh"}


@pytest.mark.parametrize(
    "query, faq_id",
    [
        ("do you take insurance", "faq-insurance"),
        ("kya UPI chalega", "faq-payment"),
        ("do you treat kids", "faq-children"),
        ("is there parking", "faq-parking"),
    ],
)
def test_faq_queries(query, faq_id):
    result = clinic_faq_lookup({"query": query})
    assert faq_id in _ids(result)


@pytest.mark.parametrize("topic, expected", [("hours", ["hours"]), ("location", ["contact"])])
def test_topic_without_query(topic, expected):
    result = clinic_faq_lookup({"topic": topic})
    assert result["status"] == "ok"
    assert _ids(result) == expected


def test_topic_services_returns_every_service():
    result = clinic_faq_lookup({"topic": "services"})
    assert len(result["results"]) == 5
    assert all(r["kind"] == "service" for r in result["results"])


# ---------------------------------------------------------------------------
# 3. No match, limits, validation
# ---------------------------------------------------------------------------


def test_no_match_returns_real_services_as_suggestions():
    result = clinic_faq_lookup({"query": "do you do dental implants"})
    assert result["status"] == "no_match"
    assert result["results"] == []
    assert {s["id"] for s in result["suggestions"]} <= {
        "svc-consultation", "svc-cleaning", "svc-root-canal", "svc-braces", "svc-whitening"
    }
    assert result["suggestions"]


def test_limit_is_respected():
    result = clinic_faq_lookup({"query": "what are your prices", "limit": 2})
    assert len(result["results"]) == 2


def test_limit_above_five_is_invalid_input():
    result = clinic_faq_lookup({"query": "rct", "limit": 6})
    assert result["status"] == "invalid_input"
    assert "limit" in result["error"]["fields"]


def test_query_or_topic_is_required():
    assert clinic_faq_lookup({})["status"] == "invalid_input"
    assert clinic_faq_lookup({"query": "   "})["status"] == "invalid_input"


def test_invalid_topic_is_invalid_input():
    assert clinic_faq_lookup({"topic": "pricing"})["status"] == "invalid_input"


def test_unknown_field_is_rejected():
    result = clinic_faq_lookup({"query": "rct", "category": "braces"})
    assert result["status"] == "invalid_input"


def test_overlong_query_is_invalid_input():
    assert clinic_faq_lookup({"query": "a" * 201})["status"] == "invalid_input"


def test_clinic_info_unavailable_is_handled_without_raising(monkeypatch):
    def _raise_knowledge_error():
        from app.knowledge import KnowledgeError

        raise KnowledgeError("simulated failure")

    monkeypatch.setattr("app.tools.clinic_tool.get_knowledge_base", _raise_knowledge_error)
    result = clinic_faq_lookup({"query": "rct"})
    assert result["status"] == "unavailable"
    assert result["error"]["code"] == "clinic_info_unavailable"


@pytest.mark.parametrize(
    "bad_arguments",
    [None, "not a dict", 123, [], {"limit": "not-a-number"}, {"query": {"nested": 1}}, {"topic": None}, {}],
)
def test_handler_never_raises_on_malformed_input(bad_arguments):
    result = clinic_faq_lookup(bad_arguments)
    assert result["status"] in {"invalid_input", "unavailable"}


def test_repeated_identical_lookup_is_deterministic():
    assert clinic_faq_lookup({"query": "rct aur braces"}) == clinic_faq_lookup({"query": "rct aur braces"})
    assert clinic_faq_lookup({"query": "zzz"}) == clinic_faq_lookup({"query": "zzz"})


# ---------------------------------------------------------------------------
# 4. Registry registration / retrieval / spec serialization
# ---------------------------------------------------------------------------


def test_registry_register_and_get():
    registry = ToolRegistry()
    spec = ToolSpec(
        name="dummy_tool",
        description="A dummy tool for registry tests.",
        input_model=ClinicFaqLookupInput,
        handler=lambda args: {"status": "ok"},
    )
    registry.register(spec)
    assert registry.get("dummy_tool") is spec
    assert registry.get("does_not_exist") is None


def test_default_registry_has_clinic_faq_lookup_registered():
    registry = build_default_registry()
    assert registry.get("clinic_faq_lookup") is CLINIC_FAQ_LOOKUP_TOOL
    assert registry.get("product_lookup") is None


def test_tool_spec_serializes_to_groq_function_schema():
    schema = CLINIC_FAQ_LOOKUP_TOOL.to_groq_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "clinic_faq_lookup"
    assert "SmileCare" in schema["function"]["description"]
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert set(params["properties"]) == {"query", "topic", "limit"}


def test_registry_list_specs_is_sorted_and_serializable():
    specs = build_default_registry().list_specs()
    json.dumps(specs)
    names = [s["function"]["name"] for s in specs]
    assert names == sorted(names)


def test_registry_execute_with_dict_and_json_string_arguments():
    registry = build_default_registry()
    assert registry.execute("clinic_faq_lookup", {"query": "rct"})["status"] == "ok"
    assert registry.execute("clinic_faq_lookup", json.dumps({"query": "rct"}))["status"] == "ok"


def test_registry_execute_with_malformed_json_string_is_invalid_input():
    assert build_default_registry().execute("clinic_faq_lookup", "{not valid json")["status"] == "invalid_input"


def test_registry_execute_unknown_tool_name_is_invalid_input():
    result = build_default_registry().execute("does_not_exist_tool", {})
    assert result["status"] == "invalid_input"
    assert result["error"]["code"] == "unknown_tool"


# ---------------------------------------------------------------------------
# 5. No network, no secrets
# ---------------------------------------------------------------------------


def test_lookup_performs_no_network_access(monkeypatch):
    def _forbidden_socket(*args, **kwargs):
        raise AssertionError("clinic_faq_lookup attempted to open a network socket")

    monkeypatch.setattr(socket, "socket", _forbidden_socket)
    assert clinic_faq_lookup({"query": "braces"})["status"] == "ok"


def test_tool_schema_contains_no_secret_looking_content():
    dumped = json.dumps(CLINIC_FAQ_LOOKUP_TOOL.to_groq_schema()).lower()
    for marker in ("api_key", "apikey", "access_token", "secret", "password", "private_key", "bearer "):
        assert marker not in dumped


def test_no_configured_secret_values_leak_into_tool_output():
    dumped = json.dumps(clinic_faq_lookup({"topic": "location"}))
    for var_name in ("WHATSAPP_ACCESS_TOKEN", "WHATSAPP_VERIFY_TOKEN", "GROQ_API_KEY"):
        value = os.environ.get(var_name)
        if value:
            assert value not in dumped
