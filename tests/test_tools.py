"""Tests for the deterministic product_lookup tool layer (app/tools/).

Self-contained: does not depend on tests/conftest.py's WhatsApp/Groq mocks,
mirroring tests/test_knowledge.py. Uses the real, tracked data/catalog.json
via app.knowledge.get_knowledge_base() (cached, loaded once).
"""

import json
import socket

import pytest

from app.tools import PRODUCT_LOOKUP_TOOL, ToolRegistry, ToolSpec, build_default_registry
from app.tools.catalog_tool import product_lookup
from app.tools.schemas import ProductLookupInput


# ---------------------------------------------------------------------------
# 1. Valid lookup by query
# ---------------------------------------------------------------------------


def test_valid_lookup_by_query_matches_name():
    result = product_lookup({"query": "Yirgacheffe"})
    assert result["status"] == "ok"
    assert result["result_count"] >= 1
    assert any(r["sku"] == "KB-SO-ETH-250" for r in result["results"])


def test_query_matches_tasting_notes_case_insensitively():
    result = product_lookup({"query": "BERGAMOT"})
    assert result["status"] == "ok"
    assert any(r["sku"] == "KB-SO-ETH-250" for r in result["results"])


# ---------------------------------------------------------------------------
# 2. Category filtering
# ---------------------------------------------------------------------------


def test_category_filtering_returns_only_that_category():
    result = product_lookup({"category": "equipment"})
    assert result["status"] == "ok"
    assert all(r["category"] == "equipment" for r in result["results"])


# ---------------------------------------------------------------------------
# 3. Roast level filtering
# ---------------------------------------------------------------------------


def test_roast_level_filtering():
    result = product_lookup({"roast_level": "light"})
    assert result["status"] == "ok"
    assert all(r["roast_level"] == "light" for r in result["results"])


# ---------------------------------------------------------------------------
# 4. Brew method filtering
# ---------------------------------------------------------------------------


def test_brew_method_filtering():
    result = product_lookup({"brew_method": "espresso"})
    assert result["status"] == "ok"
    assert all("espresso" in r["brew_methods"] for r in result["results"])


# ---------------------------------------------------------------------------
# 5. Attribute filtering
# ---------------------------------------------------------------------------


def test_attribute_filtering_organic():
    result = product_lookup({"attributes": ["organic"]})
    assert result["status"] == "ok"
    assert all("organic" in r["attributes"] for r in result["results"])


def test_attribute_filtering_wholesale_available_uses_top_level_field():
    result = product_lookup({"attributes": ["wholesale_available"], "category": "decaf"})
    assert result["status"] == "ok"
    assert all(r["wholesale_available"] is True for r in result["results"])


def test_attribute_filtering_is_and_not_or():
    result = product_lookup({"attributes": ["organic", "single_estate"]})
    assert result["status"] == "ok"
    for r in result["results"]:
        assert "organic" in r["attributes"]
        assert "single_estate" in r["attributes"]


# ---------------------------------------------------------------------------
# 6. Max price filtering
# ---------------------------------------------------------------------------


def test_max_price_filtering():
    result = product_lookup({"category": "single_origin", "max_price_inr": 750})
    assert result["status"] == "ok"
    assert all(r["price_inr"] <= 750 for r in result["results"])


# ---------------------------------------------------------------------------
# 7. Stock filtering
# ---------------------------------------------------------------------------


def test_in_stock_only_default_excludes_nothing_when_all_in_stock():
    result = product_lookup({"category": "blend"})
    assert result["status"] == "ok"
    assert all(r["in_stock"] for r in result["results"])


def test_in_stock_only_false_still_returns_valid_results():
    result = product_lookup({"category": "blend", "in_stock_only": False})
    assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# 8. Combined filters
# ---------------------------------------------------------------------------


def test_combined_filters_category_roast_and_brew_method():
    result = product_lookup(
        {"category": "single_origin", "roast_level": "medium", "brew_method": "filter"}
    )
    assert result["status"] == "ok"
    for r in result["results"]:
        assert r["category"] == "single_origin"
        assert r["roast_level"] == "medium"
        assert "filter" in r["brew_methods"]


# ---------------------------------------------------------------------------
# 9. Result limit
# ---------------------------------------------------------------------------


def test_result_limit_is_respected():
    result = product_lookup({"category": "equipment", "limit": 2})
    assert result["status"] == "ok"
    assert result["result_count"] <= 2
    assert len(result["results"]) <= 2


# ---------------------------------------------------------------------------
# 10. Hard limit of 5
# ---------------------------------------------------------------------------


def test_limit_above_five_is_rejected_by_schema():
    with pytest.raises(Exception):
        ProductLookupInput.model_validate({"category": "equipment", "limit": 6})


def test_handler_never_returns_more_than_five_results():
    # in_stock_only=False + broad max_price should match every product;
    # confirm the server-side cap still holds even at the schema's own max.
    result = product_lookup({"max_price_inr": 100000, "in_stock_only": False, "limit": 5})
    assert result["status"] == "ok"
    assert len(result["results"]) <= 5


# ---------------------------------------------------------------------------
# 11. Empty filter rejection
# ---------------------------------------------------------------------------


def test_empty_filter_set_is_rejected():
    result = product_lookup({"in_stock_only": True, "limit": 3})
    assert result["status"] == "invalid_input"
    assert result["error"]["code"] == "validation_failed"


def test_blank_query_alone_is_treated_as_no_constraint():
    result = product_lookup({"query": "   "})
    assert result["status"] == "invalid_input"


# ---------------------------------------------------------------------------
# 12. Invalid enum
# ---------------------------------------------------------------------------


def test_invalid_category_enum_is_invalid_input():
    result = product_lookup({"category": "beverage"})
    assert result["status"] == "invalid_input"
    assert "category" in result["error"]["fields"]


def test_invalid_attribute_enum_is_invalid_input():
    result = product_lookup({"attributes": ["not_a_real_attribute"]})
    assert result["status"] == "invalid_input"


# ---------------------------------------------------------------------------
# 13. Invalid numeric range
# ---------------------------------------------------------------------------


def test_negative_max_price_is_invalid_input():
    result = product_lookup({"max_price_inr": -1})
    assert result["status"] == "invalid_input"


def test_max_price_above_ceiling_is_invalid_input():
    result = product_lookup({"max_price_inr": 100001})
    assert result["status"] == "invalid_input"


def test_limit_zero_is_invalid_input():
    result = product_lookup({"category": "equipment", "limit": 0})
    assert result["status"] == "invalid_input"


# ---------------------------------------------------------------------------
# 14. Unknown field rejection
# ---------------------------------------------------------------------------


def test_unknown_field_is_rejected():
    result = product_lookup({"category": "equipment", "totally_made_up_field": "x"})
    assert result["status"] == "invalid_input"


# ---------------------------------------------------------------------------
# 15. no_match behavior
# ---------------------------------------------------------------------------


def test_no_match_when_query_matches_nothing():
    result = product_lookup({"query": "xyzzy-nonexistent-product-zzz"})
    assert result["status"] == "no_match"
    assert result["result_count"] == 0
    assert result["results"] == []


def test_no_match_when_price_ceiling_too_low():
    result = product_lookup({"category": "equipment", "max_price_inr": 1})
    assert result["status"] == "no_match"


# ---------------------------------------------------------------------------
# 16. Suggestions come only from the real catalog
# ---------------------------------------------------------------------------


def test_suggestions_are_real_catalog_skus():
    result = product_lookup({"query": "xyzzy-nonexistent-product-zzz", "category": "equipment"})
    assert result["status"] == "no_match"
    assert len(result["suggestions"]) > 0

    from app.knowledge import get_knowledge_base

    real_skus = {p.sku for p in get_knowledge_base().catalog.products}
    for suggestion in result["suggestions"]:
        assert suggestion["sku"] in real_skus


def test_suggestions_respect_structural_filters_when_possible():
    result = product_lookup({"query": "xyzzy-nonexistent-product-zzz", "category": "equipment"})
    assert result["status"] == "no_match"
    for suggestion in result["suggestions"]:
        assert suggestion["category"] == "equipment"


def test_suggestions_capped_at_limit():
    result = product_lookup({"query": "xyzzy-nonexistent-product-zzz", "limit": 2})
    assert result["status"] == "no_match"
    assert len(result["suggestions"]) <= 2


# ---------------------------------------------------------------------------
# 17. Catalog-unavailable behavior
# ---------------------------------------------------------------------------


def test_catalog_unavailable_is_handled_without_raising(monkeypatch):
    def _raise_knowledge_error():
        from app.knowledge import KnowledgeError

        raise KnowledgeError("simulated catalog failure")

    monkeypatch.setattr("app.tools.catalog_tool.get_knowledge_base", _raise_knowledge_error)

    result = product_lookup({"category": "equipment"})
    assert result["status"] == "unavailable"
    assert result["error"]["code"] == "catalog_unavailable"


# ---------------------------------------------------------------------------
# 18. Handler never raises on malformed input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_arguments",
    [
        None,
        "not a dict",
        123,
        [],
        {"limit": "not-a-number"},
        {"category": None, "roast_level": {"nested": "object"}},
        {"attributes": "not-a-list"},
        {"attributes": ["organic"] * 10},
        {"max_price_inr": float("nan")},
        {},
    ],
)
def test_handler_never_raises_on_malformed_input(bad_arguments):
    result = product_lookup(bad_arguments)
    assert result["status"] in {"invalid_input", "unavailable"}


# ---------------------------------------------------------------------------
# 19. Deterministic repeated lookup
# ---------------------------------------------------------------------------


def test_repeated_identical_lookup_is_deterministic():
    args = {"category": "single_origin", "roast_level": "medium"}
    result_a = product_lookup(dict(args))
    result_b = product_lookup(dict(args))
    assert result_a == result_b


def test_repeated_no_match_lookup_is_deterministic():
    args = {"query": "xyzzy-nonexistent-product-zzz"}
    result_a = product_lookup(dict(args))
    result_b = product_lookup(dict(args))
    assert result_a == result_b


# ---------------------------------------------------------------------------
# 20. Registry registration / retrieval / spec serialization
# ---------------------------------------------------------------------------


def test_registry_register_and_get():
    registry = ToolRegistry()
    spec = ToolSpec(
        name="dummy_tool",
        description="A dummy tool for registry tests.",
        input_model=ProductLookupInput,
        handler=lambda args: {"status": "ok"},
    )
    registry.register(spec)
    assert registry.get("dummy_tool") is spec
    assert registry.get("does_not_exist") is None


def test_default_registry_has_product_lookup_registered():
    registry = build_default_registry()
    assert registry.get("product_lookup") is PRODUCT_LOOKUP_TOOL


def test_tool_spec_serializes_to_groq_function_schema():
    schema = PRODUCT_LOOKUP_TOOL.to_groq_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "product_lookup"
    assert "description" in schema["function"]
    assert "parameters" in schema["function"]
    assert schema["function"]["parameters"]["type"] == "object"


def test_registry_list_specs_is_sorted_and_serializable():
    registry = build_default_registry()
    specs = registry.list_specs()
    assert isinstance(specs, list)
    json.dumps(specs)  # must be JSON-serializable
    names = [s["function"]["name"] for s in specs]
    assert names == sorted(names)


def test_registry_execute_by_name_with_dict_arguments():
    registry = build_default_registry()
    result = registry.execute("product_lookup", {"category": "equipment"})
    assert result["status"] == "ok"


def test_registry_execute_by_name_with_json_string_arguments():
    """Mirrors how Groq's native tool-calling API delivers arguments."""
    registry = build_default_registry()
    result = registry.execute("product_lookup", json.dumps({"category": "equipment"}))
    assert result["status"] == "ok"


def test_registry_execute_with_malformed_json_string_is_invalid_input():
    registry = build_default_registry()
    result = registry.execute("product_lookup", "{not valid json")
    assert result["status"] == "invalid_input"


def test_registry_execute_unknown_tool_name_is_invalid_input():
    registry = build_default_registry()
    result = registry.execute("does_not_exist_tool", {})
    assert result["status"] == "invalid_input"
    assert result["error"]["code"] == "unknown_tool"


# ---------------------------------------------------------------------------
# 21. No network access
# ---------------------------------------------------------------------------


def test_product_lookup_performs_no_network_access(monkeypatch):
    def _forbidden_socket(*args, **kwargs):
        raise AssertionError("product_lookup attempted to open a network socket")

    monkeypatch.setattr(socket, "socket", _forbidden_socket)

    result = product_lookup({"category": "single_origin"})
    assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# 22. No secrets in tool definitions
# ---------------------------------------------------------------------------


def test_tool_schema_contains_no_secret_looking_content():
    schema = PRODUCT_LOOKUP_TOOL.to_groq_schema()
    dumped = json.dumps(schema).lower()
    for marker in ("api_key", "apikey", "access_token", "secret", "password", "private_key", "bearer "):
        assert marker not in dumped


def test_no_configured_secret_values_leak_into_tool_output():
    import os

    result = product_lookup({"category": "equipment"})
    dumped = json.dumps(result)
    for var_name in ("WHATSAPP_ACCESS_TOKEN", "WHATSAPP_VERIFY_TOKEN", "GROQ_API_KEY"):
        value = os.environ.get(var_name)
        if value:
            assert value not in dumped
