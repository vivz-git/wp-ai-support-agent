"""Tests for the business knowledge layer (app/knowledge.py).

These tests are self-contained and do not depend on tests/conftest.py's
WhatsApp/Groq mocks -- the knowledge layer has no dependency on those. Where
we need "no secrets leaked" coverage, we read the same mock credential
values conftest.py sets in the environment, so the check is meaningful even
though this module doesn't use the `client`/`mock_*` fixtures.
"""

import json
import os
import socket

import pytest
from pydantic import ValidationError

from app.knowledge import (
    DEFAULT_BUSINESS_PROFILE_PATH,
    DEFAULT_CATALOG_PATH,
    BusinessProfile,
    Catalog,
    KnowledgeBase,
    KnowledgeError,
    build_business_digest,
    build_catalog_digest,
    get_knowledge_base,
    load_business_profile,
    load_catalog,
)


# ---------------------------------------------------------------------------
# 1. Valid business profile loads
# ---------------------------------------------------------------------------


def test_valid_business_profile_loads():
    business = load_business_profile()
    assert isinstance(business, BusinessProfile)
    assert business.is_fictional is True
    assert business.name == "Kettle & Bloom Coffee Roasters"
    assert business.location.city == "Bengaluru"
    assert len(business.faqs) >= 1
    assert business.wholesale_info.pricing_disclosed_by_agent is False


def test_default_business_profile_path_resolves_to_data_dir():
    assert DEFAULT_BUSINESS_PROFILE_PATH.name == "business.json"
    assert DEFAULT_BUSINESS_PROFILE_PATH.exists()


# ---------------------------------------------------------------------------
# 2. Valid catalog loads
# ---------------------------------------------------------------------------


def test_valid_catalog_loads():
    catalog = load_catalog()
    assert isinstance(catalog, Catalog)
    assert catalog.catalog_version
    assert len(catalog.products) >= 1


def test_default_catalog_path_resolves_to_data_dir():
    assert DEFAULT_CATALOG_PATH.name == "catalog.json"
    assert DEFAULT_CATALOG_PATH.exists()


# ---------------------------------------------------------------------------
# 3. Malformed business JSON fails
# ---------------------------------------------------------------------------


def test_malformed_business_json_syntax_fails(tmp_path):
    bad_file = tmp_path / "business.json"
    bad_file.write_text("{ this is not valid json", encoding="utf-8")

    with pytest.raises(KnowledgeError, match="not valid JSON"):
        load_business_profile(bad_file)


def test_business_json_missing_required_field_fails(tmp_path):
    data = json.loads(DEFAULT_BUSINESS_PROFILE_PATH.read_text(encoding="utf-8"))
    del data["name"]
    bad_file = tmp_path / "business.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(KnowledgeError, match="failed validation"):
        load_business_profile(bad_file)


def test_business_json_must_be_marked_fictional(tmp_path):
    data = json.loads(DEFAULT_BUSINESS_PROFILE_PATH.read_text(encoding="utf-8"))
    data["is_fictional"] = False
    bad_file = tmp_path / "business.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(KnowledgeError, match="failed validation"):
        load_business_profile(bad_file)


def test_business_json_top_level_must_be_object(tmp_path):
    bad_file = tmp_path / "business.json"
    bad_file.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(KnowledgeError, match="JSON object"):
        load_business_profile(bad_file)


def test_agent_must_never_be_allowed_to_disclose_wholesale_pricing(tmp_path):
    """Hard business-rule guard: this must fail validation even if someone
    tries to flip the flag in the data file."""
    data = json.loads(DEFAULT_BUSINESS_PROFILE_PATH.read_text(encoding="utf-8"))
    data["wholesale_info"]["pricing_disclosed_by_agent"] = True
    bad_file = tmp_path / "business.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(KnowledgeError, match="failed validation"):
        load_business_profile(bad_file)


# ---------------------------------------------------------------------------
# 4. Malformed catalog JSON fails
# ---------------------------------------------------------------------------


def test_malformed_catalog_json_syntax_fails(tmp_path):
    bad_file = tmp_path / "catalog.json"
    bad_file.write_text("[not valid json", encoding="utf-8")

    with pytest.raises(KnowledgeError, match="not valid JSON"):
        load_catalog(bad_file)


def test_catalog_json_missing_required_field_fails(tmp_path):
    data = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
    del data["products"][0]["price_inr"]
    bad_file = tmp_path / "catalog.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(KnowledgeError, match="failed validation"):
        load_catalog(bad_file)


def test_catalog_json_duplicate_sku_fails(tmp_path):
    data = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
    duplicate = dict(data["products"][0])
    data["products"].append(duplicate)
    bad_file = tmp_path / "catalog.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(KnowledgeError, match="Duplicate SKUs"):
        load_catalog(bad_file)


# ---------------------------------------------------------------------------
# 5. Missing file fails clearly
# ---------------------------------------------------------------------------


def test_missing_business_file_fails_clearly(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(KnowledgeError, match="not found"):
        load_business_profile(missing)


def test_missing_catalog_file_fails_clearly(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(KnowledgeError, match="not found"):
        load_catalog(missing)


def test_missing_file_error_message_includes_path(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(KnowledgeError) as exc_info:
        load_catalog(missing)
    assert str(missing) in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. Catalog entries validate expected fields
# ---------------------------------------------------------------------------


def test_catalog_has_fourteen_products():
    catalog = load_catalog()
    assert len(catalog.products) == 14


def test_catalog_skus_are_unique():
    catalog = load_catalog()
    skus = [p.sku for p in catalog.products]
    assert len(skus) == len(set(skus))


def test_catalog_covers_all_expected_categories():
    catalog = load_catalog()
    categories = {p.category.value for p in catalog.products}
    assert categories == {
        "single_origin",
        "blend",
        "decaf",
        "equipment",
        "subscription",
        "accessory",
    }


def test_catalog_every_product_has_valid_price_and_dispatch():
    catalog = load_catalog()
    for product in catalog.products:
        assert product.price_inr > 0
        assert 1 <= product.dispatch_days <= 30


def test_catalog_roasted_coffee_products_have_roast_and_brew_methods():
    catalog = load_catalog()
    roasted = {"single_origin", "blend", "decaf"}
    for product in catalog.products:
        if product.category.value in roasted:
            assert product.roast_level is not None, product.sku
            assert len(product.brew_methods) > 0, product.sku


def test_catalog_known_sku_lookup_by_hand():
    catalog = load_catalog()
    by_sku = {p.sku: p for p in catalog.products}
    yirgacheffe = by_sku["KB-SO-ETH-250"]
    assert yirgacheffe.name == "Yirgacheffe Light"
    assert yirgacheffe.category.value == "single_origin"
    assert yirgacheffe.price_inr == 780
    assert yirgacheffe.wholesale_available is True


def test_catalog_products_are_immutable():
    catalog = load_catalog()
    product = catalog.products[0]
    with pytest.raises(ValidationError):
        product.price_inr = 1


# ---------------------------------------------------------------------------
# 7. Invalid enum/value fails validation
# ---------------------------------------------------------------------------


def test_invalid_category_enum_fails(tmp_path):
    data = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
    data["products"][0]["category"] = "beverage"  # not a real category
    bad_file = tmp_path / "catalog.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(KnowledgeError, match="failed validation"):
        load_catalog(bad_file)


def test_invalid_roast_level_enum_fails(tmp_path):
    data = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
    data["products"][0]["roast_level"] = "extra_dark"  # not a real roast level
    bad_file = tmp_path / "catalog.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(KnowledgeError, match="failed validation"):
        load_catalog(bad_file)


def test_negative_price_fails(tmp_path):
    data = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
    data["products"][0]["price_inr"] = -100
    bad_file = tmp_path / "catalog.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(KnowledgeError, match="failed validation"):
        load_catalog(bad_file)


def test_in_stock_stock_level_mismatch_fails(tmp_path):
    data = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
    data["products"][0]["in_stock"] = False
    data["products"][0]["stock_level"] = "high"  # inconsistent
    bad_file = tmp_path / "catalog.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(KnowledgeError, match="failed validation"):
        load_catalog(bad_file)


def test_decaf_category_without_decaf_attribute_fails(tmp_path):
    data = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
    decaf_product = next(p for p in data["products"] if p["category"] == "decaf")
    decaf_product["attributes"] = []  # strips required "decaf" attribute
    bad_file = tmp_path / "catalog.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(KnowledgeError, match="failed validation"):
        load_catalog(bad_file)


def test_invalid_sku_pattern_fails(tmp_path):
    data = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
    data["products"][0]["sku"] = "not-a-valid-sku"
    bad_file = tmp_path / "catalog.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(KnowledgeError, match="failed validation"):
        load_catalog(bad_file)


# ---------------------------------------------------------------------------
# 8. Digest generation is deterministic
# ---------------------------------------------------------------------------


def test_business_digest_is_deterministic():
    kb = KnowledgeBase.load()
    digest1 = build_business_digest(kb)
    digest2 = build_business_digest(kb)
    assert digest1 == digest2
    assert "Kettle & Bloom" in digest1


def test_catalog_digest_is_deterministic():
    kb = KnowledgeBase.load()
    digest1 = build_catalog_digest(kb)
    digest2 = build_catalog_digest(kb)
    assert digest1 == digest2


def test_catalog_digest_is_sorted_by_sku():
    kb = KnowledgeBase.load()
    digest = build_catalog_digest(kb)
    sku_lines = [line for line in digest.splitlines() if line.strip().startswith("- KB-")]
    skus_in_digest = [line.split("|")[0].strip("- ").strip() for line in sku_lines]
    assert skus_in_digest == sorted(skus_in_digest)


def test_catalog_digest_respects_max_items():
    kb = KnowledgeBase.load()
    digest = build_catalog_digest(kb, max_items=2)
    sku_lines = [line for line in digest.splitlines() if line.strip().startswith("- KB-")]
    assert len(sku_lines) == 2


def test_two_separately_loaded_knowledge_bases_produce_identical_digests():
    kb_a = KnowledgeBase.load()
    kb_b = KnowledgeBase.load()
    assert build_business_digest(kb_a) == build_business_digest(kb_b)
    assert build_catalog_digest(kb_a) == build_catalog_digest(kb_b)


def test_get_knowledge_base_is_cached_and_consistent():
    kb_a = get_knowledge_base()
    kb_b = get_knowledge_base()
    assert kb_a is kb_b  # lru_cache identity
    assert kb_a.catalog.catalog_version == kb_b.catalog.catalog_version


# ---------------------------------------------------------------------------
# 9. Knowledge loader performs no network access
# ---------------------------------------------------------------------------


def test_knowledge_loading_performs_no_network_access(monkeypatch):
    def _forbidden_socket(*args, **kwargs):
        raise AssertionError("Knowledge loader attempted to open a network socket")

    monkeypatch.setattr(socket, "socket", _forbidden_socket)

    # Should complete without touching the network at all.
    kb = KnowledgeBase.load()
    assert kb.catalog.products
    build_business_digest(kb)
    build_catalog_digest(kb)


# ---------------------------------------------------------------------------
# 10. No secrets present in loaded knowledge content
# ---------------------------------------------------------------------------


def test_no_configured_secret_values_leak_into_knowledge_data():
    kb = KnowledgeBase.load()
    dumped = kb.model_dump_json()

    secret_env_vars = [
        "WHATSAPP_ACCESS_TOKEN",
        "WHATSAPP_VERIFY_TOKEN",
        "GROQ_API_KEY",
    ]
    for var_name in secret_env_vars:
        value = os.environ.get(var_name)
        if value:
            assert value not in dumped, f"{var_name} value leaked into knowledge data"


def test_knowledge_data_contains_no_secret_looking_field_names():
    kb = KnowledgeBase.load()
    dumped = json.loads(kb.model_dump_json())
    suspicious_markers = ["api_key", "apikey", "access_token", "secret", "password", "private_key"]

    def _walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                lowered_key = key.lower()
                for marker in suspicious_markers:
                    assert marker not in lowered_key, f"Suspicious field name at {path}.{key}"
                _walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                _walk(item, f"{path}[{i}]")

    _walk(dumped)


def test_business_json_and_catalog_json_files_contain_no_bearer_or_key_patterns():
    """Static grep-style check directly on the tracked data files."""
    for path in (DEFAULT_BUSINESS_PROFILE_PATH, DEFAULT_CATALOG_PATH):
        text = path.read_text(encoding="utf-8").lower()
        assert "bearer " not in text
        assert "-----begin" not in text  # PEM key markers
        assert "sk-" not in text  # common API key prefix shape
