"""Deterministic ``product_lookup`` tool handler.

Matching semantics
------------------
A product must satisfy every filter the caller supplied (logical AND across
filter *kinds*); within a kind, semantics are:

- ``query``: case-insensitive substring match. The normalized query (lower-
  cased, whitespace-stripped) must appear as a substring of a per-product
  haystack built by joining: name, origin, process, category, roast level,
  and all tasting notes (lower-cased, space-joined). No fuzzy matching, no
  tokenization, no stemming — this is intentionally simple and fully
  deterministic.
- ``category`` / ``roast_level``: exact equality against the product's
  single value.
- ``brew_method``: the product's ``brew_methods`` list must contain it.
- ``attributes``: ALL requested attributes must be present (AND, not OR).
  ``wholesale_available`` is special-cased to check the product's top-level
  ``wholesale_available`` boolean (it is not part of the catalog's
  per-product ``attributes`` list); every other attribute is checked against
  membership in that list.
- ``max_price_inr``: the product's price must be less than or equal to it.
- ``in_stock_only``: when true (the default), out-of-stock products are
  excluded entirely.

Results are sorted by SKU for determinism, then truncated to ``limit``
(already server-side-capped at 5 by ``ProductLookupInput``).

Suggestions (``no_match`` only)
--------------------------------
Suggestions are never invented: they are always real catalog products,
chosen deterministically by relaxing the *free-text and numeric* filters
(``query``, ``attributes``, ``max_price_inr``) while keeping the *structural*
filters (``category``, ``roast_level``, ``brew_method``, ``in_stock_only``).
If that relaxed search still yields nothing, the fallback is the first
``limit`` in-stock-respecting products in the catalog, sorted by SKU. This
keeps suggestions relevant when possible and always falls back to something
real rather than nothing.
"""

from typing import List, Optional

from app.knowledge import KnowledgeError, Product, get_knowledge_base
from app.tools.schemas import (
    ProductLookupErrorInfo,
    ProductLookupInput,
    ProductLookupOutput,
    ProductLookupStatus,
    ProductResult,
)

try:
    from pydantic import ValidationError
except ImportError:  # pragma: no cover - pydantic is a hard dependency
    raise


def _normalize_query(query: Optional[str]) -> Optional[str]:
    if query is None:
        return None
    return " ".join(query.lower().split())


def _product_haystack(product: Product) -> str:
    parts = [product.name, product.category.value]
    if product.origin:
        parts.append(product.origin)
    if product.process:
        parts.append(product.process)
    if product.roast_level:
        parts.append(product.roast_level.value)
    parts.extend(product.tasting_notes)
    return " ".join(parts).lower()


def _has_attribute(product: Product, attribute: str) -> bool:
    if attribute == "wholesale_available":
        return product.wholesale_available
    return attribute in {a.value for a in product.attributes}


def _matches_structural_filters(product: Product, params: ProductLookupInput) -> bool:
    """Category / roast_level / brew_method / in_stock_only only."""
    if params.in_stock_only and not product.in_stock:
        return False
    if params.category is not None and product.category != params.category:
        return False
    if params.roast_level is not None and product.roast_level != params.roast_level:
        return False
    if params.brew_method is not None and params.brew_method not in product.brew_methods:
        return False
    return True


def _matches_all_filters(product: Product, params: ProductLookupInput, normalized_query: Optional[str]) -> bool:
    if not _matches_structural_filters(product, params):
        return False
    if normalized_query and normalized_query not in _product_haystack(product):
        return False
    if params.attributes:
        if not all(_has_attribute(product, attr.value) for attr in params.attributes):
            return False
    if params.max_price_inr is not None and product.price_inr > params.max_price_inr:
        return False
    return True


def _to_result(product: Product) -> ProductResult:
    return ProductResult(
        sku=product.sku,
        name=product.name,
        category=product.category,
        roast_level=product.roast_level,
        brew_methods=list(product.brew_methods),
        price_inr=product.price_inr,
        size_g=product.size_g,
        in_stock=product.in_stock,
        stock_level=product.stock_level,
        attributes=[a.value for a in product.attributes],
        subscription_available=product.subscription_available,
        wholesale_available=product.wholesale_available,
        product_url=product.product_url,
    )


def _build_suggestions(products: List[Product], params: ProductLookupInput, limit: int) -> List[ProductResult]:
    relaxed = [p for p in products if _matches_structural_filters(p, params)]
    if not relaxed:
        relaxed = [p for p in products if (not params.in_stock_only or p.in_stock)]
    relaxed_sorted = sorted(relaxed, key=lambda p: p.sku)
    return [_to_result(p) for p in relaxed_sorted[:limit]]


def _invalid_input(code: str, message: str, fields: Optional[List[str]] = None) -> dict:
    return ProductLookupOutput(
        status=ProductLookupStatus.INVALID_INPUT,
        result_count=0,
        error=ProductLookupErrorInfo(code=code, message=message, fields=fields or []),
    ).model_dump(mode="json")


def product_lookup(arguments: dict) -> dict:
    """Handle a ``product_lookup`` tool call.

    Args:
        arguments: Raw, untrusted arguments (already decoded from JSON if
            they arrived as a JSON string — see ``app.tools.registry``).

    Returns:
        A JSON-serializable dict matching ``ProductLookupOutput``. Never
        raises: every failure mode (bad input, unavailable catalog, an
        unexpected internal error) is translated into a structured result
        with an appropriate ``status``.
    """
    try:
        if not isinstance(arguments, dict):
            return _invalid_input(
                "invalid_arguments_type",
                "Tool arguments must be a JSON object.",
            )

        try:
            params = ProductLookupInput.model_validate(arguments)
        except ValidationError as exc:
            fields = sorted({".".join(str(part) for part in err["loc"]) for err in exc.errors()})
            return _invalid_input(
                "validation_failed",
                "One or more product_lookup arguments were invalid.",
                fields=fields,
            )

        try:
            knowledge = get_knowledge_base()
        except KnowledgeError:
            return ProductLookupOutput(
                status=ProductLookupStatus.UNAVAILABLE,
                result_count=0,
                error=ProductLookupErrorInfo(
                    code="catalog_unavailable",
                    message="The product catalog is temporarily unavailable.",
                ),
            ).model_dump(mode="json")

        normalized_query = _normalize_query(params.query)
        products = knowledge.catalog.products

        matches = sorted(
            (p for p in products if _matches_all_filters(p, params, normalized_query)),
            key=lambda p: p.sku,
        )

        if matches:
            limited = matches[: params.limit]
            return ProductLookupOutput(
                status=ProductLookupStatus.OK,
                catalog_version=knowledge.catalog.catalog_version,
                normalized_query=normalized_query,
                result_count=len(limited),
                results=[_to_result(p) for p in limited],
            ).model_dump(mode="json")

        suggestions = _build_suggestions(products, params, params.limit)
        return ProductLookupOutput(
            status=ProductLookupStatus.NO_MATCH,
            catalog_version=knowledge.catalog.catalog_version,
            normalized_query=normalized_query,
            result_count=0,
            suggestions=suggestions,
        ).model_dump(mode="json")

    except Exception:
        # Final safety net: the contract promises the handler never raises.
        return _invalid_input(
            "internal_error",
            "product_lookup could not process the request.",
        )
