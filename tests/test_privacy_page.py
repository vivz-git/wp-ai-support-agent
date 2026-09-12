"""Tests for the public Privacy Policy page (docs/privacy.html).

These tests are static, read-only checks over the published HTML/text
content. They do not start the FastAPI app, make any network call, or read
``.env`` -- the privacy page is a standalone static file meant to be hosted
by GitHub Pages, independent of the running service.
"""

import re
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
PRIVACY_PATH = DOCS_DIR / "privacy.html"
INDEX_PATH = DOCS_DIR / "index.html"

# Credential-shaped patterns that must never appear in a public doc page,
# mirroring the source-wide security scan convention used elsewhere in this
# repo (see tests/test_knowledge.py, tests/test_orchestrator.py).
_CREDENTIAL_MARKERS = (
    "gsk_",
    "EAA",
    "Bearer ",
    "Authorization:",
    "PRIVATE KEY",
    "password=",
    "access_token=",
)

_LOCAL_OR_PRIVATE_URL_MARKERS = (
    "localhost",
    "127.0.0.1",
    "ngrok",
    "file://",
    "0.0.0.0",
)

_EXTERNAL_ASSET_MARKERS = (
    "<script",
    "googletagmanager",
    "google-analytics",
    "gtag(",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "cdn.",
    "unpkg.com",
    "jsdelivr.net",
    "cloudflare.com",
    "cookie",
)


@pytest.fixture(scope="module")
def privacy_html() -> str:
    return PRIVACY_PATH.read_text(encoding="utf-8")


def test_privacy_html_exists():
    """1. privacy.html exists in docs/."""
    assert PRIVACY_PATH.is_file(), "docs/privacy.html must exist"


def test_privacy_html_has_valid_basic_structure(privacy_html: str):
    """2. Valid HTML structure / basic required tags exist."""
    lowered = privacy_html.lower()
    assert lowered.strip().startswith("<!doctype html>")
    assert "<html" in lowered and "</html>" in lowered
    assert "<head>" in lowered and "</head>" in lowered
    assert "<body>" in lowered and "</body>" in lowered
    assert re.search(r"<meta[^>]+charset=", lowered) is not None
    assert re.search(r'<meta[^>]+name=["\']viewport["\']', lowered) is not None
    assert "<title>" in lowered and "</title>" in lowered


def test_privacy_policy_heading_exists(privacy_html: str):
    """3. 'Privacy Policy' heading exists."""
    assert re.search(r"<h1[^>]*>\s*Privacy Policy\s*</h1>", privacy_html) is not None


def test_effective_date_present(privacy_html: str):
    """4. Effective date exists."""
    assert "Effective date" in privacy_html
    # A real calendar date follows the label (month name + year).
    assert re.search(r"Effective date:\s*[A-Z][a-z]+ \d{1,2}, \d{4}", privacy_html)


def test_contact_information_present(privacy_html: str):
    """5. Contact information exists."""
    assert "Contact" in privacy_html
    assert re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", privacy_html), "an email-shaped contact must be present"


def test_fictional_demo_disclosure_present(privacy_html: str):
    """6. Fictional/demo disclosure exists."""
    lowered = privacy_html.lower()
    assert "fictional" in lowered
    assert "demo" in lowered
    assert "portfolio" in lowered
    assert "not a real company" in lowered or "not a real commercial service" in lowered


def test_no_credential_shaped_values(privacy_html: str):
    """7. No credential-shaped values appear."""
    for marker in _CREDENTIAL_MARKERS:
        assert marker not in privacy_html, f"credential-shaped marker {marker!r} must not appear"


def test_no_localhost_or_ngrok_urls(privacy_html: str):
    """8. No localhost/ngrok URLs appear."""
    lowered = privacy_html.lower()
    for marker in _LOCAL_OR_PRIVATE_URL_MARKERS:
        assert marker not in lowered, f"private/local URL marker {marker!r} must not appear"


def test_no_external_tracking_scripts(privacy_html: str):
    """9. No external tracking scripts exist."""
    lowered = privacy_html.lower()
    assert "<script" not in lowered
    for marker in ("google-analytics", "googletagmanager", "gtag(", "mixpanel", "segment.", "hotjar", "facebook.net"):
        assert marker not in lowered


def test_no_external_font_or_cdn_dependency(privacy_html: str):
    """10. No external font/CDN dependencies exist."""
    lowered = privacy_html.lower()
    assert "<link" not in lowered, "no external stylesheet/font <link> tags expected"
    for marker in ("fonts.googleapis.com", "fonts.gstatic.com", "cdn.", "unpkg.com", "jsdelivr.net"):
        assert marker not in lowered


def test_no_real_secrets(privacy_html: str):
    """11. No real secrets exist (only the documented placeholder domain)."""
    for marker in _CREDENTIAL_MARKERS:
        assert marker not in privacy_html
    # The only email domain referenced must be the fictional .invalid domain
    # already used by data/business.json, never a live-looking address.
    emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", privacy_html)
    assert emails, "expected at least one contact email"
    for email in emails:
        assert email.endswith(".invalid"), f"unexpected non-placeholder email domain: {email}"


def test_privacy_page_claims_match_repository_evidence(privacy_html: str):
    """12. Privacy page references only claims supported by the repository."""
    lowered = privacy_html.lower()

    # Supported claims (grounded in app/agent/state.py, app/agent/handoff.py,
    # app/agent/store.py, app/config.py, app/llm/groq_provider.py).
    assert "whatsapp" in lowered
    assert "groq" in lowered
    assert "in-memory" in lowered or "in memory" in lowered
    assert "meta" in lowered

    # Must NOT claim payment processing, since the WhatsApp assistant never
    # collects payment/card details (data/business.json: payment_methods_note).
    assert "we accept payments" not in lowered
    assert "credit card number" not in lowered
    assert "we process payments" not in lowered

    # Must NOT make unsupported compliance/legal certification claims.
    for forbidden_claim in (
        "gdpr compliant",
        "soc 2 certified",
        "soc2 certified",
        "fully encrypted",
        "zero data retention",
        "hipaa compliant",
        "ccpa compliant",
        "iso 27001 certified",
    ):
        assert forbidden_claim not in lowered, f"unsupported compliance claim found: {forbidden_claim!r}"

    # Must explicitly say personal information is not sold.
    assert "sold" in lowered or "sale of personal information" in lowered


def test_index_html_exists_and_links_to_privacy():
    """docs/index.html exists and links to privacy.html (landing page check)."""
    assert INDEX_PATH.is_file(), "docs/index.html must exist"
    index_html = INDEX_PATH.read_text(encoding="utf-8")
    assert 'href="privacy.html"' in index_html
    assert "<script" not in index_html.lower()


def test_docs_html_files_contain_no_credential_markers():
    """Extra safety net: scan every docs/*.html file for credential markers."""
    for html_file in DOCS_DIR.glob("*.html"):
        content = html_file.read_text(encoding="utf-8")
        for marker in _CREDENTIAL_MARKERS:
            assert marker not in content, f"{html_file.name} contains credential marker {marker!r}"
