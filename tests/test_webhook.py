from fastapi.testclient import TestClient

from tests.conftest import MockLLMProvider, MockWhatsAppClient


def test_webhook_verification_success(client: TestClient):
    """GET /webhook/whatsapp should return 200 with hub.challenge when credentials match."""
    challenge = "1158201444"
    response = client.get(
        "/webhook/whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "test_webhook_secret_token",
            "hub.challenge": challenge,
        },
    )
    assert response.status_code == 200
    assert response.text == challenge


def test_webhook_verification_failure_invalid_token(client: TestClient):
    """GET /webhook/whatsapp should return 403 when verify_token is invalid."""
    response = client.get(
        "/webhook/whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong_token_value",
            "hub.challenge": "1158201444",
        },
    )
    assert response.status_code == 403
    assert "Forbidden" in response.text


def test_webhook_verification_failure_invalid_mode(client: TestClient):
    """GET /webhook/whatsapp should return 403 when mode is not 'subscribe'."""
    response = client.get(
        "/webhook/whatsapp",
        params={
            "hub.mode": "unsubscribe",
            "hub.verify_token": "test_webhook_secret_token",
            "hub.challenge": "1158201444",
        },
    )
    assert response.status_code == 403


def test_webhook_verification_failure_missing_parameters(client: TestClient):
    """GET /webhook/whatsapp should return 403 when query parameters are missing."""
    response = client.get("/webhook/whatsapp")
    assert response.status_code == 403


def test_valid_incoming_text_payload(client: TestClient, valid_text_payload, mock_wa, mock_llm):
    """POST /webhook/whatsapp should parse text message, query LLM, and send WhatsApp reply."""
    response = client.post("/webhook/whatsapp", json=valid_text_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "message_id" in data

    # Verify WhatsApp client sent the reply
    assert len(mock_wa.sent_messages) == 1
    sent = mock_wa.sent_messages[0]
    assert sent["to"] == "919876543210"
    assert sent["body"] == mock_llm.response_text

    # Verify LLM was called with the user's message
    assert len(mock_llm.calls) == 1
    history = mock_llm.calls[0]
    assert len(history) == 1
    assert history[0].role == "user"
    assert history[0].content == "Hello, I want to inquire about pricing."


def test_malformed_payload_missing_keys(client: TestClient):
    """POST /webhook/whatsapp should return 400 when payload structure is invalid."""
    malformed_payloads = [
        {},
        {"invalid_key": "some_value"},
        {"object": "whatsapp_business_account"},  # missing entry
        {"entry": []},  # missing object
    ]
    for bad_payload in malformed_payloads:
        response = client.post("/webhook/whatsapp", json=bad_payload)
        assert response.status_code == 400
        assert "Malformed" in response.json()["detail"] or "missing" in response.json()["detail"]


def test_malformed_payload_invalid_json(client: TestClient):
    """POST /webhook/whatsapp should return 400 for non-JSON content."""
    response = client.post(
        "/webhook/whatsapp",
        content="this is not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert "Invalid JSON" in response.json()["detail"]


def test_unsupported_event_status_update(client: TestClient, status_update_payload, mock_wa, mock_llm):
    """POST /webhook/whatsapp should return 200 and ignore message delivery status events."""
    response = client.post("/webhook/whatsapp", json=status_update_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ignored"
    assert data["event_type"] == "status_update"

    # Ensure no outgoing message was dispatched and LLM was not invoked
    assert len(mock_wa.sent_messages) == 0
    assert len(mock_llm.calls) == 0


def test_unsupported_event_media_message(client: TestClient, unsupported_media_payload, mock_wa, mock_llm):
    """POST /webhook/whatsapp should return 200 and ignore non-text incoming messages."""
    response = client.post("/webhook/whatsapp", json=unsupported_media_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ignored"
    assert "unsupported_media_image" in data["event_type"]

    # Ensure no outgoing message was dispatched and LLM was not invoked
    assert len(mock_wa.sent_messages) == 0
    assert len(mock_llm.calls) == 0


def test_health_check(client: TestClient):
    """GET /health should return 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_meta_sample_test_event_send_failure_handled_safely(
    client: TestClient, meta_sample_test_payload, mock_llm
):
    """Meta's webhook dashboard 'Test' event has a synthetic sender that is not a
    verified test recipient. Sending to it must fail gracefully with a 200 response
    (not a 500 that triggers Meta retries), and the LLM should still run so the rest
    of the pipeline (parse -> memory -> Groq) is proven to work end-to-end."""
    from app.main import app, get_whatsapp_client

    failing_wa_client = MockWhatsAppClient(raise_error_code=131030)
    app.dependency_overrides[get_whatsapp_client] = lambda: failing_wa_client

    response = client.post("/webhook/whatsapp", json=meta_sample_test_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "send_failed"
    assert "message_id" in data

    # The LLM was still invoked (proves parse -> memory -> Groq worked)
    assert len(mock_llm.calls) == 1
    # No message was actually recorded as successfully sent
    assert len(failing_wa_client.sent_messages) == 0


def test_duplicate_webhook_delivery_does_not_send_twice(
    client: TestClient, valid_text_payload, mock_wa, mock_llm
):
    """A retried Meta webhook delivery (same message ID) must not trigger a second
    LLM call or a second outbound WhatsApp reply."""
    first_response = client.post("/webhook/whatsapp", json=valid_text_payload)
    assert first_response.status_code == 200
    assert first_response.json()["status"] == "ok"

    # Meta redelivers the identical event (e.g. because the first response was slow
    # or a prior non-200 was returned)
    second_response = client.post("/webhook/whatsapp", json=valid_text_payload)
    assert second_response.status_code == 200
    assert second_response.json()["status"] == "duplicate_ignored"

    # LLM and WhatsApp send should each have been invoked exactly once, not twice
    assert len(mock_llm.calls) == 1
    assert len(mock_wa.sent_messages) == 1


def test_llm_failure_returns_200_without_sending(
    client: TestClient, valid_text_payload, mock_wa
):
    """If the LLM provider fails, the webhook should still acknowledge with 200
    (so Meta doesn't retry and re-run the LLM again) and must not attempt to send
    a WhatsApp reply."""
    from app.main import app, get_llm_provider

    failing_llm = MockLLMProvider(raise_error=True)
    app.dependency_overrides[get_llm_provider] = lambda: failing_llm

    response = client.post("/webhook/whatsapp", json=valid_text_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "llm_error"

    assert len(mock_wa.sent_messages) == 0
