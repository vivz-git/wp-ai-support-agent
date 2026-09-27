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


def test_valid_incoming_text_payload(client: TestClient, valid_text_payload, mock_wa, mock_llm, test_drafts):
    """POST /webhook/whatsapp should parse the text message, query the LLM, and queue
    the reply as a pending draft for staff approval — never send it directly."""
    response = client.post("/webhook/whatsapp", json=valid_text_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending_approval"
    assert data["is_urgent"] is False
    assert "message_id" in data

    # Nothing reaches the patient until a staff member approves it
    assert mock_wa.sent_messages == []
    pending = test_drafts.list_pending()
    assert [d.id for d in pending] == [data["draft_id"]]
    draft = pending[0]
    assert draft.sender == "919876543210"
    assert draft.patient_message == "Hello, I want to inquire about pricing."
    assert draft.draft_text == mock_llm.response_text
    assert draft.is_urgent is False

    # Verify LLM was called with the user's message
    assert len(mock_llm.calls) >= 1
    all_contents = [msg.content for call in mock_llm.calls for msg in call]
    assert any("Hello, I want to inquire about pricing." in content for content in all_contents)


def test_approved_draft_is_sent_via_whatsapp(client: TestClient, valid_text_payload, mock_wa, mock_llm):
    """Approving the queued draft on /staff is what delivers it."""
    draft_id = client.post("/webhook/whatsapp", json=valid_text_payload).json()["draft_id"]

    response = client.post(
        f"/staff/drafts/{draft_id}/approve", data={"text": mock_llm.response_text}, follow_redirects=False
    )

    assert response.status_code == 303
    assert mock_wa.sent_messages == [{"to": "919876543210", "body": mock_llm.response_text}]


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
    client: TestClient, meta_sample_test_payload, mock_llm, test_drafts
):
    """Meta's webhook dashboard 'Test' event has a synthetic sender that is not a
    verified test recipient. The webhook still runs the agent and queues a draft
    (proving parse -> agent -> queue works end-to-end); approving that draft fails
    gracefully and puts it back in the queue instead of erroring."""
    from app.main import app, get_whatsapp_client

    failing_wa_client = MockWhatsAppClient(raise_error_code=131030)
    app.dependency_overrides[get_whatsapp_client] = lambda: failing_wa_client

    response = client.post("/webhook/whatsapp", json=meta_sample_test_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending_approval"
    assert len(mock_llm.calls) >= 1

    approve = client.post(
        f"/staff/drafts/{data['draft_id']}/approve", data={"text": "Hello"}, follow_redirects=False
    )
    assert approve.status_code == 303
    assert "notice=recipient_not_allowed" in approve.headers["location"]
    assert failing_wa_client.sent_messages == []
    assert [d.id for d in test_drafts.list_pending()] == [data["draft_id"]]


def test_duplicate_webhook_delivery_does_not_queue_twice(
    client: TestClient, valid_text_payload, mock_wa, mock_llm, test_drafts
):
    """A retried Meta webhook delivery (same message ID) must not trigger a second
    LLM call or a second draft."""
    first_response = client.post("/webhook/whatsapp", json=valid_text_payload)
    assert first_response.status_code == 200
    assert first_response.json()["status"] == "pending_approval"

    initial_llm_calls = len(mock_llm.calls)
    assert initial_llm_calls >= 1
    assert len(test_drafts.list_pending()) == 1

    # Meta redelivers the identical event (e.g. because the first response was slow
    # or a prior non-200 was returned)
    second_response = client.post("/webhook/whatsapp", json=valid_text_payload)
    assert second_response.status_code == 200
    assert second_response.json()["status"] == "duplicate_ignored"

    assert len(mock_llm.calls) == initial_llm_calls
    assert len(test_drafts.list_pending()) == 1
    assert mock_wa.sent_messages == []


def test_llm_failure_returns_200_with_safe_fallback(
    client: TestClient, valid_text_payload, mock_wa, test_drafts
):
    """If the LLM provider fails, the orchestrator produces a deterministic safe
    fallback reply, the webhook acknowledges with 200 (so Meta doesn't retry
    and re-run), and the fallback is queued for staff approval."""
    from app.agent.orchestrator import SAFE_FALLBACK_REPLY
    from app.main import app, get_llm_provider

    failing_llm = MockLLMProvider(raise_error=True)
    app.dependency_overrides[get_llm_provider] = lambda: failing_llm

    response = client.post("/webhook/whatsapp", json=valid_text_payload)

    assert response.status_code == 200
    assert response.json()["status"] == "pending_approval"
    assert [d.draft_text for d in test_drafts.list_pending()] == [SAFE_FALLBACK_REPLY]
    assert mock_wa.sent_messages == []


def test_orchestrator_unexpected_failure_returns_200_agent_error(
    client: TestClient, valid_text_payload, mock_wa, test_drafts
):
    """If AgentOrchestrator.handle_turn raises unexpectedly, the webhook catches it,
    logs safely, queues nothing, and acknowledges with 200 agent_error."""
    from unittest.mock import AsyncMock
    from app.main import app, get_orchestrator

    failing_orch = AsyncMock()
    failing_orch.handle_turn.side_effect = RuntimeError("Simulated internal crash")
    app.dependency_overrides[get_orchestrator] = lambda: failing_orch

    response = client.post("/webhook/whatsapp", json=valid_text_payload)

    assert response.status_code == 200
    assert response.json()["status"] == "agent_error"
    assert mock_wa.sent_messages == []
    assert test_drafts.list_pending() == []


def test_queue_failure_returns_200_queue_error(client: TestClient, valid_text_payload, mock_wa):
    """If the draft cannot be stored, the webhook still acknowledges with 200 and sends nothing."""
    from app.main import app, get_draft_queue

    class BrokenQueue:
        def add(self, **kwargs):
            raise RuntimeError("disk full")

    app.dependency_overrides[get_draft_queue] = lambda: BrokenQueue()

    response = client.post("/webhook/whatsapp", json=valid_text_payload)

    assert response.status_code == 200
    assert response.json()["status"] == "queue_error"
    assert mock_wa.sent_messages == []
