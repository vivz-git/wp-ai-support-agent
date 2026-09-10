import asyncio
from unittest.mock import AsyncMock, MagicMock
import httpx
import pytest

from app.whatsapp.client import WhatsAppClient, WhatsAppClientError


def test_whatsapp_client_send_text_success():
    """Verify WhatsAppClient constructs the correct Meta Graph API request and payload."""
    mock_http_client = MagicMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "messaging_product": "whatsapp",
        "contacts": [{"input": "919876543210", "wa_id": "919876543210"}],
        "messages": [{"id": "wamid.HBgL123456"}],
    }
    mock_response.raise_for_status = MagicMock()

    mock_http_client.post = AsyncMock(return_value=mock_response)

    client = WhatsAppClient(
        access_token="test_meta_token",
        phone_number_id="1234567890",
        api_version="v22.0",
        http_client=mock_http_client,
    )

    recipient = "919876543210"
    message_text = "Hello! Your appointment is confirmed."

    result = asyncio.run(client.send_text(to=recipient, body=message_text))

    assert result["messaging_product"] == "whatsapp"
    assert result["messages"][0]["id"] == "wamid.HBgL123456"

    # Verify request arguments
    mock_http_client.post.assert_awaited_once()
    args, kwargs = mock_http_client.post.call_args

    assert args[0] == "https://graph.facebook.com/v22.0/1234567890/messages"
    assert kwargs["headers"]["Authorization"] == "Bearer test_meta_token"
    assert kwargs["headers"]["Content-Type"] == "application/json"

    json_payload = kwargs["json"]
    assert json_payload["messaging_product"] == "whatsapp"
    assert json_payload["recipient_type"] == "individual"
    assert json_payload["to"] == recipient
    assert json_payload["type"] == "text"
    assert json_payload["text"]["body"] == message_text


def test_whatsapp_client_http_error():
    """Verify WhatsAppClient wraps HTTP errors safely into WhatsAppClientError."""
    mock_http_client = MagicMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = '{"error": {"message": "Invalid OAuth access token."}}'
    mock_response.json.return_value = {"error": {"message": "Invalid OAuth access token."}}

    http_error = httpx.HTTPStatusError(
        message="Client error '401 Unauthorized'",
        request=MagicMock(),
        response=mock_response,
    )
    mock_response.raise_for_status.side_effect = http_error
    mock_http_client.post = AsyncMock(return_value=mock_response)

    client = WhatsAppClient(
        access_token="expired_token",
        phone_number_id="1234567890",
        http_client=mock_http_client,
    )

    with pytest.raises(WhatsAppClientError) as exc_info:
        asyncio.run(client.send_text(to="919876543210", body="Hello"))

    assert "WhatsApp Cloud API error (401)" in str(exc_info.value)
    assert "Invalid OAuth access token" in str(exc_info.value)


def test_whatsapp_client_network_error():
    """Verify WhatsAppClient wraps network connection errors."""
    mock_http_client = MagicMock(spec=httpx.AsyncClient)
    mock_http_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

    client = WhatsAppClient(
        access_token="test_token",
        phone_number_id="1234567890",
        http_client=mock_http_client,
    )

    with pytest.raises(WhatsAppClientError) as exc_info:
        asyncio.run(client.send_text(to="919876543210", body="Hello"))

    assert "Network error" in str(exc_info.value)


def test_whatsapp_client_missing_credentials():
    """Verify WhatsAppClient raises error immediately if credentials are missing."""
    client_no_token = WhatsAppClient(access_token="", phone_number_id="12345")
    with pytest.raises(WhatsAppClientError) as exc_info:
        asyncio.run(client_no_token.send_text(to="919876543210", body="Hello"))
    assert "WHATSAPP_ACCESS_TOKEN is not configured" in str(exc_info.value)

    client_no_phone = WhatsAppClient(access_token="valid_token", phone_number_id="")
    with pytest.raises(WhatsAppClientError) as exc_info:
        asyncio.run(client_no_phone.send_text(to="919876543210", body="Hello"))
    assert "WHATSAPP_PHONE_NUMBER_ID is not configured" in str(exc_info.value)
