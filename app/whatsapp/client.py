import logging
from typing import Any, Dict, Optional
import httpx
from app.config import mask_phone_number

logger = logging.getLogger(__name__)


class WhatsAppClientError(Exception):
    """Exception raised when sending a message via WhatsApp Cloud API fails."""

    def __init__(self, message: str, error_code: Optional[int] = None):
        super().__init__(message)
        self.error_code = error_code


class WhatsAppClient:
    """Client for sending messages via the Meta WhatsApp Cloud API."""

    def __init__(
        self,
        access_token: str,
        phone_number_id: str,
        api_version: str = "v22.0",
        base_url: str = "https://graph.facebook.com",
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.api_version = api_version
        self.base_url = base_url.rstrip("/")
        self._http_client = http_client

    @property
    def endpoint_url(self) -> str:
        """Construct the full Meta Cloud API messages endpoint."""
        return f"{self.base_url}/{self.api_version}/{self.phone_number_id}/messages"

    async def send_text(self, to: str, body: str) -> Dict[str, Any]:
        """Send a plain text message to a WhatsApp user.
        
        Args:
            to: Recipient phone number in international format (e.g. '919876543210').
            body: Message body text to send.
            
        Returns:
            Decoded JSON dictionary response from Meta Cloud API.
            
        Raises:
            WhatsAppClientError: If the request fails or credentials are missing.
        """
        if not self.access_token or not self.access_token.strip():
            raise WhatsAppClientError("WHATSAPP_ACCESS_TOKEN is not configured")
        if not self.phone_number_id or not self.phone_number_id.strip():
            raise WhatsAppClientError("WHATSAPP_PHONE_NUMBER_ID is not configured")

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": body,
            },
        }

        masked_to = mask_phone_number(to)
        logger.info("Sending WhatsApp text message to %s", masked_to)

        if self._http_client is not None:
            return await self._execute_request(self._http_client, headers, payload, masked_to)

        async with httpx.AsyncClient(timeout=15.0) as client:
            return await self._execute_request(client, headers, payload, masked_to)

    async def _execute_request(
        self,
        client: httpx.AsyncClient,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        masked_to: str,
    ) -> Dict[str, Any]:
        try:
            response = await client.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            logger.info("Successfully sent WhatsApp message to %s", masked_to)
            return data
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            safe_error: str
            meta_error_code: Optional[int] = None
            try:
                err_data = exc.response.json()
                error_obj = err_data.get("error", {})
                safe_error = error_obj.get("message", exc.response.text)
                meta_error_code = error_obj.get("code")
            except Exception:
                safe_error = exc.response.text

            logger.error(
                "WhatsApp Cloud API responded with error %d (code=%s) for recipient %s: %s",
                status_code,
                meta_error_code,
                masked_to,
                safe_error,
            )
            raise WhatsAppClientError(
                f"WhatsApp Cloud API error ({status_code}): {safe_error}",
                error_code=meta_error_code,
            ) from exc
        except httpx.RequestError as exc:
            logger.error(
                "Network request to WhatsApp Cloud API failed for %s: %s",
                masked_to,
                type(exc).__name__,
            )
            raise WhatsAppClientError(
                f"Network error connecting to WhatsApp Cloud API: {type(exc).__name__}"
            ) from exc
