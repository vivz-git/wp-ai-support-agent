import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import PlainTextResponse

from app.config import Settings, get_settings, mask_phone_number
from app.llm.base import LLMProvider, LLMProviderError
from app.llm.groq_provider import GroqProvider
from app.memory import InMemoryConversationMemory
from app.whatsapp.client import WhatsAppClient, WhatsAppClientError
from app.whatsapp.models import parse_incoming_webhook

# Meta error code for "Recipient phone number not in allowed list" — returned when
# sending to a number that isn't a verified tester on a WhatsApp Cloud API test number
# (this is exactly what happens when replying to Meta's webhook dashboard test/simulator
# event, whose sender is a synthetic number, not a real customer).
META_RECIPIENT_NOT_ALLOWED_ERROR_CODE = 131030

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("whatsapp_agent")


# Module-level singletons for default dependencies
_global_memory = InMemoryConversationMemory()


def get_memory() -> InMemoryConversationMemory:
    """Dependency provider for conversation memory."""
    return _global_memory


def get_llm_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    """Dependency provider for the LLM abstraction."""
    return GroqProvider(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        system_prompt=settings.system_prompt,
    )


def get_whatsapp_client(settings: Settings = Depends(get_settings)) -> WhatsAppClient:
    """Dependency provider for WhatsApp Cloud API client."""
    return WhatsAppClient(
        access_token=settings.whatsapp_access_token,
        phone_number_id=settings.whatsapp_phone_number_id,
        api_version=settings.whatsapp_api_version,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()
    logger.info("Initializing AI WhatsApp Support Agent (Milestone 1)...")
    logger.info("FastAPI service ready. Configured API version: %s", settings.whatsapp_api_version)
    yield
    logger.info("Shutting down AI WhatsApp Support Agent...")


app = FastAPI(
    title="AI WhatsApp Support Agent",
    description="Portfolio Demo #2 - WhatsApp Cloud API + Groq LLM Assistant",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["System"])
async def health_check():
    """Liveness probe endpoint."""
    return {"status": "ok", "service": "demo-whatsapp-support"}


@app.get("/webhook/whatsapp", tags=["WhatsApp Webhook"])
async def verify_webhook(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
    settings: Settings = Depends(get_settings),
):
    """Meta WhatsApp Cloud API webhook verification handshake.
    
    Meta sends a GET request with hub.mode, hub.verify_token, and hub.challenge.
    If valid, this endpoint must return the exact hub.challenge as plain text.
    """
    configured_token = settings.whatsapp_verify_token

    if (
        hub_mode == "subscribe"
        and configured_token
        and hub_verify_token == configured_token
        and hub_challenge is not None
    ):
        logger.info("Meta webhook verification handshake succeeded")
        return PlainTextResponse(content=hub_challenge, status_code=200)

    logger.warning("Meta webhook verification handshake rejected (token mismatch or invalid mode)")
    return Response(content="Forbidden", status_code=403, media_type="text/plain")


@app.post("/webhook/whatsapp", tags=["WhatsApp Webhook"])
async def receive_webhook(
    request: Request,
    settings: Settings = Depends(get_settings),
    memory: InMemoryConversationMemory = Depends(get_memory),
    llm: LLMProvider = Depends(get_llm_provider),
    wa_client: WhatsAppClient = Depends(get_whatsapp_client),
):
    """Meta WhatsApp Cloud API event notification receiver.
    
    Receives incoming webhook payloads from Meta:
    - Parses text messages safely
    - Ignores status notifications and unsupported media without crashing
    - Dispatches to LLM abstraction
    - Sends assistant response back via WhatsApp Cloud API
    - Returns HTTP 200
    """
    try:
        payload = await request.json()
    except Exception:
        logger.warning("Rejected invalid non-JSON webhook payload")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    parse_result = parse_incoming_webhook(payload)

    if not parse_result.is_valid_structure:
        logger.warning("Rejected malformed webhook structure: %s", parse_result.reason)
        raise HTTPException(
            status_code=400,
            detail=parse_result.reason or "Malformed webhook payload",
        )

    # If not a text message event, safely return 200 OK so Meta doesn't retry
    if parse_result.event_type != "text_message" or not parse_result.message:
        logger.info(
            "Ignored unsupported/non-message event: type=%s, reason=%s",
            parse_result.event_type,
            parse_result.reason,
        )
        return {
            "status": "ignored",
            "event_type": parse_result.event_type,
            "reason": parse_result.reason,
        }

    msg = parse_result.message
    sender = msg.sender
    text = msg.text
    masked_sender = mask_phone_number(sender)

    # Idempotency: Meta retries webhook delivery (e.g. after a non-200 response) using the
    # same message ID. Without this check a retry would re-run the LLM and send a second
    # duplicate reply to the customer.
    if memory.has_processed(msg.message_id):
        logger.info(
            "Ignored duplicate webhook delivery for message_id=%s (sender %s)",
            msg.message_id,
            masked_sender,
        )
        return {"status": "duplicate_ignored", "message_id": msg.message_id}

    memory.mark_processed(msg.message_id)

    logger.info("Processing incoming message from %s (id: %s)", masked_sender, msg.message_id)

    # 1. Update in-memory history with user message
    memory.add_user_message(sender, text)

    # 2. Retrieve history for sender
    history = memory.get_messages(sender)

    # 3. Call LLM provider
    try:
        reply_text = await llm.get_agent_reply(history)
    except LLMProviderError as exc:
        # The inbound message was received and recorded; a downstream LLM failure should
        # not cause Meta to redeliver and reprocess the same event, so we still return 200.
        logger.error("LLM reply generation failed for %s: %s", masked_sender, exc)
        return {"status": "llm_error", "message_id": msg.message_id}

    # 4. Save reply in memory
    memory.add_assistant_message(sender, reply_text)

    # 5. Send reply via Meta Cloud API
    try:
        await wa_client.send_text(to=sender, body=reply_text)
    except WhatsAppClientError as exc:
        if exc.error_code == META_RECIPIENT_NOT_ALLOWED_ERROR_CODE:
            # Expected for Meta's webhook dashboard test/simulator event, whose synthetic
            # sender is never a verified recipient on a test number. Not a real failure.
            logger.info(
                "Skipped reply delivery: sender %s is not an allowed test recipient "
                "(Meta test/simulator event, error code %s)",
                masked_sender,
                exc.error_code,
            )
        else:
            logger.error("Failed to deliver WhatsApp reply to %s: %s", masked_sender, exc)
        # Either way, the incoming event was successfully received and processed; the
        # outbound delivery failure is not something a Meta retry of this same event
        # could fix, so we still acknowledge with 200 to prevent a retry storm.
        return {"status": "send_failed", "message_id": msg.message_id}

    logger.info("Completed reply cycle for %s", masked_sender)
    return {"status": "ok", "message_id": msg.message_id}


if __name__ == "__main__":
    current_settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=current_settings.app_host,
        port=current_settings.app_port,
        reload=True,
    )
