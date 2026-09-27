import logging
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import PlainTextResponse

from app.agent.orchestrator import AgentOrchestrator
from app.approval import DraftQueue
from app.config import Settings, get_settings
from app.dependencies import (
    get_conversation_store,
    get_draft_queue,
    get_handoff_sink,
    get_knowledge,
    get_lead_extractor,
    get_llm_provider,
    get_memory,
    get_orchestrator,
    get_tool_registry,
    get_whatsapp_client,
)  # noqa: F401 (re-exported for dependency overrides)
from app.knowledge import get_knowledge_base
from app.memory import InMemoryConversationMemory
from app.simulate import router as simulate_router
from app.staff import router as staff_router
from app.turn_pipeline import process_incoming_message
from app.whatsapp.models import parse_incoming_webhook

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("whatsapp_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()
    logger.info("Initializing SmileCare Dental WhatsApp assistant...")
    # Pre-warm cached knowledge base at service startup
    get_knowledge_base()
    logger.info("FastAPI service ready. Configured API version: %s", settings.whatsapp_api_version)
    yield
    logger.info("Shutting down AI WhatsApp Support Agent...")


app = FastAPI(
    title="AI WhatsApp Support Agent",
    description="Portfolio Demo #2 - SmileCare Dental WhatsApp assistant (Cloud API + Groq) with staff approval",
    version="0.4.0",
    lifespan=lifespan,
)
app.include_router(staff_router)
app.include_router(simulate_router)


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
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    drafts: DraftQueue = Depends(get_draft_queue),
):
    """Meta WhatsApp Cloud API event notification receiver.

    Receives incoming webhook payloads from Meta:
    - Parses text messages safely
    - Ignores status notifications and unsupported media without crashing
    - Hands text messages to ``process_incoming_message`` (idempotency ->
      AgentOrchestrator -> draft queue), the same pipeline the ``/simulate``
      demo page uses for a typed message
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
    return await process_incoming_message(
        sender=msg.sender,
        text=msg.text,
        message_id=msg.message_id,
        memory=memory,
        orchestrator=orchestrator,
        drafts=drafts,
    )


if __name__ == "__main__":
    current_settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=current_settings.app_host,
        port=current_settings.app_port,
        reload=True,
    )
