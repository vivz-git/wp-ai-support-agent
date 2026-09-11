import logging
from contextlib import asynccontextmanager
from typing import Optional, Tuple
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import PlainTextResponse

from app.agent.extraction import LeadExtractor
from app.agent.handoff import InMemoryHandoffSink
from app.agent.orchestrator import AgentOrchestrator
from app.agent.store import ConversationStore
from app.config import Settings, get_settings, mask_phone_number
from app.knowledge import KnowledgeBase, get_knowledge_base
from app.llm.base import LLMProvider
from app.llm.groq_provider import GroqProvider
from app.memory import InMemoryConversationMemory
from app.tools import default_registry
from app.tools.registry import ToolRegistry
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
_global_store = ConversationStore()
_global_handoff_sink = InMemoryHandoffSink()
_global_tool_registry = default_registry

_cached_llm_provider: Optional[Tuple[str, str, str, LLMProvider]] = None
_cached_wa_client: Optional[Tuple[str, str, str, WhatsAppClient]] = None
_cached_orchestrator: Optional[Tuple[int, int, int, int, int, int, AgentOrchestrator]] = None


def get_memory() -> InMemoryConversationMemory:
    """Dependency provider for conversation memory and WAMID idempotency ledger."""
    return _global_memory


def get_conversation_store() -> ConversationStore:
    """Dependency provider for conversation state store."""
    return _global_store


def get_handoff_sink() -> InMemoryHandoffSink:
    """Dependency provider for human handoff sink."""
    return _global_handoff_sink


def get_tool_registry() -> ToolRegistry:
    """Dependency provider for agent tools registry."""
    return _global_tool_registry


def get_knowledge() -> KnowledgeBase:
    """Dependency provider for business knowledge base."""
    return get_knowledge_base()


def get_llm_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    """Dependency provider for the LLM abstraction, cached by configuration."""
    global _cached_llm_provider
    cache_key = (settings.groq_api_key, settings.groq_model, settings.system_prompt)
    if _cached_llm_provider is not None and _cached_llm_provider[:3] == cache_key:
        return _cached_llm_provider[3]
    provider = GroqProvider(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        system_prompt=settings.system_prompt,
    )
    _cached_llm_provider = (*cache_key, provider)
    return provider


def get_lead_extractor(llm: LLMProvider = Depends(get_llm_provider)) -> LeadExtractor:
    """Dependency provider for lead extraction."""
    return LeadExtractor(llm=llm)


def get_whatsapp_client(settings: Settings = Depends(get_settings)) -> WhatsAppClient:
    """Dependency provider for WhatsApp Cloud API client, cached by configuration."""
    global _cached_wa_client
    cache_key = (
        settings.whatsapp_access_token,
        settings.whatsapp_phone_number_id,
        settings.whatsapp_api_version,
    )
    if _cached_wa_client is not None and _cached_wa_client[:3] == cache_key:
        return _cached_wa_client[3]
    client = WhatsAppClient(
        access_token=settings.whatsapp_access_token,
        phone_number_id=settings.whatsapp_phone_number_id,
        api_version=settings.whatsapp_api_version,
    )
    _cached_wa_client = (*cache_key, client)
    return client


def get_orchestrator(
    llm: LLMProvider = Depends(get_llm_provider),
    knowledge: KnowledgeBase = Depends(get_knowledge),
    tools: ToolRegistry = Depends(get_tool_registry),
    store: ConversationStore = Depends(get_conversation_store),
    extractor: LeadExtractor = Depends(get_lead_extractor),
    handoff_sink: InMemoryHandoffSink = Depends(get_handoff_sink),
) -> AgentOrchestrator:
    """Dependency provider for deterministic AgentOrchestrator.
    
    Caches the orchestrator instance by dependency identity so long-lived
    production singletons are not re-instantiated per request, while test
    dependency overrides instantly yield an appropriately wired instance.
    """
    global _cached_orchestrator
    cache_key = (id(llm), id(knowledge), id(tools), id(store), id(extractor), id(handoff_sink))
    if _cached_orchestrator is not None and _cached_orchestrator[:6] == cache_key:
        return _cached_orchestrator[6]
    orch = AgentOrchestrator(
        llm=llm,
        knowledge=knowledge,
        tools=tools,
        store=store,
        extractor=extractor,
        handoff_sink=handoff_sink,
    )
    _cached_orchestrator = (*cache_key, orch)
    return orch


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()
    logger.info("Initializing AI WhatsApp Support Agent (Milestone 2)...")
    # Pre-warm cached knowledge base at service startup
    get_knowledge_base()
    logger.info("FastAPI service ready. Configured API version: %s", settings.whatsapp_api_version)
    yield
    logger.info("Shutting down AI WhatsApp Support Agent...")


app = FastAPI(
    title="AI WhatsApp Support Agent",
    description="Portfolio Demo #2 - WhatsApp Cloud API + Groq LLM Assistant",
    version="0.2.0",
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
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    wa_client: WhatsAppClient = Depends(get_whatsapp_client),
):
    """Meta WhatsApp Cloud API event notification receiver.
    
    Receives incoming webhook payloads from Meta:
    - Parses text messages safely
    - Ignores status notifications and unsupported media without crashing
    - Deduplicates delivery using transport-level WAMID ledger
    - Dispatches to AgentOrchestrator
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
    # same message ID. Without this check a retry would re-run the agent and send a second
    # duplicate reply to the customer.
    if memory.has_processed(msg.message_id):
        logger.info(
            "Ignored duplicate webhook delivery for message_id=%s (sender %s)",
            msg.message_id,
            masked_sender,
        )
        return {"status": "duplicate_ignored", "message_id": msg.message_id}

    memory.mark_processed(msg.message_id)

    # Blank text guard: handle_turn requires non-blank text
    if not text.strip():
        logger.info(
            "Ignored empty text message from %s (id: %s)",
            masked_sender,
            msg.message_id,
        )
        return {
            "status": "ignored",
            "reason": "empty_text",
            "message_id": msg.message_id,
        }

    logger.info("Processing incoming message from %s (id: %s)", masked_sender, msg.message_id)

    # Call AgentOrchestrator
    try:
        turn_result = await orchestrator.handle_turn(
            sender_id=sender,
            text=text,
            message_id=msg.message_id,
        )
    except Exception as exc:
        logger.error(
            "Agent turn failed for %s: %s",
            masked_sender,
            type(exc).__name__,
        )
        logger.debug("Agent turn failure detail", exc_info=exc)
        return {"status": "agent_error", "message_id": msg.message_id}

    reply_text = turn_result.reply_text

    # Deliver reply via WhatsApp Cloud API
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
            logger.error(
                "Failed to deliver WhatsApp reply to %s: %s",
                masked_sender,
                type(exc).__name__,
            )
            logger.debug("WhatsApp delivery failure detail", exc_info=exc)
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
