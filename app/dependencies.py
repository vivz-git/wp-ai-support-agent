"""FastAPI dependency providers shared by the webhook (``app.main``), the
staff page (``app.staff``) and the demo simulator (``app.simulate``). Kept
separate so those routers can be mounted on ``app.main`` without a circular
import. All of them are re-exported from ``app.main`` for backward
compatibility with existing dependency overrides in tests.
"""

from typing import Optional, Tuple

from fastapi import Depends

from app.agent.extraction import LeadExtractor
from app.agent.handoff import InMemoryHandoffSink
from app.agent.orchestrator import AgentOrchestrator
from app.agent.store import ConversationStore
from app.approval import DraftQueue
from app.config import Settings, get_settings
from app.knowledge import KnowledgeBase, get_knowledge_base
from app.llm.base import LLMProvider
from app.llm.groq_provider import GroqProvider
from app.memory import InMemoryConversationMemory
from app.tools import default_registry
from app.tools.registry import ToolRegistry
from app.whatsapp.client import WhatsAppClient

# Module-level singletons for default dependencies
_global_memory = InMemoryConversationMemory()
_global_store = ConversationStore()
_global_handoff_sink = InMemoryHandoffSink()
_global_tool_registry = default_registry

_cached_llm_provider: Optional[Tuple[str, str, str, LLMProvider]] = None
_cached_orchestrator: Optional[Tuple[int, int, int, int, int, int, AgentOrchestrator]] = None
_cached_wa_client: Optional[Tuple[str, str, str, WhatsAppClient]] = None
_cached_draft_queue: Optional[Tuple[str, DraftQueue]] = None


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


def get_draft_queue(settings: Settings = Depends(get_settings)) -> DraftQueue:
    """Dependency provider for the staff approval queue, cached by database path."""
    global _cached_draft_queue
    if _cached_draft_queue is not None and _cached_draft_queue[0] == settings.drafts_db_path:
        return _cached_draft_queue[1]
    queue = DraftQueue(settings.drafts_db_path)
    _cached_draft_queue = (settings.drafts_db_path, queue)
    return queue
