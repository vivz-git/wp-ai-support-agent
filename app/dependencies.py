"""FastAPI dependency providers shared by the webhook (``app.main``) and the
staff page (``app.staff``). Kept separate so ``app.main`` can mount the staff
router without a circular import. Both are re-exported from ``app.main``.
"""

from typing import Optional, Tuple

from fastapi import Depends

from app.approval import DraftQueue
from app.config import Settings, get_settings
from app.whatsapp.client import WhatsAppClient

_cached_wa_client: Optional[Tuple[str, str, str, WhatsAppClient]] = None
_cached_draft_queue: Optional[Tuple[str, DraftQueue]] = None


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
