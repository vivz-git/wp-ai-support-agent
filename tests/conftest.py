import os
import pytest
from typing import Any, Dict, List, Optional, Tuple, Union
from fastapi.testclient import TestClient

# Configure test environment variables before application settings load
os.environ["WHATSAPP_ACCESS_TOKEN"] = "mock_test_access_token_12345"
os.environ["WHATSAPP_PHONE_NUMBER_ID"] = "10987654321"
os.environ["WHATSAPP_VERIFY_TOKEN"] = "test_webhook_secret_token"
os.environ["WHATSAPP_API_VERSION"] = "v22.0"
os.environ["GROQ_API_KEY"] = "mock_groq_api_key_67890"
os.environ["GROQ_MODEL"] = "llama-3.3-70b-versatile"
os.environ["MAX_MEMORY_MESSAGES"] = "10"

from app.agent.handoff import InMemoryHandoffSink
from app.agent.store import ConversationStore
from app.config import Settings, get_settings
from app.llm.base import ChatMessage, LLMProvider, LLMProviderError, LLMResponse
from app.main import (
    app,
    get_conversation_store,
    get_handoff_sink,
    get_llm_provider,
    get_memory,
    get_settings as main_get_settings,
    get_whatsapp_client,
)
from app.memory import InMemoryConversationMemory
from app.whatsapp.client import WhatsAppClient, WhatsAppClientError


class MockLLMProvider:
    """Mock LLM provider that returns predetermined responses."""

    def __init__(
        self,
        response_text: str = "Hello! I am your AI assistant. How can I help you?",
        raise_error: bool = False,
    ):
        self.response_text = response_text
        self.raise_error = raise_error
        self.calls: List[List[ChatMessage]] = []

    async def get_agent_reply(self, messages: List[ChatMessage]) -> str:
        self.calls.append(messages)
        if self.raise_error:
            raise LLMProviderError("Simulated Groq API failure")
        return self.response_text

    async def complete(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Any]] = None,
        tool_choice: Optional[Any] = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        if self.raise_error:
            raise LLMProviderError("Simulated Groq API failure")
        return LLMResponse(content=self.response_text, finish_reason="stop")


class MockWhatsAppClient:
    """Mock WhatsApp client that records sent messages without network calls."""

    def __init__(self, raise_error_code: int = None):
        self.sent_messages: List[Dict[str, str]] = []
        self.raise_error_code = raise_error_code

    async def send_text(self, to: str, body: str) -> Dict[str, Any]:
        if self.raise_error_code is not None:
            raise WhatsAppClientError(
                "WhatsApp Cloud API error (131030): Recipient phone number not in allowed list",
                error_code=self.raise_error_code,
            )
        self.sent_messages.append({"to": to, "body": body})
        return {
            "messaging_product": "whatsapp",
            "contacts": [{"input": to, "wa_id": to}],
            "messages": [{"id": f"wamid.mock_{len(self.sent_messages)}"}],
        }


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        whatsapp_access_token="mock_test_access_token_12345",
        whatsapp_phone_number_id="10987654321",
        whatsapp_verify_token="test_webhook_secret_token",
        whatsapp_api_version="v22.0",
        groq_api_key="mock_groq_api_key_67890",
        groq_model="llama-3.3-70b-versatile",
    )


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def mock_wa() -> MockWhatsAppClient:
    return MockWhatsAppClient()


@pytest.fixture
def test_memory() -> InMemoryConversationMemory:
    return InMemoryConversationMemory(max_messages=5)


@pytest.fixture
def test_store() -> ConversationStore:
    return ConversationStore()


@pytest.fixture
def test_sink() -> InMemoryHandoffSink:
    return InMemoryHandoffSink()


@pytest.fixture
def client(mock_settings, mock_llm, mock_wa, test_memory, test_store, test_sink) -> TestClient:
    """Provide a TestClient with dependency overrides for isolated testing."""
    app.dependency_overrides[main_get_settings] = lambda: mock_settings
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm
    app.dependency_overrides[get_whatsapp_client] = lambda: mock_wa
    app.dependency_overrides[get_memory] = lambda: test_memory
    app.dependency_overrides[get_conversation_store] = lambda: test_store
    app.dependency_overrides[get_handoff_sink] = lambda: test_sink

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def valid_text_payload() -> Dict[str, Any]:
    """Sample valid incoming WhatsApp text message payload."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "100609346426090",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15550234567",
                                "phone_number_id": "10987654321",
                            },
                            "contacts": [
                                {
                                    "profile": {"name": "Test User"},
                                    "wa_id": "919876543210",
                                }
                            ],
                            "messages": [
                                {
                                    "from": "919876543210",
                                    "id": "wamid.HBgLMTE1...",
                                    "timestamp": "1710000000",
                                    "text": {
                                        "body": "Hello, I want to inquire about pricing.",
                                    },
                                    "type": "text",
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }


@pytest.fixture
def status_update_payload() -> Dict[str, Any]:
    """Sample Meta status update notification payload (sent/delivered/read)."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "100609346426090",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15550234567",
                                "phone_number_id": "10987654321",
                            },
                            "statuses": [
                                {
                                    "id": "wamid.HBgLMTE1...",
                                    "status": "delivered",
                                    "timestamp": "1710000005",
                                    "recipient_id": "919876543210",
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }


@pytest.fixture
def meta_sample_test_payload() -> Dict[str, Any]:
    """Meta webhook dashboard 'Test' button synthetic sample event.

    Mirrors Meta's documented example payload used by the webhook simulator:
    a fixed placeholder sender/message ID that is never a real customer and is
    never a verified recipient on a WhatsApp Cloud API test number.
    """
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "27849239640",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "16505551111",
                                "phone_number_id": "123456123",
                            },
                            "contacts": [
                                {
                                    "profile": {"name": "Kerry Fisher"},
                                    "wa_id": "16315551181",
                                }
                            ],
                            "messages": [
                                {
                                    "from": "16315551181",
                                    "id": "wamid.HBgLMTY0NjcwNDM1OTUVAgARGBI5QjNsQTBFQjY0RUJEOAA=",
                                    "timestamp": "1603059201",
                                    "text": {"body": "This is a sample message"},
                                    "type": "text",
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }


@pytest.fixture
def unsupported_media_payload() -> Dict[str, Any]:
    """Sample incoming image payload (unsupported in Milestone 1)."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "100609346426090",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15550234567",
                                "phone_number_id": "10987654321",
                            },
                            "messages": [
                                {
                                    "from": "919876543210",
                                    "id": "wamid.HBgLMTE2...",
                                    "timestamp": "1710000010",
                                    "type": "image",
                                    "image": {"id": "123456789"},
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }
