import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.llm.base import ChatMessage, LLMProviderError, get_agent_reply
from app.llm.groq_provider import GroqProvider
from app.memory import InMemoryConversationMemory


def test_llm_abstraction_with_mock_client():
    """Verify GroqProvider formats messages correctly and extracts reply from client."""
    mock_groq_client = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a simulated AI reply."
    mock_completion.choices = [mock_choice]

    mock_groq_client.chat.completions.create = AsyncMock(return_value=mock_completion)

    provider = GroqProvider(
        api_key="test_groq_api_key",
        model="llama-3.3-70b-versatile",
        system_prompt="Test system prompt.",
        client=mock_groq_client,
    )

    messages = [
        ChatMessage(role="user", content="Hi"),
        ChatMessage(role="assistant", content="Hello!"),
        ChatMessage(role="user", content="What are your hours?"),
    ]

    reply = asyncio.run(provider.get_agent_reply(messages))

    assert reply == "This is a simulated AI reply."
    mock_groq_client.chat.completions.create.assert_awaited_once()

    # Verify call parameters
    kwargs = mock_groq_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "llama-3.3-70b-versatile"
    sent_messages = kwargs["messages"]
    assert len(sent_messages) == 4
    assert sent_messages[0] == {"role": "system", "content": "Test system prompt."}
    assert sent_messages[1] == {"role": "user", "content": "Hi"}
    assert sent_messages[2] == {"role": "assistant", "content": "Hello!"}
    assert sent_messages[3] == {"role": "user", "content": "What are your hours?"}


def test_llm_abstraction_helper_function():
    """Verify get_agent_reply helper delegates correctly to provider."""
    mock_provider = AsyncMock()
    mock_provider.get_agent_reply.return_value = "Delegated response"

    messages = [ChatMessage(role="user", content="Ping")]
    result = asyncio.run(get_agent_reply(messages, mock_provider))

    assert result == "Delegated response"
    mock_provider.get_agent_reply.assert_awaited_once_with(messages)


def test_llm_provider_error_handling():
    """Verify GroqProvider wraps API exceptions into LLMProviderError without leaking secrets."""
    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("Groq service rate limited"))

    provider = GroqProvider(
        api_key="test_groq_api_key",
        client=mock_groq_client,
    )

    messages = [ChatMessage(role="user", content="Ping")]
    with pytest.raises(LLMProviderError) as exc_info:
        asyncio.run(provider.get_agent_reply(messages))

    assert "Groq API error" in str(exc_info.value)


def test_in_memory_conversation_memory():
    """Verify InMemoryConversationMemory retains order and enforces bounded message length."""
    memory = InMemoryConversationMemory(max_messages=3)
    sender = "919876543210"

    assert memory.get_messages(sender) == []

    memory.add_user_message(sender, "Msg 1")
    memory.add_assistant_message(sender, "Reply 1")
    memory.add_user_message(sender, "Msg 2")

    messages = memory.get_messages(sender)
    assert len(messages) == 3
    assert messages[0].content == "Msg 1"
    assert messages[1].content == "Reply 1"
    assert messages[2].content == "Msg 2"

    # Add 4th message - oldest (Msg 1) should be evicted
    memory.add_assistant_message(sender, "Reply 2")
    messages = memory.get_messages(sender)
    assert len(messages) == 3
    assert messages[0].content == "Reply 1"
    assert messages[1].content == "Msg 2"
    assert messages[2].content == "Reply 2"

    # Clear memory
    memory.clear(sender)
    assert memory.get_messages(sender) == []


def test_memory_message_id_idempotency_tracking():
    """Verify has_processed/mark_processed dedupe Meta message IDs for webhook retries."""
    memory = InMemoryConversationMemory()

    assert memory.has_processed("wamid.ABC123") is False

    memory.mark_processed("wamid.ABC123")
    assert memory.has_processed("wamid.ABC123") is True

    # A different message ID is unaffected
    assert memory.has_processed("wamid.DEF456") is False

    # Marking the same ID again is a no-op, not an error
    memory.mark_processed("wamid.ABC123")
    assert memory.has_processed("wamid.ABC123") is True
