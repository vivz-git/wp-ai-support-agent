import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.agent.extraction import _build_messages as build_extraction_messages
from app.agent.prompts import PromptBuilder
from app.agent.state import ConversationState, Intent
from app.knowledge import get_knowledge_base
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


# ---------------------------------------------------------------------------
# Slice 16: GroqProvider must not double up system messages.
#
# Prompt ownership rule: if the caller already supplies a system message
# (PromptBuilder, LeadExtractor), GroqProvider must preserve it as the one
# and only system message and must NOT also inject its own configured
# ``system_prompt``. If the caller supplies no system message at all
# (legacy ``get_agent_reply`` callers), the configured default is still
# prepended for backward compatibility.
# ---------------------------------------------------------------------------


def _system_messages(formatted):
    return [m for m in formatted if m["role"] == "system"]


def test_format_messages_prepends_default_system_prompt_when_none_supplied():
    """1. No system message in input -> configured system prompt is prepended."""
    provider = GroqProvider(api_key="test_groq_api_key", system_prompt="Default system prompt.", client=MagicMock())
    messages = [ChatMessage(role="user", content="Hi")]

    formatted = provider._format_messages(messages)

    assert len(formatted) == 2
    assert formatted[0] == {"role": "system", "content": "Default system prompt."}


def test_format_messages_does_not_prepend_when_caller_supplies_system_message():
    """2. One system message present -> configured system prompt is NOT prepended."""
    provider = GroqProvider(api_key="test_groq_api_key", system_prompt="Default system prompt.", client=MagicMock())
    messages = [
        ChatMessage(role="system", content="Caller system prompt."),
        ChatMessage(role="user", content="Hi"),
    ]

    formatted = provider._format_messages(messages)

    system_msgs = _system_messages(formatted)
    assert len(system_msgs) == 1
    assert all("Default system prompt." != m["content"] for m in formatted)


def test_format_messages_preserves_caller_system_prompt_exactly():
    """3. Caller's system prompt content reaches the wire format unmodified."""
    caller_prompt = "Very specific caller instructions.\nWith a second line."
    provider = GroqProvider(api_key="test_groq_api_key", system_prompt="Default.", client=MagicMock())
    messages = [
        ChatMessage(role="system", content=caller_prompt),
        ChatMessage(role="user", content="Hi"),
    ]

    formatted = provider._format_messages(messages)

    assert formatted[0] == {"role": "system", "content": caller_prompt}


def test_format_messages_agent_prompt_builder_yields_single_system_message():
    """4. Real PromptBuilder output -> exactly one system message reaches Groq formatting."""
    knowledge = get_knowledge_base()
    state = ConversationState.new("919876543210")
    state.begin_turn()
    state.set_intent(Intent.PRODUCT_INQUIRY, 0.87)
    bundle = PromptBuilder.build(state=state, knowledge=knowledge, current_message="Hi, what's fresh?")
    messages = bundle.to_messages()

    provider = GroqProvider(api_key="test_groq_api_key", system_prompt="Legacy Milestone 1 default.", client=MagicMock())
    formatted = provider._format_messages(messages)

    system_msgs = _system_messages(formatted)
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == bundle.system_prompt
    assert "Legacy Milestone 1 default." not in system_msgs[0]["content"]


def test_format_messages_extraction_yields_single_system_message():
    """5. Real LeadExtractor message list -> exactly one system message, the extraction prompt."""
    messages = build_extraction_messages("I'd like a filter roast, please.")

    provider = GroqProvider(api_key="test_groq_api_key", system_prompt="Legacy Milestone 1 default.", client=MagicMock())
    formatted = provider._format_messages(messages)

    system_msgs = _system_messages(formatted)
    assert len(system_msgs) == 1
    assert "data-extraction function" in system_msgs[0]["content"]
    assert "Legacy Milestone 1 default." not in system_msgs[0]["content"]


def test_format_messages_system_prompt_ordering_preserved():
    """6. System message stays first, followed by the rest of the conversation in order."""
    provider = GroqProvider(api_key="test_groq_api_key", system_prompt="Default.", client=MagicMock())
    messages = [
        ChatMessage(role="system", content="Caller system."),
        ChatMessage(role="user", content="first"),
        ChatMessage(role="assistant", content="second"),
        ChatMessage(role="user", content="third"),
    ]

    formatted = provider._format_messages(messages)

    assert [m["content"] for m in formatted] == ["Caller system.", "first", "second", "third"]


def test_format_messages_ordinary_messages_unchanged_shape():
    """7. Plain user/assistant messages keep the exact {"role", "content"} shape."""
    provider = GroqProvider(api_key="test_groq_api_key", client=MagicMock())
    messages = [
        ChatMessage(role="user", content="Hi"),
        ChatMessage(role="assistant", content="Hello"),
    ]

    formatted = provider._format_messages(messages)

    assert all(set(m.keys()) == {"role", "content"} for m in formatted)


def test_format_messages_tool_transcript_fields_preserved():
    """8 & 9. Tool-calling messages keep their tool_calls/tool_call_id wire fields."""
    provider = GroqProvider(api_key="test_groq_api_key", client=MagicMock())
    messages = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                {"id": "call_1", "type": "function", "function": {"name": "product_lookup", "arguments": "{}"}}
            ],
        ),
        ChatMessage(role="tool", content='{"status": "ok"}', tool_call_id="call_1"),
    ]

    formatted = provider._format_messages(messages)

    assert _system_messages(formatted) == [{"role": "system", "content": "sys"}]
    assistant_msg, tool_msg = formatted[1], formatted[2]
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "product_lookup"
    assert tool_msg == {"role": "tool", "content": '{"status": "ok"}', "tool_call_id": "call_1"}


def test_format_messages_no_duplicate_system_messages_regardless_of_input():
    """11. Never more than one system message reaches the wire format."""
    provider = GroqProvider(api_key="test_groq_api_key", system_prompt="Default.", client=MagicMock())

    no_system = provider._format_messages([ChatMessage(role="user", content="hi")])
    with_system = provider._format_messages(
        [ChatMessage(role="system", content="caller"), ChatMessage(role="user", content="hi")]
    )

    assert len(_system_messages(no_system)) == 1
    assert len(_system_messages(with_system)) == 1


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
