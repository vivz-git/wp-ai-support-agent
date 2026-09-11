import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.llm.base import (
    ChatMessage,
    LLMProviderError,
    LLMResponse,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from app.llm.groq_provider import GroqProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call_mock(call_id: str, name: str, arguments: str) -> MagicMock:
    tool_call = MagicMock()
    tool_call.id = call_id
    tool_call.function = MagicMock()
    tool_call.function.name = name
    tool_call.function.arguments = arguments
    return tool_call


def _make_completion_mock(
    content=None,
    tool_calls=None,
    finish_reason="stop",
    reasoning=None,
    usage=(12, 34, 46),
):
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    message.reasoning = reasoning

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    completion = MagicMock()
    completion.choices = [choice]

    if usage is not None:
        usage_obj = MagicMock()
        usage_obj.prompt_tokens, usage_obj.completion_tokens, usage_obj.total_tokens = usage
        completion.usage = usage_obj
    else:
        completion.usage = None

    return completion


def _provider(mock_client) -> GroqProvider:
    return GroqProvider(api_key="test_groq_api_key", model="openai/gpt-oss-120b", client=mock_client)


PRODUCT_TOOL = ToolSpec(
    name="product_lookup",
    description="Look up products in the catalog.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": [],
    },
)


# ---------------------------------------------------------------------------
# 1. ToolSpec serialization
# ---------------------------------------------------------------------------


def test_tool_spec_serializes_to_openai_schema():
    schema = PRODUCT_TOOL.to_openai_tool()
    assert schema == {
        "type": "function",
        "function": {
            "name": "product_lookup",
            "description": "Look up products in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": [],
            },
        },
    }


# ---------------------------------------------------------------------------
# 2 & 3. Valid Groq tool-call response parsing (single and multiple)
# ---------------------------------------------------------------------------


def test_complete_parses_single_tool_call():
    tool_call = _make_tool_call_mock("call_abc123", "product_lookup", '{"query": "dark roast"}')
    completion = _make_completion_mock(content=None, tool_calls=[tool_call], finish_reason="tool_calls")

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)
    provider = _provider(mock_client)

    response = asyncio.run(provider.complete([ChatMessage(role="user", content="Any dark roast?")], tools=[PRODUCT_TOOL]))

    assert isinstance(response, LLMResponse)
    assert response.finish_reason == "tool_calls"
    assert response.content is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_abc123"
    assert response.tool_calls[0].name == "product_lookup"


def test_complete_parses_multiple_tool_calls():
    call_1 = _make_tool_call_mock("call_1", "product_lookup", '{"query": "espresso"}')
    call_2 = _make_tool_call_mock("call_2", "product_lookup", '{"query": "decaf"}')
    completion = _make_completion_mock(content=None, tool_calls=[call_1, call_2], finish_reason="tool_calls")

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)
    provider = _provider(mock_client)

    response = asyncio.run(provider.complete([ChatMessage(role="user", content="Two things")], tools=[PRODUCT_TOOL]))

    assert len(response.tool_calls) == 2
    assert [tc.id for tc in response.tool_calls] == ["call_1", "call_2"]
    assert [tc.name for tc in response.tool_calls] == ["product_lookup", "product_lookup"]


# ---------------------------------------------------------------------------
# 4 & 5. Parsed arguments + raw argument preservation
# ---------------------------------------------------------------------------


def test_tool_call_exposes_parsed_and_raw_arguments():
    raw = '{"query": "dark roast", "limit": 3}'
    tool_call = ToolCall.from_raw_arguments(id="call_1", name="product_lookup", raw_arguments=raw)

    assert tool_call.raw_arguments == raw
    assert tool_call.arguments == {"query": "dark roast", "limit": 3}
    assert tool_call.parse_error is None
    assert tool_call.is_valid is True


# ---------------------------------------------------------------------------
# 6. Malformed JSON arguments
# ---------------------------------------------------------------------------


def test_tool_call_malformed_json_produces_explicit_parse_error():
    raw = '{"query": "dark roast",'  # truncated / invalid JSON
    tool_call = ToolCall.from_raw_arguments(id="call_1", name="product_lookup", raw_arguments=raw)

    assert tool_call.arguments is None
    assert tool_call.is_valid is False
    assert tool_call.parse_error is not None
    assert tool_call.raw_arguments == raw


def test_tool_call_non_object_json_is_treated_as_malformed():
    raw = '["not", "an", "object"]'
    tool_call = ToolCall.from_raw_arguments(id="call_1", name="product_lookup", raw_arguments=raw)

    assert tool_call.arguments is None
    assert tool_call.is_valid is False
    assert "object" in tool_call.parse_error


def test_complete_surfaces_malformed_tool_call_arguments_without_raising():
    tool_call = _make_tool_call_mock("call_1", "product_lookup", "{not valid json")
    completion = _make_completion_mock(content=None, tool_calls=[tool_call], finish_reason="tool_calls")

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)
    provider = _provider(mock_client)

    response = asyncio.run(provider.complete([ChatMessage(role="user", content="Hi")], tools=[PRODUCT_TOOL]))

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].is_valid is False
    assert response.tool_calls[0].arguments is None


# ---------------------------------------------------------------------------
# 7 & 8. finish_reason handling
# ---------------------------------------------------------------------------


def test_complete_finish_reason_tool_calls():
    tool_call = _make_tool_call_mock("call_1", "product_lookup", "{}")
    completion = _make_completion_mock(content=None, tool_calls=[tool_call], finish_reason="tool_calls")

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)
    provider = _provider(mock_client)

    response = asyncio.run(provider.complete([ChatMessage(role="user", content="Hi")], tools=[PRODUCT_TOOL]))
    assert response.finish_reason == "tool_calls"
    assert response.has_tool_calls is True


def test_complete_finish_reason_stop():
    completion = _make_completion_mock(content="Hello there!", tool_calls=None, finish_reason="stop")

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)
    provider = _provider(mock_client)

    response = asyncio.run(provider.complete([ChatMessage(role="user", content="Hi")]))
    assert response.finish_reason == "stop"
    assert response.content == "Hello there!"
    assert response.has_tool_calls is False
    assert response.tool_calls == []


# ---------------------------------------------------------------------------
# 9. Reasoning extraction
# ---------------------------------------------------------------------------


def test_complete_extracts_reasoning_separately_from_content():
    completion = _make_completion_mock(
        content="The answer is 4.",
        tool_calls=None,
        finish_reason="stop",
        reasoning="2 + 2 = 4, a basic arithmetic fact.",
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)
    provider = _provider(mock_client)

    response = asyncio.run(provider.complete([ChatMessage(role="user", content="What is 2+2?")]))

    assert response.content == "The answer is 4."
    assert response.reasoning == "2 + 2 = 4, a basic arithmetic fact."
    assert "arithmetic" not in (response.content or "")


# ---------------------------------------------------------------------------
# 10. Usage extraction
# ---------------------------------------------------------------------------


def test_complete_extracts_usage():
    completion = _make_completion_mock(content="Hi", tool_calls=None, finish_reason="stop", usage=(10, 20, 30))

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)
    provider = _provider(mock_client)

    response = asyncio.run(provider.complete([ChatMessage(role="user", content="Hi")]))

    assert response.usage == TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)


def test_complete_usage_absent_is_none():
    completion = _make_completion_mock(content="Hi", tool_calls=None, finish_reason="stop", usage=None)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)
    provider = _provider(mock_client)

    response = asyncio.run(provider.complete([ChatMessage(role="user", content="Hi")]))
    assert response.usage is None


# ---------------------------------------------------------------------------
# 11 & 12. tools included/omitted in the request
# ---------------------------------------------------------------------------


def test_complete_includes_tools_when_supplied():
    completion = _make_completion_mock(content="Hi", tool_calls=None, finish_reason="stop")
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)
    provider = _provider(mock_client)

    asyncio.run(provider.complete([ChatMessage(role="user", content="Hi")], tools=[PRODUCT_TOOL]))

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "tools" in kwargs
    assert kwargs["tools"] == [PRODUCT_TOOL.to_openai_tool()]


def test_complete_omits_tools_when_not_supplied():
    completion = _make_completion_mock(content="Hi", tool_calls=None, finish_reason="stop")
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)
    provider = _provider(mock_client)

    asyncio.run(provider.complete([ChatMessage(role="user", content="Hi")]))

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


# ---------------------------------------------------------------------------
# 13, 14, 15. tool_choice variants
# ---------------------------------------------------------------------------


def test_complete_tool_choice_auto():
    completion = _make_completion_mock(content="Hi", tool_calls=None, finish_reason="stop")
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)
    provider = _provider(mock_client)

    asyncio.run(provider.complete([ChatMessage(role="user", content="Hi")], tools=[PRODUCT_TOOL], tool_choice="auto"))

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["tool_choice"] == "auto"


def test_complete_tool_choice_required():
    completion = _make_completion_mock(content=None, tool_calls=None, finish_reason="stop")
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)
    provider = _provider(mock_client)

    asyncio.run(
        provider.complete([ChatMessage(role="user", content="Hi")], tools=[PRODUCT_TOOL], tool_choice="required")
    )

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["tool_choice"] == "required"


def test_complete_tool_choice_explicit_function_name():
    completion = _make_completion_mock(content=None, tool_calls=None, finish_reason="stop")
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)
    provider = _provider(mock_client)

    asyncio.run(
        provider.complete(
            [ChatMessage(role="user", content="Hi")], tools=[PRODUCT_TOOL], tool_choice="product_lookup"
        )
    )

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "product_lookup"}}


def test_complete_tool_choice_explicit_dict_passthrough():
    completion = _make_completion_mock(content=None, tool_calls=None, finish_reason="stop")
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)
    provider = _provider(mock_client)

    explicit_choice = {"type": "function", "function": {"name": "product_lookup"}}
    asyncio.run(
        provider.complete([ChatMessage(role="user", content="Hi")], tools=[PRODUCT_TOOL], tool_choice=explicit_choice)
    )

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["tool_choice"] == explicit_choice


# ---------------------------------------------------------------------------
# 16. tool_choice="none" is NOT sent
# ---------------------------------------------------------------------------


def test_complete_tool_choice_none_is_rejected_not_sent():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    provider = _provider(mock_client)

    with pytest.raises(LLMProviderError) as exc_info:
        asyncio.run(
            provider.complete([ChatMessage(role="user", content="Hi")], tools=[PRODUCT_TOOL], tool_choice="none")
        )

    assert "none" in str(exc_info.value)
    mock_client.chat.completions.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# 17. Provider/API error handling
# ---------------------------------------------------------------------------


def test_complete_wraps_provider_exception_in_llm_provider_error():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("Groq service unavailable"))
    provider = _provider(mock_client)

    with pytest.raises(LLMProviderError) as exc_info:
        asyncio.run(provider.complete([ChatMessage(role="user", content="Hi")]))

    assert "Groq API error" in str(exc_info.value)
    assert "test_groq_api_key" not in str(exc_info.value)


def test_complete_empty_choices_raises_llm_provider_error():
    completion = MagicMock()
    completion.choices = []
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)
    provider = _provider(mock_client)

    with pytest.raises(LLMProviderError):
        asyncio.run(provider.complete([ChatMessage(role="user", content="Hi")]))


# ---------------------------------------------------------------------------
# 18. Existing get_agent_reply compatibility
# ---------------------------------------------------------------------------


def test_get_agent_reply_unaffected_by_new_complete_method():
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Simple reply."
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

    provider = _provider(mock_client)
    reply = asyncio.run(provider.get_agent_reply([ChatMessage(role="user", content="Hi")]))

    assert reply == "Simple reply."
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


# ---------------------------------------------------------------------------
# 19. Deterministic response normalization
# ---------------------------------------------------------------------------


def test_complete_normalizes_missing_tool_calls_to_empty_list():
    completion = _make_completion_mock(content="Hi", tool_calls=None, finish_reason="stop")
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)
    provider = _provider(mock_client)

    response = asyncio.run(provider.complete([ChatMessage(role="user", content="Hi")]))
    assert response.tool_calls == []
    assert isinstance(response.tool_calls, list)


def test_complete_preserves_none_content_during_tool_call():
    tool_call = _make_tool_call_mock("call_1", "product_lookup", "{}")
    completion = _make_completion_mock(content=None, tool_calls=[tool_call], finish_reason="tool_calls")
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)
    provider = _provider(mock_client)

    response = asyncio.run(provider.complete([ChatMessage(role="user", content="Hi")], tools=[PRODUCT_TOOL]))
    assert response.content is None


# ---------------------------------------------------------------------------
# 20. No secrets in serialized ToolSpec
# ---------------------------------------------------------------------------


def test_tool_spec_serialization_contains_no_secrets():
    schema = PRODUCT_TOOL.to_openai_tool()
    serialized = str(schema)
    assert "api_key" not in serialized.lower()
    assert "test_groq_api_key" not in serialized
    assert set(schema.keys()) == {"type", "function"}
    assert set(schema["function"].keys()) == {"name", "description", "parameters"}
