import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Represents a conversational message in a chat history."""
    role: str = Field(..., description="Role of the speaker: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message text content")


class LLMProviderError(Exception):
    """Exception raised for LLM provider failures."""
    pass


# ---------------------------------------------------------------------------
# Native tool-calling types
#
# These describe the provider-agnostic shape of native function/tool calling
# (Groq/OpenAI-compatible). They are intentionally separate from
# ``app.tools.registry.ToolSpec``, which binds a tool to a Pydantic input
# model and a local Python handler for execution — that is a tool-execution
# concern. The types here only describe what an LLM request/response looks
# like and carry no handler or execution logic.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """A tool description ready to hand to a provider's native tool-calling API.

    Attributes:
        name: Function name presented to the model.
        description: Model-facing description of what the tool does.
        parameters: JSON Schema object describing the tool's arguments.
    """

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_openai_tool(self) -> Dict[str, Any]:
        """Serialize to the Groq/OpenAI-compatible native tool schema.

        Shape: ``{"type": "function", "function": {"name", "description",
        "parameters"}}``. Contains only the public argument contract — no
        secrets or implementation details.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the model.

    Tool-call arguments arrive from the provider as a raw JSON string, which
    is not guaranteed to be valid JSON or to decode to an object. Rather than
    letting a decode failure raise (and silently coercing it into an empty or
    partial dict), this type preserves the raw string alongside the parsed
    result and an explicit ``parse_error`` so a future orchestrator can
    detect and handle malformed tool calls deterministically.

    Attributes:
        id: Provider-assigned tool call ID (needed to correlate a later
            ``role="tool"`` result message with this call).
        name: Name of the function the model wants to call.
        raw_arguments: The unmodified JSON string as sent by the provider.
        arguments: The parsed arguments dict, or ``None`` if parsing failed.
        parse_error: ``None`` if ``raw_arguments`` decoded to a JSON object;
            otherwise a short, explicit description of why it did not.
    """

    id: str
    name: str
    raw_arguments: str
    arguments: Optional[Dict[str, Any]]
    parse_error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Whether ``raw_arguments`` decoded successfully to a JSON object."""
        return self.parse_error is None

    @classmethod
    def from_raw_arguments(cls, id: str, name: str, raw_arguments: str) -> "ToolCall":
        """Build a ``ToolCall``, safely parsing ``raw_arguments`` as JSON.

        Never raises: a JSON decode failure, or JSON that decodes to
        something other than an object (e.g. a list or scalar), produces a
        ``ToolCall`` with ``arguments=None`` and an explicit ``parse_error``
        instead of an exception or a fabricated dict.
        """
        try:
            parsed = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError) as exc:
            return cls(
                id=id,
                name=name,
                raw_arguments=raw_arguments,
                arguments=None,
                parse_error=f"Malformed tool-call arguments JSON: {exc}",
            )
        if not isinstance(parsed, dict):
            return cls(
                id=id,
                name=name,
                raw_arguments=raw_arguments,
                arguments=None,
                parse_error=(
                    f"Tool-call arguments JSON decoded to {type(parsed).__name__}, expected an object"
                ),
            )
        return cls(id=id, name=name, raw_arguments=raw_arguments, arguments=parsed, parse_error=None)


@dataclass(frozen=True)
class TokenUsage:
    """Token accounting for a single completion, as reported by the provider."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    """Normalized result of a native ``complete()`` call.

    Attributes:
        content: The assistant's text reply, or ``None`` when the model
            issued tool calls instead of (or in addition to producing no)
            text content. Never mixed with ``reasoning``.
        tool_calls: Tool calls requested by the model, in the order the
            provider returned them. Empty when the model did not call a tool.
        finish_reason: The provider's finish reason, e.g. ``"stop"`` or
            ``"tool_calls"``.
        reasoning: The provider's separate reasoning trace, if present.
        usage: Token usage for this completion, if the provider reported it.
    """

    content: Optional[str]
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    reasoning: Optional[str] = None
    usage: Optional[TokenUsage] = None

    @property
    def has_tool_calls(self) -> bool:
        """Whether the model requested one or more tool calls."""
        return len(self.tool_calls) > 0


# A tool_choice value as accepted by ``LLMProvider.complete``: ``"auto"``,
# ``"required"``, a bare function name (shorthand for explicit selection), or
# an already-built Groq/OpenAI-compatible named-tool-choice dict. ``"none"``
# is deliberately unsupported — see ``groq_provider`` for why.
ToolChoice = Union[str, Dict[str, Any]]


@runtime_checkable
class LLMProvider(Protocol):
    """Abstract protocol for an LLM runtime provider."""

    async def get_agent_reply(self, messages: List[ChatMessage]) -> str:
        """Generate an agent reply given a conversation history.

        Args:
            messages: Ordered list of prior messages in the conversation.

        Returns:
            The textual assistant reply.

        Raises:
            LLMProviderError: If the underlying model call fails.
        """
        ...

    async def complete(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> LLMResponse:
        """Generate a completion, optionally with native tool-calling.

        Args:
            messages: Ordered list of prior messages in the conversation.
            tools: Tools the model may call. When omitted, no ``tools``
                parameter is sent to the provider at all.
            tool_choice: How the model should use ``tools``: ``"auto"``,
                ``"required"``, a function name, or an explicit named-tool
                dict. ``"none"`` is not supported and must not be passed.

        Returns:
            A normalized ``LLMResponse`` describing the provider's reply.

        Raises:
            LLMProviderError: If the underlying model call fails, or if an
                unsupported ``tool_choice`` is requested.
        """
        ...


async def get_agent_reply(messages: List[ChatMessage], provider: LLMProvider) -> str:
    """High-level abstraction for generating an agent reply.
    
    Args:
        messages: Conversation messages to pass to the model.
        provider: An instance satisfying the LLMProvider protocol.
        
    Returns:
        The text response generated by the provider.
    """
    return await provider.get_agent_reply(messages)
