import logging
from typing import Any, Dict, List, Optional
from groq import AsyncGroq
from app.llm.base import (
    ChatMessage,
    LLMProviderError,
    LLMResponse,
    TokenUsage,
    ToolCall,
    ToolChoice,
    ToolSpec,
)

logger = logging.getLogger(__name__)

# Groq's tool_choice="none" is confirmed unreliable (Slice 0 provider spike)
# and must never be sent to the API.
_UNSUPPORTED_TOOL_CHOICE = "none"


def _resolve_tool_choice(tool_choice: Optional[ToolChoice]) -> Optional[ToolChoice]:
    """Validate and normalize a ``tool_choice`` value for the Groq API.

    Supports ``"auto"``, ``"required"``, a bare function name (normalized to
    the explicit named-tool-choice dict), and an already-built named-tool
    dict. Deliberately rejects ``"none"`` instead of sending it to Groq,
    since it is confirmed unreliable there.
    """
    if tool_choice is None:
        return None
    if tool_choice == _UNSUPPORTED_TOOL_CHOICE:
        raise LLMProviderError(
            "tool_choice='none' is unreliable on Groq and must not be sent. "
            "Omit tool_choice (and tools) instead."
        )
    if isinstance(tool_choice, str):
        if tool_choice in ("auto", "required"):
            return tool_choice
        return {"type": "function", "function": {"name": tool_choice}}
    if isinstance(tool_choice, dict):
        return tool_choice
    raise LLMProviderError(f"Unsupported tool_choice value: {tool_choice!r}")


class GroqProvider:
    """LLM provider implementation for Groq Cloud API."""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        system_prompt: str = (
            "You are a professional, helpful, and polite customer support AI agent for WhatsApp. "
            "Keep your answers concise, clear, and direct, suitable for WhatsApp messaging."
        ),
        client: Optional[Any] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self._client = client

    def _get_client(self) -> AsyncGroq:
        if self._client is not None:
            return self._client
        if not self.api_key or not self.api_key.strip():
            raise LLMProviderError("GROQ_API_KEY is not configured or is empty")
        self._client = AsyncGroq(api_key=self.api_key)
        return self._client

    def _format_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        formatted: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        for msg in messages:
            formatted.append({"role": msg.role, "content": msg.content})
        return formatted

    async def get_agent_reply(self, messages: List[ChatMessage]) -> str:
        """Generate an agent reply using the Groq API.

        Args:
            messages: List of previous chat messages.

        Returns:
            The text response from the Groq model.

        Raises:
            LLMProviderError: If the Groq client or API fails.
        """
        client = self._get_client()
        formatted_messages = self._format_messages(messages)

        try:
            logger.info("Calling Groq LLM with model: %s", self.model)
            response = await client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=0.7,
                max_tokens=500,
            )

            if not response.choices or not response.choices[0].message:
                raise LLMProviderError("Groq returned an empty response choices list")

            content = response.choices[0].message.content or ""
            return content.strip()
        except LLMProviderError:
            raise
        except Exception as exc:
            # Never log the API key or raw auth details
            logger.error("Groq completion failed with error: %s", type(exc).__name__)
            raise LLMProviderError(f"Groq API error ({type(exc).__name__}): {str(exc)}") from exc

    async def complete(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> LLMResponse:
        """Generate a completion via the Groq API, with optional native tool-calling.

        Args:
            messages: List of previous chat messages.
            tools: Tools the model may call. Omitted from the request
                entirely when ``None`` or empty.
            tool_choice: ``"auto"``, ``"required"``, a function name, or an
                explicit named-tool dict. ``"none"`` is rejected rather than
                sent to Groq (confirmed unreliable there).

        Returns:
            A normalized ``LLMResponse``.

        Raises:
            LLMProviderError: If the Groq client/API fails, or if an
                unsupported ``tool_choice`` is requested.
        """
        resolved_tool_choice = _resolve_tool_choice(tool_choice)

        client = self._get_client()
        formatted_messages = self._format_messages(messages)

        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": 0.7,
            "max_tokens": 500,
        }
        if tools:
            request_kwargs["tools"] = [tool.to_openai_tool() for tool in tools]
            if resolved_tool_choice is not None:
                request_kwargs["tool_choice"] = resolved_tool_choice

        try:
            logger.info("Calling Groq LLM (complete) with model: %s", self.model)
            response = await client.chat.completions.create(**request_kwargs)

            if not response.choices or not response.choices[0].message:
                raise LLMProviderError("Groq returned an empty response choices list")

            choice = response.choices[0]
            message = choice.message

            raw_tool_calls = getattr(message, "tool_calls", None) or []
            tool_calls = [
                ToolCall.from_raw_arguments(
                    id=tc.id,
                    name=tc.function.name,
                    raw_arguments=tc.function.arguments,
                )
                for tc in raw_tool_calls
            ]

            usage_obj = getattr(response, "usage", None)
            usage = (
                TokenUsage(
                    prompt_tokens=usage_obj.prompt_tokens,
                    completion_tokens=usage_obj.completion_tokens,
                    total_tokens=usage_obj.total_tokens,
                )
                if usage_obj is not None
                else None
            )

            return LLMResponse(
                content=message.content,
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason,
                reasoning=getattr(message, "reasoning", None),
                usage=usage,
            )
        except LLMProviderError:
            raise
        except Exception as exc:
            # Never log the API key or raw auth details
            logger.error("Groq completion failed with error: %s", type(exc).__name__)
            raise LLMProviderError(f"Groq API error ({type(exc).__name__}): {str(exc)}") from exc
