import logging
from typing import Any, Dict, List, Optional
from groq import AsyncGroq
from app.llm.base import ChatMessage, LLMProviderError

logger = logging.getLogger(__name__)


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

        formatted_messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        for msg in messages:
            formatted_messages.append({"role": msg.role, "content": msg.content})

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
