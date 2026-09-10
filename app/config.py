import os
import logging
from functools import lru_cache
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def mask_phone_number(phone: str) -> str:
    """Mask phone number to prevent logging PII.
    
    Example: 919876543210 -> ********3210
    """
    if not phone:
        return "[none]"
    cleaned = phone.strip()
    if len(cleaned) <= 4:
        return "****"
    return f"{'*' * (len(cleaned) - 4)}{cleaned[-4:]}"


def mask_secret(secret: str) -> str:
    """Mask secret tokens to prevent logging sensitive credentials."""
    if not secret:
        return "[not set]"
    if len(secret) <= 6:
        return "******"
    return f"{secret[:3]}...{secret[-3:]}"


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Meta WhatsApp Cloud API credentials
    whatsapp_access_token: str = Field(
        default="",
        alias="WHATSAPP_ACCESS_TOKEN",
        description="Meta WhatsApp Cloud API access token",
    )
    whatsapp_phone_number_id: str = Field(
        default="",
        alias="WHATSAPP_PHONE_NUMBER_ID",
        description="Meta Phone Number ID from Meta App Dashboard",
    )
    whatsapp_verify_token: str = Field(
        default="",
        alias="WHATSAPP_VERIFY_TOKEN",
        description="Custom verification token used during Meta webhook handshake",
    )
    whatsapp_api_version: str = Field(
        default="v22.0",
        alias="WHATSAPP_API_VERSION",
        description="Meta Graph API version",
    )

    # Groq LLM Configuration
    groq_api_key: str = Field(
        default="",
        alias="GROQ_API_KEY",
        description="API key for Groq Cloud",
    )
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        alias="GROQ_MODEL",
        description="Groq model ID to use for conversation replies",
    )

    # Agent / Memory Configuration
    system_prompt: str = Field(
        default=(
            "You are a professional, helpful, and polite customer support AI agent for WhatsApp. "
            "Keep your answers concise, clear, and direct, suitable for WhatsApp messaging. "
            "Do not output markdown tables or complex formatting. If you cannot help, offer polite assistance."
        ),
        alias="SYSTEM_PROMPT",
        description="Base system prompt guiding the AI assistant",
    )
    max_memory_messages: int = Field(
        default=10,
        alias="MAX_MEMORY_MESSAGES",
        description="Maximum recent messages to retain in memory per user",
    )

    # Server Configuration
    app_port: int = Field(
        default=8000,
        alias="APP_PORT",
        description="Port to bind the FastAPI application",
    )
    app_host: str = Field(
        default="0.0.0.0",
        alias="APP_HOST",
        description="Host to bind the FastAPI application",
    )

    def validate_required_credentials(self) -> None:
        """Validate that all essential credentials are configured.
        
        Raises:
            ValueError: If any required environment variable is missing or empty.
        """
        missing: List[str] = []
        if not self.whatsapp_access_token.strip():
            missing.append("WHATSAPP_ACCESS_TOKEN")
        if not self.whatsapp_phone_number_id.strip():
            missing.append("WHATSAPP_PHONE_NUMBER_ID")
        if not self.whatsapp_verify_token.strip():
            missing.append("WHATSAPP_VERIFY_TOKEN")
        if not self.groq_api_key.strip():
            missing.append("GROQ_API_KEY")

        if missing:
            error_msg = f"Missing required environment variables: {', '.join(missing)}"
            logger.error(error_msg)
            raise ValueError(error_msg)


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
