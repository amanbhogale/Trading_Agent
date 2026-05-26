# llm_service.py
import os
import click
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from typing import List, Optional, Any, Dict
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """
    Configuration for the LLM service.
    
    Attributes:
        model     : model identifier e.g. "gemini-2.5-flash"
        api_key   : provider API key
        provider  : langchain provider name          (default: openrouter)
        temperature: sampling temperature             (default: 0.0)
        max_tokens : maximum tokens in response       (default: 1024)
        base_url  : override endpoint for the provider
    """
    model       : str
    api_key     : str
    provider    : str            = "openrouter"          # langchain provider tag
    temperature : float          = 0.0
    max_tokens  : int            = 1024
    base_url    : Optional[str]  = "https://openrouter.ai/api/v1"

    # ---- quick factory helpers ----------------------------------------
    @classmethod
    def for_openrouter(cls, model: str, api_key: str, **kwargs) -> "LLMConfig":
        return cls(
            model    = model,
            api_key  = api_key,
            provider = "openrouter",          # openrouter is openai-compatible
            base_url = "https://openrouter.ai/api/v1",
            **kwargs,
        )

    @classmethod
    def for_openai(cls, model: str, api_key: str, **kwargs) -> "LLMConfig":
        return cls(
            model    = model,
            api_key  = api_key,
            provider = "openai",
            base_url = None,              # use SDK default
            **kwargs,
        )

    @classmethod
    def for_anthropic(cls, model: str, api_key: str, **kwargs) -> "LLMConfig":
        return cls(
            model    = model,
            api_key  = api_key,
            provider = "anthropic",
            base_url = None,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class LLMService:
    """
    Thin, reusable wrapper around any LangChain-supported chat model.

    Usage
    -----
        config  = LLMConfig.for_openrouter("google/gemini-2.5-flash", api_key)
        service = LLMService(config)

        reply = service.invoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello!"),
        ])
        print(reply.content)
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = self._build_client()

    # ---- private ----------------------------------------------------------

    def _build_client(self):
        """Initialise the underlying LangChain chat model."""
        init_kwargs: Dict[str, Any] = dict(
            model_provider = self.config.provider,
            api_key        = self.config.api_key,
            temperature    = self.config.temperature,
            max_tokens     = self.config.max_tokens,
        )
        # only pass base_url when explicitly set
        if self.config.base_url:
            init_kwargs["base_url"] = self.config.base_url

        logger.info(
            "Initialising model=%s  provider=%s  base_url=%s",
            self.config.model, self.config.provider, self.config.base_url,
        )
        return init_chat_model(self.config.model, **init_kwargs)

    # ---- public API -------------------------------------------------------

    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        Send a list of messages and return the model's reply.

        Parameters
        ----------
        messages : list of BaseMessage
            Ordered conversation history (System → Human → AI → …).

        Returns
        -------
        BaseMessage
            The model's response message.
        """
        if not messages:
            raise ValueError("messages list must not be empty")

        logger.debug("Invoking model with %d message(s)", len(messages))
        response: BaseMessage = self._client.invoke(messages)
        logger.debug("Response received: %s", response.content[:80])
        return response

    def invoke_with_system(
        self,
        system_prompt: str,
        user_message: str,
    ) -> str:
        """
        Convenience wrapper – pass raw strings, get a string back.

        Parameters
        ----------
        system_prompt : str
        user_message  : str

        Returns
        -------
        str  – the model's reply text
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        return self.invoke(messages).content

    def with_tools(self, tools: list) -> "LLMService":
        """
        Return a *new* LLMService whose client has the given tools bound.

        Parameters
        ----------
        tools : list of @tool-decorated functions

        Returns
        -------
        LLMService  (new instance, original is unchanged)
        """
        new_service = LLMService.__new__(LLMService)
        new_service.config  = self.config
        new_service._client = self._client.bind_tools(tools)
        return new_service

    # ---- dunder -----------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LLMService(model={self.config.model!r}, "
            f"provider={self.config.provider!r})"
        )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

@click.command()
@click.option("--model",    "-m", default=None, help="Model identifier")
@click.option("--api-key",  "-k", default=None, help="Provider API key")
@click.option("--base-url", "-u", default=None, help="Custom base URL")
@click.option("--system",   "-s",
              default="You are a helpful assistant.",
              show_default=True,
              help="System prompt")
def main(model, api_key, base_url, system):
    """Interactive single-turn chat via the LLM service."""

    # fall back to interactive prompts when flags not supplied
    model   = model   or click.prompt("Model (e.g. google/gemini-2.5-flash)")
    api_key = api_key or click.prompt("API key", hide_input=True)

    if base_url is None:
        base_url = click.prompt(
            "Base URL (enter to use OpenRouter default)",
            default="https://openrouter.ai/api/v1",
        )

    config  = LLMConfig.for_openrouter(model, api_key) if base_url else \
              LLMConfig(model=model, api_key=api_key, base_url=base_url)

    service = LLMService(config)
    click.echo(f"\n✅  Connected → {service}\n")

    user_input = click.prompt("Your message")

    reply = service.invoke_with_system(system, user_input)
    click.echo(f"\n🤖  {reply}\n")


if __name__ == "__main__":
    main()
