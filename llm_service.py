# llm reusablle service for llm related operations```
import os
import click 
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage , SystemMessage , BaseMessage
from typing import List 
from dataclasses import dataclass 

@dataclass
class LLMConfig:
    model : str
    api_key : str
    provider : str = "openrouter"
    temperature : float = 0.0
    max_tokens : int = 1024
    base_url : Optional[str] = "https://openrouter.ai/api/v1"

class LLMService:
    def __init__(self , config : LLMConfig):
        self.config = config
        self.client = init_chat_model(
                config.model,
                model_provider=config.provider,
                api_key=config.api_key,
                temperature=config.temperature,
                base_url=config.base_url,
            )
    def invoke(self , messages : List[BaseMessage]) -> BaseMessage:
        response = self.client.invoke(messages)
        return response
def main():
    model = click.prompt("Enter the LLM model (e.g., gemini-2.5-flash, gpt-4): ")
    api_key = click.prompt("Enter your API key: ", hide_input=True)
    base_url = click.prompt("Enter the base URL (optional, press enter to skip): ", default="")

    config = LLMConfig(model=model, api_key=api_key, base_url=base_url or None)
    llm_service = LLMService(config)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        #human message through click
        HumanMessage(content=click.prompt("Enter your message: "))

    ]
    response = llm_service.invoke(messages)
    click.echo(f"LLM Response: {response.content}")


if __name__ == "__main__":
    main()

