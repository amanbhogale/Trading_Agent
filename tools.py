from types import TracebackType
import os
from typing import Literal , Callable , Any
from langchain.agents.structured_output import ResponseFormat
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import content
from langgraph.pregel.debug import map_debug_checkpoint
from langgraph.store.base import Result
from pydantic import KafkaDsn
from requests import get
from tavily import TavilyClient
from deepagents import create_deep_agent



travily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

print(f"{travily_client}")


def internet_search(
    query: str,
    max_results : int = 5,
    topic : Literal["general", "news","finance"] = "general",
    include_raw_content : bool = False,
):
    response = travily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )
    "Run a web search"
    return {
        "query": query,
        "answer" : response.get("answer"),
        "results": [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "content": response.get("content"),
                "score":r.get("score"),
            }
            for r in response.get("results" , [])

            ]
    }
""" here the results are not proper as expectd, 
 but the solution here would be to use to use two stage respponse format run search() keep only
 the high score results
 call extract() on the those urls

"""
results = internet_search("what is the current stock price and makets sentiemnt for $AAPL" , topic="finance",   include_raw_content=True)
print(results)
