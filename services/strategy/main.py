"""
Strategy Agent Microservice
"""
import os, sys, logging
from typing import Optional

sys.path.insert(0, "/app")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("strategy-service")

app = FastAPI(title="Strategy Agent Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_agent = None

def get_agent():
    global _agent
    if _agent is None:
        from src.trading_system.llm_service import LLMService, LLMConfig
        from src.trading_system.sub_agents import StrategyAgent
        config = LLMConfig(
            model=os.getenv("LLM_MODEL", "google/gemini-2.5-pro"),
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            provider="openai",
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=0.0,
            max_tokens=4096,
        )
        _agent = StrategyAgent(LLMService(config))
        logger.info("StrategyAgent initialized ✅")
    return _agent


class RunRequest(BaseModel):
    message: str
    context: Optional[str] = None

class RunResponse(BaseModel):
    result: str
    service: str = "strategy"


@app.get("/health")
def health():
    return {"status": "ok", "service": "strategy"}


@app.post("/run", response_model=RunResponse)
def run(req: RunRequest):
    try:
        agent = get_agent()
        msg = req.message
        if req.context:
            msg = f"{req.message}\n\n--- Context from prior agents ---\n{req.context}"
        result = agent.run(msg)
        return RunResponse(result=result)
    except Exception as e:
        logger.exception("Error running StrategyAgent")
        raise HTTPException(status_code=500, detail=str(e))
