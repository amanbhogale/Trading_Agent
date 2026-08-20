"""
Execution Agent Microservice
"""
import os, sys, logging
from typing import Optional, Dict

sys.path.insert(0, "/app")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("execution-service")

app = FastAPI(title="Execution Agent Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_agent = None

def get_agent():
    global _agent
    if _agent is None:
        from src.trading_system.llm_service import LLMService, LLMConfig
        from src.trading_system.sub_agents import ExecutionAgent
        config = LLMConfig(
            model=os.getenv("LLM_MODEL", "google/gemini-2.5-pro"),
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            provider="openai",
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=0.0,
            max_tokens=4096,
        )
        _agent = ExecutionAgent(LLMService(config))
        logger.info("ExecutionAgent initialized ✅")
    return _agent


class RunRequest(BaseModel):
    message: str
    context: Optional[str] = None


class TradeRequest(BaseModel):
    order: Dict


class RunResponse(BaseModel):
    result: str
    service: str = "execution"


@app.get("/health")
def health():
    return {"status": "ok", "service": "execution"}


@app.post("/run", response_model=RunResponse)
def run(req: RunRequest):
    """General agent conversation endpoint."""
    try:
        agent = get_agent()
        result = agent.run(req.message)
        return RunResponse(result=result)
    except Exception as e:
        logger.exception("Error running ExecutionAgent")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute-with-approval", response_model=RunResponse)
def execute_with_approval(req: TradeRequest):
    """Execute a trade order (requires external approval gate)."""
    try:
        agent = get_agent()
        result = agent.execute_with_approval(req.order)
        return RunResponse(result=result)
    except Exception as e:
        logger.exception("Error in execute_with_approval")
        raise HTTPException(status_code=500, detail=str(e))
