"""
Orchestrator Service
Acts as the central router. Calls individual agent services via HTTP,
mirrors the logic in main_agent.py but over the network.
"""
import os, sys, re, json, logging
from typing import Dict, Optional

sys.path.insert(0, "/app")

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator-service")

app = FastAPI(title="Orchestrator Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Service URLs (resolved via Docker internal DNS) ────────────────────────────
SERVICES = {
    "market_data":   os.getenv("MARKET_DATA_URL",   "http://market-data:8000"),
    "analysis":      os.getenv("ANALYSIS_URL",       "http://analysis:8000"),
    "strategy":      os.getenv("STRATEGY_URL",       "http://strategy:8000"),
    "visualization": os.getenv("VISUALIZATION_URL",  "http://visualization:8000"),
    "risk":          os.getenv("RISK_URL",           "http://risk:8000"),
    "execution":     os.getenv("EXECUTION_URL",      "http://execution:8000"),
}

# LLM used only for intent classification and synthesis
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        from src.trading_system.llm_service import LLMService, LLMConfig
        config = LLMConfig(
            model=os.getenv("LLM_MODEL", "google/gemini-2.5-pro"),
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            provider="openai",
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=0.0,
            max_tokens=4096,
        )
        _llm = LLMService(config)
        logger.info("Orchestrator LLM initialized ✅")
    return _llm


# ── Shepherd / Safety layer ────────────────────────────────────────────────────
_shepherd = None

def get_shepherd():
    global _shepherd
    if _shepherd is None:
        from src.trading_system.shepherd_layer import ShepherdSafetyLayer
        _shepherd = ShepherdSafetyLayer(workspace_root="/app")
    return _shepherd


# ── Helpers ────────────────────────────────────────────────────────────────────
def _classify_intent(user_msg: str) -> str:
    from langchain_core.messages import SystemMessage, HumanMessage
    prompt = f"""
Classify this trading request into one or more of:
market_data | analysis | strategy | visualization | risk | execution | general

Request: "{user_msg}"

Reply with comma-separated agent names only. Example: market_data,analysis
"""
    resp = get_llm().invoke([
        SystemMessage(content="You are an intent classifier for a trading system."),
        HumanMessage(content=prompt),
    ])
    return resp.content.strip().lower()


def _synthesize(user_msg: str, agent_results: Dict[str, str]) -> str:
    from langchain_core.messages import SystemMessage, HumanMessage
    context = "\n\n".join(f"[{agent}]\n{result}" for agent, result in agent_results.items())
    prompt = f"""
User asked: {user_msg}

Sub-agent results:
{context}

Provide a comprehensive, well-structured final answer.
If trade action is recommended, clearly state it requires human approval.
"""
    resp = get_llm().invoke([
        SystemMessage(content="You are a Master Trading Orchestrator synthesizing sub-agent results."),
        HumanMessage(content=prompt),
    ])
    return resp.content


def _call_service(service_name: str, message: str, context: str = "") -> str:
    url = SERVICES[service_name]
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(f"{url}/run", json={"message": message, "context": context})
            r.raise_for_status()
            return r.json()["result"]
    except httpx.HTTPError as e:
        logger.error("Service %s failed: %s", service_name, e)
        return f"⚠️ {service_name} service unavailable: {e}"


def _extract_asset(msg: str) -> str:
    matches = re.findall(r'\b([A-Z]{2,10})\b', msg)
    stopwords = {'THE', 'AND', 'FOR', 'WITH', 'THIS', 'THAT', 'FROM',
                 'BUY', 'SELL', 'HOLD', 'RSI', 'MACD', 'EMA', 'SMA'}
    tickers = [m for m in matches if m not in stopwords]
    return tickers[0] if tickers else "UNKNOWN"


def _extract_timeframe(msg: str) -> str:
    tf_match = re.search(r'\b(\d+[mhd]|\d+\s*(?:min|hour|day|week))\b', msg.lower())
    return tf_match.group(0) if tf_match else "1d"


# ── Request / Response models ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    result: str

class TradeRequest(BaseModel):
    order: Dict
    proposal_id: Optional[str] = None

class SettleRequest(BaseModel):
    accept: bool
    reason: str = ""


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "orchestrator"}


@app.get("/health/agents")
def health_agents():
    """Probe all downstream agent services."""
    results = {}
    with httpx.Client(timeout=5.0) as client:
        for name, url in SERVICES.items():
            try:
                r = client.get(f"{url}/health")
                results[name] = r.json()
            except Exception as e:
                results[name] = {"status": "unreachable", "error": str(e)}
    return results


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Main routing endpoint — classifies intent and calls appropriate agents."""
    user_message = req.message
    logger.info("Received message: %s", user_message[:100])

    intent = _classify_intent(user_message)
    logger.info("Intent: %s", intent)

    agent_results: Dict[str, str] = {}
    context_so_far = ""

    if "market_data" in intent:
        result = _call_service("market_data", user_message)
        agent_results["market_data"] = result
        context_so_far += f"\n[market_data]\n{result}\n"

    if "analysis" in intent:
        result = _call_service("analysis", user_message, context_so_far)
        agent_results["analysis"] = result
        context_so_far += f"\n[analysis]\n{result}\n"

    if "strategy" in intent:
        result = _call_service("strategy", user_message, context_so_far)
        agent_results["strategy"] = result
        context_so_far += f"\n[strategy]\n{result}\n"

        # Shepherd: retain as a reviewable proposal
        try:
            shepherd = get_shepherd()
            proposal = shepherd.propose_strategy(
                prompt=user_message,
                asset=_extract_asset(user_message),
                timeframe=_extract_timeframe(user_message),
                agent_result=result,
            )
            agent_results["shepherd_strategy"] = (
                f"📋 Strategy proposal **{proposal.proposal_id}** created "
                f"(status: {proposal.status.value}).\n"
                f"Settle via: `/api/shepherd/settle/strategy/{proposal.proposal_id}`"
            )
        except Exception as e:
            logger.warning("Shepherd strategy proposal failed: %s", e)

    if "visualization" in intent:
        agent_results["visualization"] = _call_service("visualization", user_message)

    if "risk" in intent:
        result = _call_service("risk", user_message, context_so_far)
        agent_results["risk"] = result
        context_so_far += f"\n[risk]\n{result}\n"

    if "execution" in intent:
        agent_results["execution"] = (
            "⚠️ Trade execution requires explicit human approval via the Execute Trade tab.\n"
            "🛡️ All trades flow through the Shepherd safety layer — "
            "orders are retained as proposals until explicitly settled."
        )

    if not agent_results:
        from langchain_core.messages import SystemMessage, HumanMessage
        resp = get_llm().invoke([
            SystemMessage(content="You are a helpful trading assistant."),
            HumanMessage(content=user_message),
        ])
        return ChatResponse(result=resp.content)

    return ChatResponse(result=_synthesize(user_message, agent_results))


@app.post("/execute-trade")
def execute_trade(req: TradeRequest):
    """Route a trade order through risk check + Shepherd proposal."""
    order = req.order
    # 1. Risk check via risk service
    risk_result = _call_service("risk", f"Evaluate risk for: {json.dumps(order)}")

    # 2. Shepherd retained proposal
    try:
        shepherd = get_shepherd()
        from src.trading_system.shepherd_layer import ProposalStatus
        proposal = shepherd.propose_trade(order=order, risk_check=risk_result)

        if proposal.status == ProposalStatus.REJECTED:
            return {
                "status": "rejected",
                "reason": risk_result,
                "proposal_id": proposal.proposal_id,
            }

        return {
            "status": "pending",
            "proposal_id": proposal.proposal_id,
            "message": (
                f"📋 Trade proposal **{proposal.proposal_id}** created\n"
                f"Risk: {risk_result}\n"
                f"Settle via: POST /shepherd/settle/trade/{proposal.proposal_id}"
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/shepherd/settle/trade/{proposal_id}")
def settle_trade(proposal_id: str, req: SettleRequest):
    try:
        from src.trading_system.shepherd_layer import ProposalStatus
        from src.trading_system import tools as T
        shepherd = get_shepherd()
        proposal = shepherd.settle_trade(
            proposal_id=proposal_id,
            accept=req.accept,
            execute_fn=T.place_order.invoke if req.accept else None,
            reason=req.reason,
        )
        return {"status": proposal.status.value, "proposal_id": proposal_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/shepherd/settle/strategy/{proposal_id}")
def settle_strategy(proposal_id: str, req: SettleRequest):
    try:
        shepherd = get_shepherd()
        proposal = shepherd.settle_strategy(
            proposal_id=proposal_id,
            accept=req.accept,
            reason=req.reason,
        )
        return {"status": proposal.status.value, "proposal_id": proposal_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/shepherd/proposals")
def list_proposals():
    try:
        from src.trading_system.shepherd_layer import ProposalStatus
        shepherd = get_shepherd()
        return shepherd.list_proposals(status=ProposalStatus.PENDING)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
