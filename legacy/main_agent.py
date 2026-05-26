# main_agent.py
import os
import json
import logging
from typing import Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from llm_service import LLMService, LLMConfig
from memory import MemoryManager
from sub_agents import (
    MarketDataAgent,
    AnalysisAgent,
    StrategyAgent,
    VisualizationAgent,
    RiskAgent,
    ExecutionAgent,
)
import tools as T

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)
memory = MemoryManager()

ORCHESTRATOR_SYSTEM = """
You are the Master Trading Orchestrator.

You coordinate a team of specialized sub-agents:
  1. market_data   – live quotes, portfolio, historical data
  2. analysis      – technical indicators, signals
  3. strategy      – backtesting, strategy design
  4. visualization – charts, dashboards
  5. risk          – position sizing, risk checks
  6. execution     – order placement (human approval required)

For every user request:
  a) Identify which agents are needed
  b) Break the task into sequential sub-tasks
  c) Route each sub-task to the correct agent
  d) Synthesize all results into a coherent final response
  e) Always mention if human approval is required for trades

You have access to memory – use context from previous turns.
"""


class OrchestratorAgent:
    """
    Master agent that routes requests to specialized sub-agents.
    Integrates with Gradio via run() and execute_trade().
    """

    AGENT_ID = "orchestrator"

    def __init__(
        self,
        llm_service      : LLMService,
        approval_callback = None,     # Gradio will inject this
    ) -> None:
        self.llm      = llm_service
        self.mem      = memory
        self.approve  = approval_callback  # fn(order) -> bool

        # ── spawn sub-agents ─────────────────────────────────────────────
        self.market_data   = MarketDataAgent(llm_service)
        self.analysis      = AnalysisAgent(llm_service)
        self.strategy      = StrategyAgent(llm_service)
        self.viz           = VisualizationAgent(llm_service)
        self.risk          = RiskAgent(llm_service)
        self.execution     = ExecutionAgent(llm_service, approval_callback)

        # plain LLM for orchestration reasoning (no tools)
        self._orchestrator_llm = llm_service

        logger.info("OrchestratorAgent ready ✅")

    # ── routing ──────────────────────────────────────────────────────────

    def _classify_intent(self, user_msg: str) -> str:
        """Ask the LLM to classify which agents are needed."""
        prompt = f"""
Classify this trading request into one or more of:
market_data | analysis | strategy | visualization | risk | execution | general

Request: "{user_msg}"

Reply with comma-separated agent names only. Example: market_data,analysis
"""
        resp = self._orchestrator_llm.invoke([
            SystemMessage(content="You are an intent classifier for a trading system."),
            HumanMessage(content=prompt),
        ])
        return resp.content.strip().lower()

    def _synthesize(self, user_msg: str, agent_results: Dict[str, str]) -> str:
        """Ask the LLM to synthesize sub-agent outputs into a final answer."""
        context = "\n\n".join(
            f"[{agent}]\n{result}"
            for agent, result in agent_results.items()
        )
        prompt = f"""
User asked: {user_msg}

Sub-agent results:
{context}

Provide a comprehensive, well-structured final answer.
If trade action is recommended, clearly state it requires human approval.
"""
        resp = self._orchestrator_llm.invoke([
            SystemMessage(content=ORCHESTRATOR_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return resp.content

    # ── public API ───────────────────────────────────────────────────────

    def run(self, user_message: str) -> str:
        """
        Main entry-point called by Gradio.
        Routes to sub-agents and returns synthesized response.
        """
        logger.info("Orchestrator received: %s", user_message[:100])
        self.mem.append_conversation(self.AGENT_ID, "human", user_message)

        intent = self._classify_intent(user_message)
        logger.info("Intent: %s", intent)

        agent_results: Dict[str, str] = {}

        if "market_data" in intent:
            agent_results["market_data"] = self.market_data.run(user_message)

        if "analysis" in intent:
            agent_results["analysis"] = self.analysis.run(user_message)

        if "strategy" in intent:
            agent_results["strategy"] = self.strategy.run(user_message)

        if "visualization" in intent:
            agent_results["visualization"] = self.viz.run(user_message)

        if "risk" in intent:
            agent_results["risk"] = self.risk.run(user_message)

        if "execution" in intent:
            agent_results["execution"] = (
                "⚠️ Trade execution requires explicit human approval "
                "via the Execute Trade tab."
            )

        if not agent_results:
            # general question – answer directly
            resp = self._orchestrator_llm.invoke([
                SystemMessage(content=ORCHESTRATOR_SYSTEM),
                HumanMessage(content=user_message),
            ])
            final = resp.content
        else:
            final = self._synthesize(user_message, agent_results)

        self.mem.append_conversation(self.AGENT_ID, "ai", final)
        return final

    def execute_trade(self, order: Dict) -> str:
        """Called explicitly from Gradio trade execution tab."""
        risk_check = self.risk.run(
            f"Evaluate risk for: {json.dumps(order)}"
        )
        if "REJECTED" in risk_check.upper():
            return f"❌ Risk agent rejected trade:\n{risk_check}"

        return self.execution.execute_with_approval(order)

    def get_dashboard(self) -> str:
        """Generate and return portfolio dashboard path."""
        return self.viz.run("Create a portfolio dashboard")

    def analyse_symbol(self, symbol: str) -> Dict[str, str]:
        """Full analysis pipeline for one symbol."""
        return {
            "market_data"   : self.market_data.run(f"Get quote for {symbol}"),
            "indicators"    : self.analysis.run(f"Calculate all indicators for {symbol}"),
            "chart"         : self.viz.run(f"Create candlestick chart for {symbol}"),
        }


# ---------------------------------------------------------------------------
# Factory function  (used by dashboard.py)
# ---------------------------------------------------------------------------

def build_orchestrator(
    model       : str,
    api_key     : str,
    base_url    : str  = "https://openrouter.ai/api/v1",
    temperature : float = 0.0,
    max_tokens  : int   = 4096,
    approval_fn         = None,
) -> OrchestratorAgent:
    """
    Construct a fully-wired OrchestratorAgent from raw credentials.
    Called once from the Gradio dashboard on startup.
    """
    config = LLMConfig(
        model       = model,
        api_key     = api_key,
        provider    = "openai",      # openrouter is openai-compatible
        temperature = temperature,
        max_tokens  = max_tokens,
        base_url    = base_url,
    )
    service = LLMService(config)
    return OrchestratorAgent(service, approval_callback=approval_fn)
