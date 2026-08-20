# main_agent.py
import os
import json
import logging
from typing import Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from .llm_service import LLMService, LLMConfig
from .memory import MemoryManager
from .sub_agents import (
    MarketDataAgent,
    AnalysisAgent,
    StrategyAgent,
    VisualizationAgent,
    RiskAgent,
    ExecutionAgent,
)
from . import tools as T
from .shepherd_layer import ShepherdSafetyLayer, ProposalStatus

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

Safety & Governance (Shepherd Layer):
  All strategy code generation and trade execution flows through the
  Shepherd safety layer, which provides:
  - Sandboxed execution for generated code
  - Retained changesets — proposals are held for human review
  - Durable execution traces for audit compliance
  - Explicit settlement: select() to accept, discard() to reject

For every user request:
  a) Identify which agents are needed
  b) Break the task into sequential sub-tasks
  c) Route each sub-task to the correct agent
  d) Synthesize all results into a coherent final response
  e) Strategy proposals and trade orders are RETAINED for review
  f) Always mention pending proposals that need human settlement

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
        workspace_root   : str = ".",
    ) -> None:
        self.llm      = llm_service
        self.mem      = memory
        self.approve  = approval_callback  # fn(order) -> bool

        # ── Shepherd Safety Layer ─────────────────────────────────────────
        self.shepherd = ShepherdSafetyLayer(workspace_root=workspace_root)
        logger.info("Shepherd safety layer attached to orchestrator ✅")

        # ── spawn sub-agents ─────────────────────────────────────────────
        self.market_data   = MarketDataAgent(llm_service)
        self.analysis      = AnalysisAgent(llm_service)
        self.strategy      = StrategyAgent(llm_service)
        self.viz           = VisualizationAgent(llm_service)
        self.risk          = RiskAgent(llm_service)
        self.execution     = ExecutionAgent(llm_service, approval_callback)

        # plain LLM for orchestration reasoning (no tools)
        self._orchestrator_llm = llm_service

        # ── Build LangGraph StateGraph for Orchestrator ──────────────────
        from typing import Annotated, Sequence, TypedDict, Any
        from langgraph.graph import StateGraph, START, END
        from langgraph.graph.message import add_messages
        from .sub_agents import checkpointer

        class OrchestratorState(TypedDict):
            messages: Annotated[Sequence[Any], add_messages]

        def orchestrate_node(state: OrchestratorState) -> Dict:
            # We only process the last human message
            last_msg = state["messages"][-1]
            user_message = last_msg.content

            logger.info("Orchestrator received: %s", user_message[:100])
            intent = self._classify_intent(user_message)
            logger.info("Intent: %s", intent)

            agent_results: Dict[str, str] = {}
            # Accumulated findings get appended to each subsequent agent's
            # prompt so the pipeline is a real pipeline (plan -> data ->
            # analyse -> risk -> execute), not five agents working blind to
            # each other on the same raw user message.
            context_so_far = ""

            def _with_context(base_msg: str) -> str:
                if not context_so_far:
                    return base_msg
                return (
                    f"{base_msg}\n\n"
                    f"--- Findings from earlier agents in this pipeline ---\n"
                    f"{context_so_far}"
                )

            if "market_data" in intent:
                result = self.market_data.run(user_message)
                agent_results["market_data"] = result
                context_so_far += f"\n[market_data]\n{result}\n"

            if "analysis" in intent:
                result = self.analysis.run(_with_context(user_message))
                agent_results["analysis"] = result
                context_so_far += f"\n[analysis]\n{result}\n"

            if "strategy" in intent:
                result = self.strategy.run(_with_context(user_message))
                agent_results["strategy"] = result
                context_so_far += f"\n[strategy]\n{result}\n"

                # ── Shepherd: retain strategy as a reviewable proposal ────
                try:
                    # Extract asset/timeframe from user message heuristically
                    asset = self._extract_asset(user_message)
                    timeframe = self._extract_timeframe(user_message)
                    proposal = self.shepherd.propose_strategy(
                        prompt       = user_message,
                        asset        = asset,
                        timeframe    = timeframe,
                        agent_result = result,
                    )
                    agent_results["shepherd_strategy"] = (
                        f"📋 Strategy proposal **{proposal.proposal_id}** created "
                        f"(status: {proposal.status.value}).\n"
                        f"Changed files: {proposal.changed_paths}\n"
                        f"Review the diff and settle with: "
                        f"`/api/shepherd/settle/strategy/{proposal.proposal_id}`"
                    )
                except Exception as e:
                    logger.warning("Shepherd strategy proposal failed: %s", e)

            if "visualization" in intent:
                agent_results["visualization"] = self.viz.run(user_message)

            if "risk" in intent:
                result = self.risk.run(_with_context(user_message))
                agent_results["risk"] = result
                context_so_far += f"\n[risk]\n{result}\n"

            if "execution" in intent:
                agent_results["execution"] = (
                    "⚠️ Trade execution requires explicit human approval "
                    "via the Execute Trade tab.\n"
                    "🛡️ All trades flow through the Shepherd safety layer — "
                    "orders are retained as proposals until explicitly settled."
                )

            if not agent_results:
                resp = self._orchestrator_llm.invoke([
                    SystemMessage(content=ORCHESTRATOR_SYSTEM),
                    HumanMessage(content=user_message),
                ])
                final = resp.content
            else:
                final = self._synthesize(user_message, agent_results)

            return {"messages": [AIMessage(content=final)]}

        builder = StateGraph(OrchestratorState)
        builder.add_node("orchestrate", orchestrate_node)
        builder.add_edge(START, "orchestrate")
        builder.add_edge("orchestrate", END)
        
        self.graph = builder.compile(checkpointer=checkpointer)
        logger.info("OrchestratorAgent LangGraph ready ✅")

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
        Routes to sub-agents and returns synthesized response via LangGraph.
        """
        config = {"configurable": {"thread_id": self.AGENT_ID}}
        response = self.graph.invoke(
            {"messages": [HumanMessage(content=user_message)]}, 
            config=config
        )
        return response["messages"][-1].content

    def execute_trade(self, order: Dict) -> str:
        """
        Called explicitly from Gradio trade execution tab.

        Now routes through Shepherd's retained proposal model:
        1. RiskAgent evaluates the order
        2. Shepherd creates a retained trade proposal
        3. Proposal is returned for human review (not auto-executed)
        4. Human settles via settle_trade_proposal()
        """
        risk_check = self.risk.run(
            f"Evaluate risk for: {json.dumps(order)}"
        )

        # Create a retained trade proposal via Shepherd
        proposal = self.shepherd.propose_trade(
            order      = order,
            risk_check = risk_check,
        )

        if proposal.status == ProposalStatus.REJECTED:
            return (
                f"❌ Trade auto-rejected by risk check:\n{risk_check}\n"
                f"Proposal ID: {proposal.proposal_id}"
            )

        return (
            f"📋 Trade proposal **{proposal.proposal_id}** created\n"
            f"⚠️  TRADE REQUEST\n"
            f"  Symbol : {order.get('symbol')}\n"
            f"  Action : {order.get('transaction')}\n"
            f"  Qty    : {order.get('quantity')}\n"
            f"  Type   : {order.get('order_type', 'MARKET')}\n"
            f"  Price  : {order.get('price', 'MARKET')}\n\n"
            f"Risk Assessment:\n{risk_check}\n\n"
            f"🛡️ Settle via: `/api/shepherd/settle/trade/{proposal.proposal_id}`"
        )

    def settle_trade_proposal(
        self, proposal_id: str, accept: bool, reason: str = ""
    ) -> str:
        """
        Settle a retained trade proposal (Shepherd settlement).

        Parameters
        ----------
        proposal_id : str
            The trade proposal ID to settle
        accept : bool
            True to execute, False to reject
        reason : str
            Optional reason for the decision
        """
        proposal = self.shepherd.settle_trade(
            proposal_id = proposal_id,
            accept      = accept,
            execute_fn  = T.place_order.invoke if accept else None,
            reason      = reason,
        )
        if proposal.status == ProposalStatus.ACCEPTED:
            return f"✅ Trade {proposal_id} EXECUTED\n{proposal.result}"
        elif proposal.status == ProposalStatus.REJECTED:
            return f"❌ Trade {proposal_id} REJECTED (reason: {reason})"
        else:
            return f"⚠️ Trade {proposal_id} status: {proposal.status.value}\n{proposal.error}"

    def settle_strategy_proposal(
        self, proposal_id: str, accept: bool, reason: str = ""
    ) -> str:
        """
        Settle a retained strategy proposal (Shepherd settlement).

        Parameters
        ----------
        proposal_id : str
            The strategy proposal ID to settle
        accept : bool
            True to write strategy file, False to discard
        reason : str
            Optional reason for the decision
        """
        proposal = self.shepherd.settle_strategy(
            proposal_id = proposal_id,
            accept      = accept,
            reason      = reason,
        )
        if proposal.status == ProposalStatus.ACCEPTED:
            return f"✅ Strategy {proposal_id} ACCEPTED → {proposal.file_path}"
        elif proposal.status == ProposalStatus.REJECTED:
            return f"❌ Strategy {proposal_id} DISCARDED (reason: {reason})"
        else:
            return f"⚠️ Strategy {proposal_id} status: {proposal.status.value}"

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

    # ── Shepherd helper methods ───────────────────────────────────────────

    def get_pending_proposals(self) -> Dict:
        """Get all pending proposals from the Shepherd safety layer."""
        return self.shepherd.list_proposals(status=ProposalStatus.PENDING)

    def get_proposal_diff(self, proposal_id: str) -> Optional[str]:
        """Get the diff/changeset for a strategy proposal."""
        proposal = self.shepherd.get_strategy_proposal(proposal_id)
        if proposal:
            return proposal.diff
        return None

    def get_execution_traces(self, limit: int = 20) -> list:
        """Get recent execution traces for audit."""
        return self.shepherd.list_traces(limit=limit)

    def _extract_asset(self, msg: str) -> str:
        """Heuristically extract asset/symbol from user message."""
        # Common patterns: "RELIANCE", "NIFTY", "BTC", "AAPL"
        import re
        # Look for uppercase words that look like ticker symbols
        matches = re.findall(r'\b([A-Z]{2,10})\b', msg)
        # Filter out common English words
        stopwords = {'THE', 'AND', 'FOR', 'WITH', 'THIS', 'THAT', 'FROM',
                     'BUY', 'SELL', 'HOLD', 'RSI', 'MACD', 'EMA', 'SMA'}
        tickers = [m for m in matches if m not in stopwords]
        return tickers[0] if tickers else "UNKNOWN"

    def _extract_timeframe(self, msg: str) -> str:
        """Heuristically extract timeframe from user message."""
        import re
        tf_match = re.search(r'\b(\d+[mhd]|\d+\s*(?:min|hour|day|week))\b', msg.lower())
        return tf_match.group(0) if tf_match else "1d"


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
