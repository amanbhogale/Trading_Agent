# sub_agents.py
import json
import logging
from typing import Dict, List, Optional, Tuple
from langchain_core.messages import (
    HumanMessage, SystemMessage, AIMessage, ToolMessage
)

from llm_service import LLMService, LLMConfig
from memory import MemoryManager
import tools as T

logger = logging.getLogger(__name__)
memory = MemoryManager()


# ---------------------------------------------------------------------------
# Base Sub-Agent
# ---------------------------------------------------------------------------

class BaseSubAgent:
    """All sub-agents inherit from this."""

    SYSTEM_PROMPT: str = "You are a helpful AI agent."
    AGENT_ID     : str = "base_agent"

    def __init__(self, llm_service: LLMService) -> None:
        self.llm     = llm_service
        self.mem     = memory
        self._client = llm_service.with_tools(self._get_tools())

    def _get_tools(self) -> list:
        return []

    def _build_messages(self, user_msg: str) -> List:
        mem   = self.mem.load_agent_memory(self.AGENT_ID)
        msgs  = [SystemMessage(content=self.SYSTEM_PROMPT)]

        # last 6 conversation turns for context window efficiency
        for turn in mem.conversation[-6:]:
            cls = HumanMessage if turn["role"] == "human" else AIMessage
            msgs.append(cls(content=turn["content"]))

        msgs.append(HumanMessage(content=user_msg))
        return msgs

    def _run_tool_loop(self, messages: List) -> str:
        """Agentic loop: call LLM → execute tools → repeat until done."""
        tool_map = {t.name: t for t in self._get_tools()}

        for _ in range(10):           # max 10 iterations
            response = self._client.invoke(messages)
            messages.append(response)

            if not getattr(response, "tool_calls", None):
                # no more tool calls → done
                return response.content

            for tc in response.tool_calls:
                fn_name = tc["name"]
                fn_args = tc["args"]
                logger.info("[%s] tool=%s  args=%s", self.AGENT_ID, fn_name, fn_args)

                if fn_name in tool_map:
                    result = tool_map[fn_name].invoke(fn_args)
                else:
                    result = json.dumps({"error": f"unknown tool {fn_name}"})

                messages.append(ToolMessage(
                    content      = str(result),
                    tool_call_id = tc["id"],
                ))

        return "Max iterations reached."

    def run(self, user_message: str) -> str:
        self.mem.append_conversation(self.AGENT_ID, "human", user_message)
        messages = self._build_messages(user_message)
        response = self._run_tool_loop(messages)
        self.mem.append_conversation(self.AGENT_ID, "ai", response)
        return response


# ---------------------------------------------------------------------------
# Market Data Agent
# ---------------------------------------------------------------------------

class MarketDataAgent(BaseSubAgent):
    AGENT_ID = "market_data_agent"
    SYSTEM_PROMPT = """
    You are a Market Data Specialist.
    Your job: fetch live quotes, historical OHLCV data, and portfolio info.
    Always return structured JSON summaries.
    Be concise and factual.
    """

    def _get_tools(self):
        return [
            T.get_quote,
            T.get_historical_data,
            T.get_portfolio,
            T.get_funds,
            T.get_orders,
        ]


# ---------------------------------------------------------------------------
# Technical Analysis Agent
# ---------------------------------------------------------------------------

class AnalysisAgent(BaseSubAgent):
    AGENT_ID = "analysis_agent"
    SYSTEM_PROMPT = """
    You are a Technical Analysis Expert.
    You compute RSI, MACD, Bollinger Bands, EMA crossovers.
    You interpret signals and give BUY / SELL / HOLD recommendations
    with clear reasoning based on indicator confluence.
    Always mention the key levels (support/resistance).
    """

    def _get_tools(self):
        return [
            T.calculate_indicators,
            T.get_historical_data,
            T.load_strategy_tool,
            T.list_strategies_tool,
        ]


# ---------------------------------------------------------------------------
# Strategy & Backtest Agent
# ---------------------------------------------------------------------------

class StrategyAgent(BaseSubAgent):
    AGENT_ID = "strategy_agent"
    SYSTEM_PROMPT = """
    You are a Quantitative Strategy Designer.
    You design, backtest, and evaluate trading strategies.
    Available strategies: sma_crossover, rsi_mean_reversion, macd_trend.
    Always report: total return, sharpe ratio, max drawdown, win rate,
    number of trades, and comparison vs buy-and-hold.
    Save promising strategies for future reference.
    """

    def _get_tools(self):
        return [
            T.backtest_strategy,
            T.save_strategy_tool,
            T.load_strategy_tool,
            T.list_strategies_tool,
            T.get_historical_data,
        ]


# ---------------------------------------------------------------------------
# Visualization Agent
# ---------------------------------------------------------------------------

class VisualizationAgent(BaseSubAgent):
    AGENT_ID = "viz_agent"
    SYSTEM_PROMPT = """
    You are a Data Visualization Expert.
    You create candlestick charts, equity curves, and portfolio dashboards.
    When a chart is created, return the file path and a brief description
    of what the chart shows and key visual patterns.
    """

    def _get_tools(self):
        return [
            T.create_candlestick_chart,
            T.create_backtest_chart,
            T.create_portfolio_dashboard,
            T.list_visualizations_tool,
        ]


# ---------------------------------------------------------------------------
# Risk Agent
# ---------------------------------------------------------------------------

class RiskAgent(BaseSubAgent):
    AGENT_ID = "risk_agent"
    SYSTEM_PROMPT = """
    You are a Risk Management Officer.
    Evaluate proposed trades for:
    - Position sizing (Kelly criterion / fixed fractional)
    - Portfolio concentration risk
    - Drawdown limits
    - Available margin vs required margin
    Always output: APPROVED / REJECTED with explicit reason and suggested
    position size.
    """

    def _get_tools(self):
        return [
            T.get_portfolio,
            T.get_funds,
            T.calculate_indicators,
            T.get_trade_log_tool,
        ]


# ---------------------------------------------------------------------------
# Execution Agent  (requires human approval gate)
# ---------------------------------------------------------------------------

class ExecutionAgent(BaseSubAgent):
    AGENT_ID = "execution_agent"
    SYSTEM_PROMPT = """
    You are a Trade Execution Specialist.
    You place, modify, and cancel orders on Zerodha Kite.
    NEVER place an order without explicit human approval.
    Confirm order details before execution.
    Always log the result to memory.
    """

    def __init__(self, llm_service: LLMService, approval_fn=None) -> None:
        super().__init__(llm_service)
        # approval_fn: callable(order_details) -> bool
        self._approve = approval_fn or (lambda _: False)

    def _get_tools(self):
        return [
            T.get_orders,
            T.cancel_order,
            T.get_funds,
            T.get_trade_log_tool,
        ]

    def execute_with_approval(self, order: Dict) -> str:
        """Human-in-the-loop execution gate."""
        summary = (
            f"⚠️  TRADE REQUEST\n"
            f"  Symbol : {order.get('symbol')}\n"
            f"  Action : {order.get('transaction')}\n"
            f"  Qty    : {order.get('quantity')}\n"
            f"  Type   : {order.get('order_type','MARKET')}\n"
            f"  Price  : {order.get('price', 'MARKET')}\n"
        )

        if self._approve(order):
            result = T.place_order.invoke(order)
            return f"✅ Order placed\n{result}"
        else:
            return f"❌ Order rejected by human\n{summary}"
