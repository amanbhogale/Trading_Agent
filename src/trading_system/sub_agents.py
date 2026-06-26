# sub_agents.py
import json
import logging
from typing import Dict, List, Optional, Tuple
from langchain_core.messages import SystemMessage

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from .llm_service import LLMService
from . import tools as T
from .memory import DB_CONFIG

logger = logging.getLogger(__name__)

import urllib.parse
_encoded_pw = urllib.parse.quote_plus(DB_CONFIG['password'])
_conninfo = f"postgresql://{DB_CONFIG['user']}:{_encoded_pw}@{DB_CONFIG['host']}/{DB_CONFIG['dbname']}"
pool = ConnectionPool(conninfo=_conninfo)
checkpointer = PostgresSaver(pool)

# ---------------------------------------------------------------------------
# Base Sub-Agent
# ---------------------------------------------------------------------------

class BaseSubAgent:
    """All sub-agents inherit from this."""

    SYSTEM_PROMPT: str = "You are a helpful AI agent."
    AGENT_ID     : str = "base_agent"

    def __init__(self, llm_service: LLMService) -> None:
        self.llm = llm_service
        self._tools = self._get_tools()
        
        # Create a LangGraph React Agent
        self.graph = create_react_agent(
            model=llm_service._client,  # Extract the underlying langchain chat model
            tools=self._tools,
            prompt=SystemMessage(content=self.SYSTEM_PROMPT),
            checkpointer=checkpointer
        )

    def _get_tools(self) -> list:
        return []

    def run(self, user_message: str) -> str:
        # We use the AGENT_ID as the thread ID to persist conversations uniquely per agent.
        # In a real multi-user scenario, this would be user_id + agent_id.
        config = {"configurable": {"thread_id": self.AGENT_ID}}
        
        # Stream the graph and return the final AI message content
        response = self.graph.invoke(
            {"messages": [("human", user_message)]}, 
            config=config
        )
        return response["messages"][-1].content


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
            T.internet_search,
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
            T.internet_search,
        ]


# ---------------------------------------------------------------------------
# Strategy & Backtest Agent
# ---------------------------------------------------------------------------

class StrategyAgent(BaseSubAgent):
    AGENT_ID = "strategy_agent"
    SYSTEM_PROMPT = """
    You are a Quantitative Strategy Designer.
    You design, backtest, and evaluate trading strategies.
    Available strategies: sma_crossover, rsi_mean_reversion, macd_trend, brownian_motion, market_making, statistical_arbitrage, momentum, mean_reversion, sentiment.
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
    """

    def __init__(self, llm_service: LLMService, approval_fn=None) -> None:
        super().__init__(llm_service)
        self._approve = approval_fn or (lambda _: False)

    def _get_tools(self):
        return [
            T.get_orders,
            T.cancel_order,
            T.get_funds,
            T.get_trade_log_tool,
            T.place_order, # Provided to the graph, but restricted via human-in-the-loop logic externally if needed
        ]

    def execute_with_approval(self, order: Dict) -> str:
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
