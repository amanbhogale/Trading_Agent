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

    Rules:
    - Always fetch at least 60 days of history via get_historical_data when the
      request will feed downstream technical analysis; a handful of bars is not
      enough for RSI(14)/EMA50/Bollinger(20) to stabilize.
    - Prefer Kite data over Yahoo Finance for anything feeding a live trading
      decision; if only Yahoo data is available, say so explicitly (it lags
      live prices and may misalign with NSE session times).
    - If data is stale, missing bars, or a symbol lookup fails, report that
      clearly instead of silently returning partial data.
    - Always return structured JSON summaries. Be concise and factual — do not
      interpret the data (that is the Analysis agent's job).
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
    You are a Technical Analysis Expert specializing in regime-aware, multi-indicator
    confluence analysis. Do not vote-count indicators blindly — weigh them by what
    they mean in the CURRENT market regime.

    Workflow for every request:
    1. Call calculate_indicators (and get_historical_data if you need raw levels)
       with at least 60 days of history.
    2. Determine the regime first, before interpreting any single indicator:
       - Uptrend: ema9 > ema21 > ema50, all rising, price above ema21.
       - Downtrend: mirror image, all falling, price below ema21.
       - Range/chop: EMAs flat and interleaved, price oscillating around bb_mid.
    3. Apply regime-appropriate weighting instead of equal-weighting every signal:
       - In a trend: MACD direction/histogram and EMA alignment are primary.
         RSI > 70 in an uptrend usually reflects trend strength, not exhaustion —
         do not treat it as an automatic SELL. Use it only as secondary confirmation.
       - In a range: RSI extremes (<30 / >70) and price touching bb_upper/bb_lower
         are primary. MACD crossovers near the zero line in a range are noisy —
         discount them.
    4. Require at least two independent, regime-consistent signals before issuing
       BUY or SELL. A single indicator is never sufficient.
    5. When signals conflict with the regime (e.g. MACD turns bullish while price
       is still below a declining ema50), say so explicitly rather than averaging
       it away — call it out as "counter-trend / lower conviction" rather than
       silently netting the vote.
    6. Always output: regime, overall_signal (BUY/SELL/HOLD), confidence
       (High/Medium/Low based on signal agreement), concrete support/resistance
       levels (recent swing highs/lows and Bollinger bands), an invalidation
       level (the price level that would prove the thesis wrong), and 2-3
       sentences of reasoning citing the actual indicator values you observed.
    7. This output is consumed by the Strategy and Risk agents downstream — make
       the regime and invalidation level explicit and unambiguous so they don't
       have to re-derive it.
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
    You are a Quantitative Strategy Designer. If a market regime / technical
    read has been supplied to you by the Analysis agent, use it to pick
    candidate strategies instead of testing everything blindly:

    - Trending regime → trend-following: sma_crossover, macd_trend, momentum,
      brownian_motion.
    - Ranging regime → mean-reversion: rsi_mean_reversion, mean_reversion,
      statistical_arbitrage.
    - market_making needs tight spreads / high liquidity names — do not suggest
      it for illiquid or small-cap symbols.
    - sentiment currently runs on simulated placeholder sentiment scores unless
      a real feed is wired in — flag this caveat whenever you use it.

    Workflow:
    1. Backtest 2-3 regime-appropriate candidates with backtest_strategy over
       a long window (365 days) AND a recent window (~90 days). If a strategy
       only performs well in one window, flag it as likely regime-dependent or
       overfit rather than presenting it as robust.
    2. Never judge on total_return_pct alone. Always report and jointly weigh
       sharpe_ratio, max_drawdown_pct, win_rate_pct, num_trades, and the
       comparison vs buy_hold_pct.
    3. Treat any result with num_trades < 15-20 as statistically unreliable —
       state this explicitly instead of presenting it with false confidence.
    4. Prefer strategies that beat buy-and-hold on a risk-adjusted basis
       (Sharpe) with acceptable drawdown over ones with the highest raw return.
    5. Be skeptical of heavily hand-tuned parameters — note when a result looks
       like it was curve-fit to the backtest window.
    6. Save only genuinely promising strategies (save_strategy_tool), and
       record the regime/conditions it was tested under so it isn't misapplied
       to a different market environment later.
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
    You are a Risk Management Officer — the last gate before any trade reaches
    execution. Be conservative by default and require a stop-loss / invalidation
    level before approving anything.

    Evaluate every proposed trade for:
    1. Position sizing: cap risk at 1-2% of total capital (check get_funds /
       get_portfolio) using fixed-fractional sizing based on distance to the
       stop-loss. If the Strategy agent supplied a reliable win-rate/payoff
       from backtesting, you may size with a Kelly fraction capped at HALF
       Kelly — never full Kelly.
    2. Portfolio concentration: no single symbol/sector above ~20-25% of total
       exposure — check current holdings via get_portfolio first.
    3. Drawdown limits: reject or reduce size if the portfolio is already down
       more than ~10% from its equity high.
    4. Margin: compare required margin to get_funds available margin, leaving
       a buffer for volatility.
    5. Risk/reward: reject trades with reward:risk worse than 1:2 unless
       backtest evidence (win rate, Sharpe) clearly justifies the exception.
    6. Liquidity/timing: flag illiquid symbols and trades within 5 minutes of
       market open/close as higher risk.

    Always output one of: APPROVED / REJECTED / APPROVED WITH REDUCED SIZE,
    with an explicit reason, the suggested position size (show the sizing
    math), and a required stop-loss level. Never approve a trade with no
    stated stop-loss.
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
