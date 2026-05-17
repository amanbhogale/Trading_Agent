from types import TracebackType
import os
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime , timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Literal , Callable , Any, Optional
from langchain.agents.structured_output import ResponseFormat
from langchain_core.language_models import FakeListChatModel
from langchain_core.tools import tool
from langchain_core.messages import content
from langgraph.pregel.debug import map_debug_checkpoint
from langgraph.store.base import Result
from pydantic import KafkaDsn
from requests import get
from tavily import TavilyClient
from deepagents import create_deep_agent
from memory import MemoryManager
from kiteconnect import KiteConnect , KiteTicker

logger = logging.getLogger(__name__)
memory = MemoryManager()


_kite = Optional[KiteConnect]

def get_kite() -> KiteConnect:
    if _kite is None:
        raise RuntimeError("Kite not initialised – call init_kite() first")
    return _kite

def init_kite(api_key: str, access_token: str) -> KiteConnect:
    global _kite
    _kite = KiteConnect(api_key=api_key)
    _kite.set_access_token(access_token)
    logger.info("Kite initialised ✅")
    return _kite

#======================#
#internet search tool using travily
#=======================#
travily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool

def internet_search(
    query               : str,
    max_results         : int                              = 5,
    topic               : Literal["general","news","finance"] = "general",
    include_raw_content : bool                             = False,
) -> dict:
    """
    Run a web search using Tavily and return structured results.
    query               — search query string.
    max_results         — maximum number of results to return (default 5).
    topic               — search category: general | news | finance.
    include_raw_content — whether to include full raw page content.
    Returns a dict with query, answer summary, and list of results.
    """
    response = tavily_client.search(
        query               = query,
        max_results         = max_results,
        topic               = topic,
        include_raw_content = include_raw_content,
    )
    return {
        "query"  : query,
        "answer" : response.get("answer"),
        "results": [
            {
                "title"  : r.get("title"),
                "url"    : r.get("url"),
                "content": r.get("content"),   # ← was response.get() (bug)
                "score"  : r.get("score"),
            }
            for r in response.get("results", [])
        ],
    }
# ═══════════════════════════════════════════════════════════════════════════
# KITE / MARKET DATA TOOLS
# ═══════════════════════════════════════════════════════════════════════════

@tool
def get_portfolio() -> str:
    """Fetch current portfolio holdings from Zerodha Kite."""
    try:
        kite     = get_kite()
        holdings = kite.holdings()
        positions = kite.positions()

        result = {
            "holdings"  : holdings,
            "positions" : positions["net"],
            "day_pos"   : positions["day"],
            "fetched_at": datetime.now().isoformat(),
        }
        memory.update_context("orchestrator", "last_portfolio", result)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_quote(symbols: str) -> str:
    """
    Fetch live quotes for comma-separated symbols.
    Example: "NSE:INFY,NSE:TCS"
    """
    try:
        kite   = get_kite()
        syms   = [s.strip() for s in symbols.split(",")]
        quotes = kite.quote(syms)
        return json.dumps(quotes, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_historical_data(
    symbol: str,
    interval: str = "day",
    days: int = 30,
) -> str:
    """
    Fetch OHLCV historical data for a symbol.

    Parameters
    ----------
    symbol   : e.g. "NSE:INFY"
    interval : minute | 3minute | 5minute | 15minute | 30minute | 60minute | day | week | month
    days     : number of days to look back
    """
    try:
        kite       = get_kite()
        instrument = kite.ltp(symbol)[symbol]["instrument_token"]
        to_date    = datetime.now()
        from_date  = to_date - timedelta(days=days)

        data = kite.historical_data(instrument, from_date, to_date, interval)
        df   = pd.DataFrame(data)
        memory.update_context("data_agent", f"hist_{symbol}", df.to_dict())
        return df.to_json(orient="records", date_format="iso")
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def place_order(
    symbol      : str,
    exchange    : str,
    transaction : str,
    quantity    : int,
    order_type  : str = "MARKET",
    price       : float = 0.0,
    product     : str = "CNC",
) -> str:
    """
    Place a buy/sell order on Kite. Requires human approval.

    Parameters
    ----------
    symbol      : e.g. "INFY"
    exchange    : NSE | BSE
    transaction : BUY | SELL
    quantity    : number of shares
    order_type  : MARKET | LIMIT
    price       : required for LIMIT orders
    product     : CNC | MIS | NRML
    """
    try:
        kite = get_kite()
        params = dict(
            tradingsymbol    = symbol,
            exchange         = exchange,
            transaction_type = transaction.upper(),
            quantity         = quantity,
            order_type       = order_type,
            product          = product,
        )
        if order_type == "LIMIT":
            params["price"] = price

        order_id = kite.place_order(variety=kite.VARIETY_REGULAR, **params)

        trade = {**params, "order_id": order_id, "status": "placed"}
        memory.log_trade(trade)
        return json.dumps({"order_id": order_id, "status": "SUCCESS"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_orders() -> str:
    """Fetch today's order book from Kite."""
    try:
        return json.dumps(get_kite().orders(), default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def cancel_order(order_id: str) -> str:
    """Cancel a pending order by order_id."""
    try:
        get_kite().cancel_order(
            variety  = KiteConnect.VARIETY_REGULAR,
            order_id = order_id,
        )
        return json.dumps({"status": "CANCELLED", "order_id": order_id})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_funds() -> str:
    """Fetch available margin/funds from Kite."""
    try:
        margins = get_kite().margins()
        return json.dumps(margins, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS TOOLS
# ═══════════════════════════════════════════════════════════════════════════

@tool
def calculate_indicators(symbol: str, days: int = 60) -> str:
    """
    Calculate RSI, MACD, Bollinger Bands, EMA for a symbol.
    Returns JSON with all indicator values.
    """
    try:
        raw = json.loads(get_historical_data(symbol, "day", days))
        df  = pd.DataFrame(raw)
        df["date"]  = pd.to_datetime(df["date"])
        df          = df.sort_values("date")
        close       = df["close"]

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema12        = close.ewm(span=12).mean()
        ema26        = close.ewm(span=26).mean()
        df["macd"]   = ema12 - ema26
        df["signal"] = df["macd"].ewm(span=9).mean()
        df["hist"]   = df["macd"] - df["signal"]

        # Bollinger Bands
        sma20          = close.rolling(20).mean()
        std20          = close.rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_mid"]   = sma20
        df["bb_lower"] = sma20 - 2 * std20

        # EMAs
        df["ema9"]  = close.ewm(span=9).mean()
        df["ema21"] = close.ewm(span=21).mean()
        df["ema50"] = close.ewm(span=50).mean()

        last = df.iloc[-1].to_dict()
        memory.update_context("analysis_agent", f"indicators_{symbol}", last)
        return json.dumps(last, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def backtest_strategy(
    symbol     : str,
    strategy   : str,
    params     : str = "{}",
    days       : int = 365,
) -> str:
    """
    Backtest a strategy on historical data.

    Parameters
    ----------
    symbol   : e.g. "NSE:INFY"
    strategy : "sma_crossover" | "rsi_mean_reversion" | "macd_trend"
    params   : JSON string with strategy params
    days     : lookback days

    Returns backtest statistics as JSON.
    """
    try:
        raw    = json.loads(get_historical_data.invoke(symbol, "day", days))
        df     = pd.DataFrame(raw)
        df["date"]  = pd.to_datetime(df["date"])
        df          = df.sort_values("date").reset_index(drop=True)
        p      = json.loads(params)

        # ── strategy logic ──────────────────────────────────────────────
        if strategy == "sma_crossover":
            fast = p.get("fast", 20)
            slow = p.get("slow", 50)
            df["sma_fast"] = df["close"].rolling(fast).mean()
            df["sma_slow"] = df["close"].rolling(slow).mean()
            df["signal"]   = (df["sma_fast"] > df["sma_slow"]).astype(int)

        elif strategy == "rsi_mean_reversion":
            ob = p.get("overbought", 70)
            os = p.get("oversold",   30)
            delta = df["close"].diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rsi   = 100 - (100 / (1 + gain / loss))
            df["signal"] = 0
            df.loc[rsi < os, "signal"] = 1
            df.loc[rsi > ob, "signal"] = 0

        elif strategy == "macd_trend":
            ema12 = df["close"].ewm(span=12).mean()
            ema26 = df["close"].ewm(span=26).mean()
            macd  = ema12 - ema26
            sig   = macd.ewm(span=9).mean()
            df["signal"] = ((macd > sig) & (macd > 0)).astype(int)

        else:
            return json.dumps({"error": f"Unknown strategy: {strategy}"})

        # ── returns ─────────────────────────────────────────────────────
        df["position"] = df["signal"].shift(1)
        df["ret"]      = df["close"].pct_change()
        df["strat_ret"]= df["ret"] * df["position"]
        df             = df.dropna()

        cum_ret    = (1 + df["strat_ret"]).cumprod().iloc[-1] - 1
        bh_ret     = (df["close"].iloc[-1] / df["close"].iloc[0]) - 1
        sharpe     = (df["strat_ret"].mean() / df["strat_ret"].std()) * (252 ** 0.5)
        max_dd     = (df["strat_ret"].cumsum() - df["strat_ret"].cumsum().cummax()).min()
        win_rate   = (df["strat_ret"] > 0).mean()
        n_trades   = df["position"].diff().abs().sum()

        result = {
            "strategy"         : strategy,
            "symbol"           : symbol,
            "period_days"      : days,
            "total_return_pct" : round(cum_ret * 100, 2),
            "buy_hold_pct"     : round(bh_ret * 100, 2),
            "sharpe_ratio"     : round(sharpe, 3),
            "max_drawdown_pct" : round(max_dd * 100, 2),
            "win_rate_pct"     : round(win_rate * 100, 2),
            "num_trades"       : int(n_trades),
            "daily_returns"    : df["strat_ret"].tolist(),
        }

        memory.save_strategy(f"{symbol}_{strategy}_{days}d", result)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION TOOLS
# ═══════════════════════════════════════════════════════════════════════════

@tool
def create_candlestick_chart(symbol: str, days: int = 60) -> str:
    """
    Generate an interactive Plotly candlestick chart with volume & indicators.
    Returns path to saved HTML file.
    """
    try:
        raw = json.loads(get_historical_data(symbol, "day", days))
        df  = pd.DataFrame(raw)
        df["date"] = pd.to_datetime(df["date"])

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=[symbol, "Volume", "RSI"],
        )

        # candlestick
        fig.add_trace(go.Candlestick(
            x=df["date"], open=df["open"],
            high=df["high"], low=df["low"], close=df["close"],
            name="Price",
        ), row=1, col=1)

        # EMAs
        for span, color in [(9, "orange"), (21, "blue"), (50, "red")]:
            ema = df["close"].ewm(span=span).mean()
            fig.add_trace(go.Scatter(
                x=df["date"], y=ema, name=f"EMA{span}",
                line=dict(color=color, width=1),
            ), row=1, col=1)

        # volume
        colors = ["red" if c < o else "green"
                  for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(
            x=df["date"], y=df["volume"],
            marker_color=colors, name="Volume",
        ), row=2, col=1)

        # RSI
        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = 100 - (100 / (1 + gain / loss))
        fig.add_trace(go.Scatter(
            x=df["date"], y=rsi, name="RSI",
            line=dict(color="purple"),
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red",   row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(
            title  = f"{symbol} – {days}d Chart",
            height = 800,
            template = "plotly_dark",
            xaxis_rangeslider_visible = False,
        )

        name = f"{symbol.replace(':', '_')}_{days}d"
        path = Path("data/visualizations") / f"{name}.html"
        fig.write_html(str(path))
        memory.save_visualization_meta(name, {"symbol": symbol, "days": days})
        return json.dumps({"chart_path": str(path), "status": "saved"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def create_backtest_chart(strategy_name: str) -> str:
    """
    Plot equity curve for a previously saved back-test result.
    Returns path to saved HTML chart.
    """
    try:
        strat = memory.load_strategy(strategy_name)
        if not strat:
            return json.dumps({"error": f"Strategy {strategy_name!r} not found"})

        rets       = pd.Series(strat["daily_returns"])
        equity     = (1 + rets).cumprod()
        drawdown   = equity / equity.cummax() - 1

        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=["Equity Curve", "Drawdown"])
        fig.add_trace(go.Scatter(y=equity, name="Strategy", fill="tozeroy"), row=1, col=1)
        fig.add_trace(go.Scatter(y=drawdown, name="Drawdown",
                                 fill="tozeroy", line=dict(color="red")), row=2, col=1)
        fig.update_layout(
            title    = f"Backtest: {strategy_name}",
            height   = 600,
            template = "plotly_dark",
        )

        path = Path("data/visualizations") / f"{strategy_name}_equity.html"
        fig.write_html(str(path))
        return json.dumps({"chart_path": str(path), "stats": {
            k: v for k, v in strat.items() if k != "daily_returns"
        }})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def create_portfolio_dashboard() -> str:
    """Generate a portfolio overview dashboard and return HTML path."""
    try:
        portfolio = json.loads(get_portfolio())
        holdings  = pd.DataFrame(portfolio.get("holdings", []))

        if holdings.empty:
            return json.dumps({"error": "No holdings found"})

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Portfolio Allocation", "P&L per Stock",
                "Day P&L", "Investment vs Current",
            ],
        )

        # allocation pie
        fig.add_trace(go.Pie(
            labels=holdings["tradingsymbol"],
            values=holdings["last_price"] * holdings["quantity"],
            name="Allocation",
        ), row=1, col=1)

        # P&L bar
        pnl_colors = ["green" if p > 0 else "red"
                      for p in holdings["pnl"]]
        fig.add_trace(go.Bar(
            x=holdings["tradingsymbol"],
            y=holdings["pnl"],
            marker_color=pnl_colors,
            name="P&L",
        ), row=1, col=2)

        # day P&L
        fig.add_trace(go.Bar(
            x=holdings["tradingsymbol"],
            y=holdings.get("day_change_percentage", [0] * len(holdings)),
            name="Day Change %",
        ), row=2, col=1)

        # investment vs current
        fig.add_trace(go.Bar(
            name="Invested",
            x=holdings["tradingsymbol"],
            y=holdings["average_price"] * holdings["quantity"],
        ), row=2, col=2)
        fig.add_trace(go.Bar(
            name="Current",
            x=holdings["tradingsymbol"],
            y=holdings["last_price"] * holdings["quantity"],
        ), row=2, col=2)

        fig.update_layout(
            title    = "Portfolio Dashboard",
            height   = 800,
            template = "plotly_dark",
        )

        path = Path("data/visualizations/portfolio_dashboard.html")
        fig.write_html(str(path))
        return json.dumps({"dashboard_path": str(path)})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════
# FILESYSTEM / MEMORY TOOLS
# ═══════════════════════════════════════════════════════════════════════════

@tool
def save_strategy_tool(name: str, strategy_json: str) -> str:
    """Persist a strategy definition to disk."""
    try:
        strat = json.loads(strategy_json)
        path  = memory.save_strategy(name, strat)
        return json.dumps({"saved": path})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def load_strategy_tool(name: str) -> str:
    """Load a saved strategy by name."""
    try:
        strat = memory.load_strategy(name)
        return json.dumps(strat or {"error": "not found"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def list_strategies_tool() -> str:
    """List all saved strategies."""
    return json.dumps({"strategies": memory.list_strategies()})


@tool
def get_trade_log_tool(date: str = "") -> str:
    """Get trade log for a given date (YYYY-MM-DD). Defaults to today."""
    return json.dumps(memory.get_trade_log(date or None))


@tool
def list_visualizations_tool() -> str:
    """List all saved visualization files."""
    return json.dumps({"visualizations": memory.list_visualizations()})



