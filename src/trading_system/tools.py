import os
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Literal, Callable, Any, Mapping, Optional

from langchain_core.tools import tool
from requests import get
from tavily import TavilyClient
from kiteconnect import KiteConnect, KiteTicker

from .memory import MemoryManager

logger = logging.getLogger(__name__)
memory = MemoryManager()


_kite = Optional[KiteConnect]

def get_kite() -> KiteConnect:
    if _kite is None:
        raise RuntimeError("Kite not initialised – call init_kite() first")
    return _kite

def init_kite(api_key: str, access_token: str) -> KiteConnect:
    global _kite
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        # Validate the credentials actively
        kite.profile()
        _kite = kite
        logger.info("Kite initialised and verified ✅")
        return _kite
    except Exception as e:
        _kite = None
        logger.error("Kite verification failed: %s", e)
        raise RuntimeError(f"Kite verification failed. Please check your API key/Access token. Details: {e}")
def _kite_to_yf(symbol : str)-> str:
    """convert kite symbols to yahoo Finance ticker. 

    NSE:INFY -> INFY.NS
    BSE:INFY -> INFY.BO
    INFY -> INFY.NS
    APPL -> AAPL
    )
    """
    if ":" not in symbol:
        #heuristic if it look like an indian ticker with no dots add .NS
        if symbol.isalpha() and len(symbol) <= 10 and symbol.upper() == symbol:
            return symbol + ".NS"
        return symbol
    exchange , ticker = symbol.split(":" , 1)
    suffix_map = {"NSE": ".NS" , "BSE" : ".BO" , "MCX": ".BO"}
    return ticker + suffix_map.get(exchange.upper(), "")

 
def kite_available() -> bool:
    return _kite is not None


def _yf_interval(kite_interval:str)-> str:
    """map kite interval string to yfinance interval strings."""
    mapping = {
        "minute"   : "1m",
        "3minute"  : "2m",
        "5minute"  : "5m",
        "15minute" : "15m",
        "30minute" : "30m",
        "60minute" : "60m",
        "day"      : "1d",
        "week"     : "1wk",
        "month"    : "1mo",
    }
    return mapping.get(kite_interval, "1d")
 
 
def _days_to_period(days: int, interval: str) -> str:
    """Convert lookback days to a yfinance period string."""
    days = int(days)
    if interval in ("1m",):
        # yfinance caps 1m at 7d
        days = min(days, 7)
    if days <= 7:
        return "7d"
    if days <= 30:
        return "1mo"
    if days <= 90:
        return "3mo"
    if days <= 180:
        return "6mo"
    if days <= 365:
        return "1y"
    if days <= 730:
        return "2y"
    return "5y"
# ═══════════════════════════════════════════════════════════════════════════
# YAHOO FINANCE DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════
def _yf_historical(symbol: str, interval: str = "day", days: int = 30) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance and return a clean DataFrame.
 
    Columns: date, open, high, low, close, volume
    """
    days = int(days)
    yf_sym    = _kite_to_yf(symbol)
    yf_int    = _yf_interval(interval)
    period    = _days_to_period(days, yf_int)
 
    ticker = yf.Ticker(yf_sym)
    df     = ticker.history(period=period, interval=yf_int, auto_adjust=True)
 
    if df.empty:
        raise ValueError(
            f"Yahoo Finance returned no data for '{yf_sym}'. "
            "Check the symbol (e.g. NSE:INFY, BSE:TCS) and try again."
        )
 
    df = df.reset_index()
    # Normalise column names
    df.columns = [c.lower() for c in df.columns]
 
    # yfinance uses 'datetime' or 'date' depending on interval
    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})
 
    # Keep only the columns we need
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df   = df[keep].copy()
 
    # Strip tz info so JSON serialisation doesn't fail
    if hasattr(df["date"].dtype, "tz") and df["date"].dtype.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
 
    # Drop rows where 'close' is missing (e.g. current incomplete day glitch)
    df = df.dropna(subset=["close"])

    # Trim to requested number of days
    if interval == "day":
        df = df.tail(days)
 
    df = df.sort_values("date").reset_index(drop=True)
    return df
 
 
def _yf_quote(symbol: str) -> dict:
    """Fetch live / last-traded quote from Yahoo Finance."""
    yf_sym = _kite_to_yf(symbol)
    ticker = yf.Ticker(yf_sym)
    info   = ticker.fast_info
 
    return {
        "symbol"          : symbol,
        "yf_symbol"       : yf_sym,
        "last_price"      : info.last_price,
        "previous_close"  : info.previous_close,
        "open"            : info.open,
        "day_high"        : info.day_high,
        "day_low"         : info.day_low,
        "volume"          : info.last_volume,
        "market_cap"      : getattr(info, "market_cap", None),
        "52w_high"        : info.fifty_two_week_high,
        "52w_low"         : info.fifty_two_week_low,
        "fetched_at"      : datetime.now().isoformat(),
        "source"          : "yahoo_finance",
    }
 
 
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
    response = travily_client.search(
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
        if not kite_available():
            return json.dumps({
                "error" : "portfolio requires kite authentication on."
                "please connect via config tab"
                })

        kite     = get_kite()
        holdings = kite.holdings()
        positions = kite.positions()

        result = {
            "holdings"  : holdings,
            "positions" : positions["net"],
            "day_pos"   : positions["day"],
            "fetched_at": datetime.now().isoformat(),
            "source" : "kite",
        }
        memory.update_context("orchestrator", "last_portfolio", result)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_quote(symbols: str) -> str:
    """
    Fetch live quotes for one or more symbols (comma-separated).
 
    Uses Kite if authenticated, Yahoo Finance otherwise.
 
    Parameters
    ----------
    symbols : comma-separated Kite-style symbols, e.g. "NSE:INFY,NSE:TCS"
    """
    try:
        sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
 
        # ── Kite path ────────────────────────────────────────────────────
        if kite_available():
            try:
                quotes = get_kite().quote(sym_list)
                result = {
                    "source" : "kite",
                    "quotes" : quotes,
                    "fetched_at": datetime.now().isoformat(),
                }
                return json.dumps(result, default=str)
            except Exception as e:
                logger.warning("Kite quote failed, falling back to Yahoo Finance: %s", e)
 
        # ── Yahoo Finance path ───────────────────────────────────────────
        quotes = {}
        errors = {}
        for sym in sym_list:
            try:
                quotes[sym] = _yf_quote(sym)
            except Exception as e:
                errors[sym] = str(e)
 
        return json.dumps({
            "source"    : "yahoo_finance",
            "quotes"    : quotes,
            "errors"    : errors,
            "fetched_at": datetime.now().isoformat(),
        }, default=str)
 
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def get_historical_data(symbol: str, interval: str = "day", days: int = 30):
    """
    Fetch OHLCV historical data for a symbol.
 
    Uses Kite if authenticated, Yahoo Finance otherwise.
    Returns a JSON array of OHLCV records with columns:
    date, open, high, low, close, volume.
 
    Parameters
    ----------
    symbol   : Kite-style symbol, e.g. "NSE:INFY"
    interval : minute | 3minute | 5minute | 15minute | 30minute |
               60minute | day | week | month
    days     : number of calendar days to look back (default 30)
    """
    days = int(days)
    try:
        if kite_available():
            try:
                kite = get_kite()
                instrument = kite.ltp(symbol)[symbol]["instrument_token"]
                to_date = datetime.now()
                from_date = to_date - timedelta(days=days)
                data = kite.historical_data(instrument, from_date, to_date, interval)
                df = pd.DataFrame(data)
                memory.update_context("data_agent", f"hist_{symbol}", df.to_dict())
                return df.to_json(orient="records", date_format="iso")
            except Exception as e:
                logger.warning("Kite failed, falling back to Yahoo: %s", e)

        df = _yf_historical(symbol, interval, days)
        memory.update_context("data_agent", f"hist_{symbol}", df.to_dict())
        return df.to_json(orient="records", date_format="iso")
    except Exception as e:
        logger.error("get_historical_data failed: %s", e)
        return json.dumps({"error": str(e)})
@tool
def get_funds() -> str:
    """
    Fetch available margin/funds from Kite.
    Requires Kite authentication.
    """
    try:
        if not kite_available():
            return json.dumps({
                "error": "Funds check requires Kite authentication."
            })
        margins = get_kite().margins()
        return json.dumps(margins, default=str)
 
    except Exception as e:
        return json.dumps({"error": str(e)})
 
 
@tool
def get_orders() -> str:
    """Fetch today's order book from Kite. Requires authentication."""
    try:
        if not kite_available():
            return json.dumps({"error": "Order book requires Kite authentication."})
        return json.dumps(get_kite().orders(), default=str)
 
    except Exception as e:
        return json.dumps({"error": str(e)})
 
 
@tool
def place_order(
    symbol     : str,
    exchange   : str,
    transaction: str,
    quantity   : int,
    order_type : str  = "MARKET",
    price      : float = 0.0,
    product    : str  = "CNC",
) -> str:
    """
    Place a buy/sell order on Kite. REQUIRES human approval + Kite auth.
 
    Parameters
    ----------
    symbol      : trading symbol, e.g. "INFY"
    exchange    : NSE | BSE
    transaction : BUY | SELL
    quantity    : number of shares
    order_type  : MARKET | LIMIT
    price       : required only for LIMIT orders
    product     : CNC | MIS | NRML
    """
    try:
        if not kite_available():
            return json.dumps({
                "error": "Order placement requires Kite authentication."
            })
 
        from kiteconnect import KiteConnect
        kite   = get_kite()
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
 
        order_id = kite.place_order(variety=KiteConnect.VARIETY_REGULAR, **params)
 
        trade = {**params, "order_id": order_id, "status": "placed"}
        memory.log_trade(trade)
        return json.dumps({"order_id": order_id, "status": "SUCCESS"})
 
    except Exception as e:
        return json.dumps({"error": str(e)})
 
 
@tool
def cancel_order(order_id: str) -> str:
    """Cancel a pending Kite order by order_id."""
    try:
        if not kite_available():
            return json.dumps({"error": "Cancel requires Kite authentication."})
 
        from kiteconnect import KiteConnect
        get_kite().cancel_order(
            variety  = KiteConnect.VARIETY_REGULAR,
            order_id = order_id,
        )
        return json.dumps({"status": "CANCELLED", "order_id": order_id})
 
    except Exception as e:
        return json.dumps({"error": str(e)})
 
 




# ═══════════════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS TOOLS
# ═══════════════════════════════════════════════════════════════════════════
 
def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI, MACD, Bollinger Bands, and EMAs on a OHLCV DataFrame.
    Mutates df in place and returns it.
    """
    close = df["close"]
 
    # RSI
    delta        = close.diff()
    gain         = delta.clip(lower=0).rolling(14).mean()
    loss         = (-delta.clip(upper=0)).rolling(14).mean()
    rs           = gain / loss
    df["rsi"]    = 100 - (100 / (1 + rs))
 
    # MACD
    ema12         = close.ewm(span=12, adjust=False).mean()
    ema26         = close.ewm(span=26, adjust=False).mean()
    df["macd"]    = ema12 - ema26
    df["signal"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["signal"]
 
    # Bollinger Bands
    sma20          = close.rolling(20).mean()
    std20          = close.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_mid"]   = sma20
    df["bb_lower"] = sma20 - 2 * std20
 
    # EMAs
    df["ema9"]  = close.ewm(span=9,  adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
 
    return df
 
 
@tool
def calculate_indicators(symbol: str, days: int = 60) -> str:
    """
    Calculate RSI, MACD, Bollinger Bands, and EMAs for a symbol.
 
    Returns a JSON object with the latest indicator values plus
    a BUY / SELL / HOLD signal summary.
 
    Parameters
    ----------
    symbol : e.g. "NSE:INFY"
    days   : lookback days for data (minimum 60 recommended)
    """
    try:
        raw  = json.loads(get_historical_data.invoke({"symbol": symbol, "days": days}))
        if isinstance(raw, dict) and "error" in raw:
            return json.dumps(raw)
 
        df   = pd.DataFrame(raw)
        df["date"] = pd.to_datetime(df["date"])
        df   = df.sort_values("date").reset_index(drop=True)
        df   = _compute_indicators(df)
 
        last = df.iloc[-1].to_dict()
 
        # ── simple signal logic ──────────────────────────────────────────
        signals = []
        rsi = last.get("rsi")
        if rsi is not None:
            if rsi < 30:
                signals.append("RSI oversold → bullish")
            elif rsi > 70:
                signals.append("RSI overbought → bearish")
 
        macd = last.get("macd")
        sig  = last.get("signal")
        if macd is not None and sig is not None:
            signals.append("MACD above signal → bullish" if macd > sig
                           else "MACD below signal → bearish")
 
        ema9  = last.get("ema9")
        ema21 = last.get("ema21")
        if ema9 and ema21:
            signals.append("EMA9 > EMA21 → uptrend" if ema9 > ema21
                           else "EMA9 < EMA21 → downtrend")
 
        bullish = sum(1 for s in signals if "bullish" in s or "uptrend" in s)
        bearish = sum(1 for s in signals if "bearish" in s or "downtrend" in s)
        overall = "BUY" if bullish > bearish else ("SELL" if bearish > bullish else "HOLD")
 
        last["signals"] = signals
        last["overall_signal"] = overall
        last["symbol"]  = symbol
 
        memory.update_context("analysis_agent", f"indicators_{symbol}", last)
        return json.dumps(last, default=str)
 
    except Exception as e:
        return json.dumps({"error": str(e)})
 
 
@tool
def backtest_strategy(
    symbol  : str,
    strategy: str,
    params  : str = "{}",
    days    : int = 365,
) -> str:
    """
    Backtest a strategy on historical OHLCV data.
 
    Parameters
    ----------
    symbol   : e.g. "NSE:INFY"
    strategy : sma_crossover | rsi_mean_reversion | macd_trend
    params   : JSON string with strategy-specific parameters
    days     : lookback days (default 365)
 
    Returned stats: total_return_pct, buy_hold_pct, sharpe_ratio,
                    max_drawdown_pct, win_rate_pct, num_trades
    """
    try:
        raw = json.loads(get_historical_data.invoke({"symbol": symbol, "days": days}))
        if isinstance(raw, dict) and "error" in raw:
            return json.dumps(raw)
 
        df  = pd.DataFrame(raw)
        df["date"]  = pd.to_datetime(df["date"])
        df  = df.sort_values("date").reset_index(drop=True)
        p   = json.loads(params)
 
        # ── strategy signal generation ───────────────────────────────────
        if strategy == "sma_crossover":
            fast             = p.get("fast", 20)
            slow             = p.get("slow", 50)
            df["sma_fast"]   = df["close"].rolling(fast).mean()
            df["sma_slow"]   = df["close"].rolling(slow).mean()
            df["sig"]        = (df["sma_fast"] > df["sma_slow"]).astype(int)
 
        elif strategy == "rsi_mean_reversion":
            ob     = p.get("overbought", 70)
            ov     = p.get("oversold",   30)
            delta  = df["close"].diff()
            gain   = delta.clip(lower=0).rolling(14).mean()
            loss   = (-delta.clip(upper=0)).rolling(14).mean()
            rsi    = 100 - (100 / (1 + gain / loss))
            df["sig"] = 0
            df.loc[rsi < ov, "sig"] = 1
            df.loc[rsi > ob, "sig"] = 0
 
        elif strategy == "macd_trend":
            ema12        = df["close"].ewm(span=12, adjust=False).mean()
            ema26        = df["close"].ewm(span=26, adjust=False).mean()
            macd         = ema12 - ema26
            sig_line     = macd.ewm(span=9, adjust=False).mean()
            df["sig"]    = ((macd > sig_line) & (macd > 0)).astype(int)
 
        elif strategy == "brownian_motion":
            # Geometric Brownian Motion (GBM) drift-based strategy
            window     = p.get("window", 20)
            threshold  = p.get("threshold", 0.0)
            log_ret    = np.log(df["close"] / df["close"].shift(1))
            
            mu         = log_ret.rolling(window).mean()
            sigma      = log_ret.rolling(window).std()
            
            # Expected return drift in GBM
            drift      = mu + 0.5 * sigma**2
            df["sig"]  = (drift > threshold).astype(int)
            
        else:
            return json.dumps({"error": f"Unknown strategy: {strategy!r}"})
 
        # ── returns computation ──────────────────────────────────────────
        df["position"] = df["sig"].shift(1)
        df["ret"]      = df["close"].pct_change()
        df["strat_ret"]= df["ret"] * df["position"]
        df             = df.dropna()
 
        cum_ret  = float((1 + df["strat_ret"]).cumprod().iloc[-1] - 1)
        bh_ret   = float((df["close"].iloc[-1] / df["close"].iloc[0]) - 1)
        mean_ret = df["strat_ret"].mean()
        std_ret  = df["strat_ret"].std()
        sharpe   = float((mean_ret / std_ret) * (252 ** 0.5)) if std_ret else 0.0
        max_dd   = float(
            (df["strat_ret"].cumsum() - df["strat_ret"].cumsum().cummax()).min()
        )
        win_rate = float((df["strat_ret"] > 0).mean())
        n_trades = int(df["position"].diff().abs().sum())
 
        result = {
            "strategy"         : strategy,
            "symbol"           : symbol,
            "period_days"      : days,
            "total_return_pct" : round(cum_ret  * 100, 2),
            "buy_hold_pct"     : round(bh_ret   * 100, 2),
            "sharpe_ratio"     : round(sharpe,           3),
            "max_drawdown_pct" : round(max_dd   * 100, 2),
            "win_rate_pct"     : round(win_rate * 100, 2),
            "num_trades"       : n_trades,
            "daily_returns"    : df["strat_ret"].tolist(),
            "source"           : "kite" if kite_available() else "yahoo_finance",
        }
 
        memory.save_strategy(f"{symbol}_{strategy}_{days}d", result)
        return json.dumps(result, default=str)
 
    except Exception as e:
        return json.dumps({"error": str(e)})



# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION TOOLS
# ═══════════════════════════════════════════════════════════════════════════
## helper 

def _safe_col(df: pd.DataFrame, col: str, default=0) -> pd.Series:
    """Return df[col] if present, otherwise a constant Series."""
    return df[col] if col in df.columns else pd.Series([default] * len(df))

@tool
def create_candlestick_chart(symbol: str, days: int = 60) -> str:
    """
    Generate an interactive Plotly candlestick chart with volume, EMAs, and RSI.
 
    Returns JSON with the path to the saved HTML file.
 
    Parameters
    ----------
    symbol : e.g. "NSE:INFY"
    days   : number of days of data to display (default 60)
    """
    logger.info("create_candlestick_chart: symbol=%s days=%d", symbol, days)
    try:
        try:
            df = _yf_historical(symbol, interval="day", days=days)
        except Exception as e:
            return json.dumps({"error": f"Yahoo Finance data fetch failed: {e}"})

        if df is None or df.empty:
            return json.dumps({"error": f"No data returned for {symbol}"})

        df.columns = [c.lower() for c in df.columns]
 
        required = {"date", "open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns)
        if missing:
            return json.dumps({"error": f"Missing columns in data: {missing}"})
 
        df["date"] = pd.to_datetime(df["date"])
 
        # ── compute indicators ───────────────────────────────────────────
        df = _compute_indicators(df)
 
        # ── build figure ─────────────────────────────────────────────────
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=[f"{symbol} – Price", "Volume", "RSI"],
        )
 
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df["date"], open=df["open"],
            high=df["high"], low=df["low"], close=df["close"],
            name="Price", increasing_line_color="lime",
            decreasing_line_color="red",
        ), row=1, col=1)
 
        # EMAs
        for span, color in [(9, "orange"), (21, "cyan"), (50, "magenta")]:
            col_name = f"ema{span}"
            if col_name in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df[col_name],
                    name=f"EMA{span}",
                    line=dict(color=color, width=1),
                ), row=1, col=1)
 
        # Bollinger Bands
        for band, color, dash in [
            ("bb_upper", "gray", "dot"),
            ("bb_mid",   "white","dash"),
            ("bb_lower", "gray", "dot"),
        ]:
            if band in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df[band], name=band,
                    line=dict(color=color, width=1, dash=dash),
                    opacity=0.5,
                ), row=1, col=1)
 
        # Volume bars
        vol_colors = [
            "lime" if c >= o else "red"
            for c, o in zip(df["close"], df["open"])
        ]
        fig.add_trace(go.Bar(
            x=df["date"], y=df["volume"],
            marker_color=vol_colors, name="Volume",
        ), row=2, col=1)
 
        # RSI
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["rsi"],
            name="RSI", line=dict(color="purple"),
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red",   row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
 
        fig.update_layout(
            title                    = f"{symbol} – {days}d Chart",
            height                   = 820,
            template                 = "plotly_dark",
            xaxis_rangeslider_visible= False,
        )
 
        name = f"{symbol.replace(':', '_')}_{days}d"
        path = Path("data/visualizations") / f"{name}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))
        memory.save_visualization_meta(name, {
            "symbol": symbol, "days": days,
            "source": "kite" if kite_available() else "yahoo_finance",
        })
        return json.dumps({"chart_path": str(path), "status": "saved", "rows": len(df)})
 
    except Exception as e:
        logger.exception("create_candlestick_chart failed")
        return json.dumps({"error": str(e)})
 
 
@tool
def create_backtest_chart(strategy_name: str) -> str:
    """
    Plot the equity curve and drawdown for a saved backtest.
 
    Parameters
    ----------
    strategy_name : name as saved by backtest_strategy (e.g. "NSE:INFY_sma_crossover_365d")
    """
    try:
        strat = memory.load_strategy(strategy_name)
        if not strat:
            return json.dumps({"error": f"Strategy {strategy_name!r} not found"})
 
        rets     = pd.Series(strat["daily_returns"])
        equity   = (1 + rets).cumprod()
        drawdown = equity / equity.cummax() - 1
 
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Equity Curve", "Drawdown"],
        )
        fig.add_trace(go.Scatter(
            y=equity, name="Strategy",
            fill="tozeroy", line=dict(color="lime"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            y=drawdown, name="Drawdown",
            fill="tozeroy", line=dict(color="red"),
        ), row=2, col=1)
        fig.update_layout(
            title    = f"Backtest: {strategy_name}",
            height   = 600,
            template = "plotly_dark",
        )
 
        path = Path("data/visualizations") / f"{strategy_name}_equity.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))
 
        return json.dumps({
            "chart_path": str(path),
            "stats"     : {k: v for k, v in strat.items() if k != "daily_returns"},
        })
    except Exception as e:
        return json.dumps({"error": str(e)})
 
 
@tool
def create_portfolio_dashboard() -> str:
    """
    Generate a portfolio overview dashboard (requires Kite auth).
    Returns JSON with path to the saved HTML dashboard.
    """
    try:
        raw       = json.loads(get_portfolio.invoke({}))
        if "error" in raw:
            return json.dumps(raw)
 
        holdings  = pd.DataFrame(raw.get("holdings", []))
        if holdings.empty:
            return json.dumps({"error": "No holdings found in portfolio"})
 
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Portfolio Allocation", "P&L per Stock",
                "Day Change %", "Invested vs Current Value",
            ],
        )
 
        # Allocation pie
        fig.add_trace(go.Pie(
            labels = holdings["tradingsymbol"],
            values = holdings["last_price"] * holdings["quantity"],
            name   = "Allocation",
        ), row=1, col=1)
 
        # P&L bars
        pnl_colors = [
            "lime" if p > 0 else "red" for p in holdings["pnl"]
        ]
        fig.add_trace(go.Bar(
            x=holdings["tradingsymbol"],
            y=holdings["pnl"],
            marker_color=pnl_colors,
            name="P&L",
        ), row=1, col=2)
 
        # Day change %
        day_chg = holdings.get("day_change_percentage",
                               pd.Series([0.0] * len(holdings)))
        fig.add_trace(go.Bar(
            x=holdings["tradingsymbol"],
            y=day_chg,
            name="Day Change %",
        ), row=2, col=1)
 
        # Invested vs Current
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
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))
        return json.dumps({"dashboard_path": str(path)})
 
    except Exception as e:
        return json.dumps({"error": str(e)})
 
 
 
# ═══════════════════════════════════════════════════════════════════════════
# FILESYSTEM / MEMORY TOOLS
# ═══════════════════════════════════════════════════════════════════════════
 
@tool
def save_strategy_tool(name: str, strategy_json: str) -> str:
    """Persist a strategy definition or backtest result to disk."""
    try:
        strat = json.loads(strategy_json)
        path  = memory.save_strategy(name, strat)
        return json.dumps({"saved": path})
    except Exception as e:
        return json.dumps({"error": str(e)})
 
 
@tool
def load_strategy_tool(name: str) -> str:
    """Load a previously saved strategy by name."""
    try:
        strat = memory.load_strategy(name)
        return json.dumps(strat or {"error": "not found"})
    except Exception as e:
        return json.dumps({"error": str(e)})
 
 
@tool
def list_strategies_tool() -> str:
    """List all saved strategies on disk."""
    return json.dumps({"strategies": memory.list_strategies()})
 
 
@tool
def get_trade_log_tool(date: str = "") -> str:
    """Get trade log for a date (YYYY-MM-DD). Defaults to today."""
    return json.dumps(memory.get_trade_log(date or None))
 
 
@tool
def list_visualizations_tool() -> str:
    """List all saved visualization HTML files."""
    return json.dumps({"visualizations": memory.list_visualizations()})
 
 
# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Testing Yahoo Finance data layer ===\n")
 
    # Quote
    print("→ get_quote('NSE:INFY')")
    print(get_quote.invoke({"symbols": "NSE:INFY"}))
 
    # Historical
    print("\n→ get_historical_data('NSE:INFY', days=10)")
    raw = get_historical_data.invoke({"symbol": "NSE:INFY", "days": 10})
    df  = pd.DataFrame(json.loads(raw))
    print(df.tail(3))
 
    # Indicators
    print("\n→ calculate_indicators('NSE:INFY', days=90)")
    ind = json.loads(calculate_indicators.invoke({"symbol": "NSE:INFY", "days": 90}))
    for k in ["rsi", "macd", "ema21", "overall_signal"]:
        print(f"   {k}: {ind.get(k)}")
 
    # Chart
    print("\n→ create_candlestick_chart('NSE:INFY', days=60)")
    print(create_candlestick_chart.invoke({"symbol": "NSE:INFY", "days": 60}))
