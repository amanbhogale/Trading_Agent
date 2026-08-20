"""
mongo_etl.py
============
PySpark ETL pipeline: fetches NSE/BSE historical OHLCV data from Yahoo Finance
and writes it to MongoDB Atlas.

Pipeline stages
---------------
1. EXTRACT  – Read active symbols from `securities` collection (or bootstrap list)
2. FETCH    – Download OHLCV via yfinance (concurrent, per-timeframe)
3. TRANSFORM – Compute technical indicators via PySpark + applyInPandas
4. LOAD     – Bulk-upsert into ohlcv / ohlcv_5m / ohlcv_1m via pymongo

Environment variables
---------------------
  MONGO_URI         MongoDB Atlas connection string  (required)
  MONGO_DB_NAME     Database name                    (default: trading_db)
  MONGO_BATCH_SIZE  Upsert batch size                (default: 500)

Entry points
------------
  run_mongo_etl()              called by a service
  if __name__ == '__main__':   standalone execution
"""

import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pymongo
import yfinance as yf
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError, ConnectionFailure, PyMongoError
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("mongo_etl")

# ---------------------------------------------------------------------------
# Environment / config
# ---------------------------------------------------------------------------
MONGO_URI         = os.environ.get("MONGO_URI", "")
MONGO_DB_NAME     = os.environ.get("MONGO_DB_NAME", "trading_db")
MONGO_BATCH_SIZE  = int(os.environ.get("MONGO_BATCH_SIZE", "500"))

MAX_WORKERS       = 8    # concurrent yfinance fetch threads
SYMBOL_BATCH_SIZE = 100  # symbols per yfinance batch download
UPSERT_RETRIES    = 3    # retry attempts for pymongo bulk_write
RETRY_BASE_SLEEP  = 2.0  # exponential back-off base (seconds)

# ---------------------------------------------------------------------------
# Bootstrap NSE symbol list (top 100 liquid stocks on NSE)
# ---------------------------------------------------------------------------
BOOTSTRAP_NSE_SYMBOLS: List[str] = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "BAJFINANCE.NS",
    "BHARTIARTL.NS", "WIPRO.NS", "HINDUNILVR.NS", "ITC.NS", "ONGC.NS",
    "COALINDIA.NS", "NTPC.NS", "POWERGRID.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "MARUTI.NS", "BAJAJFINSV.NS", "TECHM.NS", "HCLTECH.NS", "SUNPHARMA.NS",
    "DRREDDY.NS", "DIVISLAB.NS", "CIPLA.NS", "ADANIPORTS.NS", "ULTRACEMCO.NS",
    "GRASIM.NS", "ASIANPAINT.NS", "TITAN.NS", "NESTLEIND.NS", "BRITANNIA.NS",
    "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HINDALCO.NS", "TATACONSUM.NS",
    "M&M.NS", "INDUSINDBK.NS", "APOLLOHOSP.NS", "ADANIENT.NS", "VEDL.NS",
    "BPCL.NS", "IOC.NS", "HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS",
    "UPL.NS", "PIDILITIND.NS", "GODREJCP.NS", "MCDOWELL-N.NS", "HAVELLS.NS",
    "VOLTAS.NS", "BERGEPAINT.NS", "COLPAL.NS", "MARICO.NS", "DABUR.NS",
    "EMAMILTD.NS", "GLAND.NS", "TORNTPHARM.NS", "ALKEM.NS", "AUROPHARMA.NS",
    "BIOCON.NS", "LUPIN.NS", "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS",
    "RBLBANK.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "UNIONBANK.NS",
    "INDHOTEL.NS", "IRCTC.NS", "CONCOR.NS", "BALKRISIND.NS", "CHOLAFIN.NS",
    "MUTHOOTFIN.NS", "LICHSGFIN.NS", "RECLTD.NS", "PFC.NS", "IRFC.NS",
    "NHPC.NS", "SJVN.NS", "TATAPOWER.NS", "TORNTPOWER.NS", "CESC.NS",
    "JINDALSAW.NS", "SAIL.NS", "NMDC.NS", "HINDCOPPER.NS", "NATIONALUM.NS",
    "AARTIIND.NS", "DEEPAKNITRITE.NS", "TATACHEM.NS", "CHAMBLFERT.NS",
    "COROMANDEL.NS", "PIIND.NS",
]

# ---------------------------------------------------------------------------
# Timeframe configuration
# ---------------------------------------------------------------------------
# (yf_interval, yf_period_or_start, target_collection, timeframe_label, resample_rule)
# resample_rule=None means use raw data from yfinance
TIMEFRAME_CONFIG: List[Dict[str, Any]] = [
    {
        "label":      "1D",
        "yf_interval": "1d",
        "yf_period":  "5y",
        "collection": "ohlcv",
        "resample":   None,
    },
    {
        "label":      "1W",
        "yf_interval": "1d",         # fetch daily, resample to weekly
        "yf_period":  "5y",
        "collection": "ohlcv",
        "resample":   "W",
    },
    {
        "label":      "1M",
        "yf_interval": "1d",         # fetch daily, resample to monthly
        "yf_period":  "5y",
        "collection": "ohlcv",
        "resample":   "ME",          # month-end frequency (pandas >= 2.2)
    },
    {
        "label":      "1h",
        "yf_interval": "1h",
        "yf_period":  "60d",
        "collection": "ohlcv",
        "resample":   None,
    },
    {
        "label":      "15m",
        "yf_interval": "15m",
        "yf_period":  "60d",
        "collection": "ohlcv",
        "resample":   None,
    },
    {
        "label":      "5m",
        "yf_interval": "5m",
        "yf_period":  "60d",
        "collection": "ohlcv_5m",
        "resample":   None,
    },
    {
        "label":      "1m",
        "yf_interval": "1m",
        "yf_period":  "7d",
        "collection": "ohlcv_1m",
        "resample":   None,
    },
]

# ---------------------------------------------------------------------------
# Spark schema for OHLCV
# ---------------------------------------------------------------------------
SPARK_OHLCV_SCHEMA = StructType([
    StructField("symbol",    StringType(),    False),
    StructField("exchange",  StringType(),    False),
    StructField("timeframe", StringType(),    False),
    StructField("ts",        TimestampType(), False),
    StructField("open",      DoubleType(),    True),
    StructField("high",      DoubleType(),    True),
    StructField("low",       DoubleType(),    True),
    StructField("close",     DoubleType(),    True),
    StructField("volume",    LongType(),      True),
    StructField("oi",        LongType(),      True),
    StructField("vwap",      DoubleType(),    True),
])

# ---------------------------------------------------------------------------
# Indicator output schema (added by applyInPandas)
# ---------------------------------------------------------------------------
INDICATOR_SCHEMA = StructType([
    StructField("symbol",      StringType(),    False),
    StructField("exchange",    StringType(),    False),
    StructField("timeframe",   StringType(),    False),
    StructField("ts",          TimestampType(), False),
    StructField("open",        DoubleType(),    True),
    StructField("high",        DoubleType(),    True),
    StructField("low",         DoubleType(),    True),
    StructField("close",       DoubleType(),    True),
    StructField("volume",      LongType(),      True),
    StructField("oi",          LongType(),      True),
    StructField("vwap",        DoubleType(),    True),
    StructField("ema9",        DoubleType(),    True),
    StructField("ema21",       DoubleType(),    True),
    StructField("ema50",       DoubleType(),    True),
    StructField("ema200",      DoubleType(),    True),
    StructField("rsi14",       DoubleType(),    True),
    StructField("macd",        DoubleType(),    True),
    StructField("macd_signal", DoubleType(),    True),
    StructField("macd_hist",   DoubleType(),    True),
    StructField("bb_upper",    DoubleType(),    True),
    StructField("bb_mid",      DoubleType(),    True),
    StructField("bb_lower",    DoubleType(),    True),
    StructField("atr14",       DoubleType(),    True),
])


# ===========================================================================
# MongoDB helpers
# ===========================================================================

def _get_client() -> MongoClient:
    """Return a cached MongoDB client."""
    if not MONGO_URI:
        raise EnvironmentError("MONGO_URI environment variable is not set.")
    client: MongoClient = MongoClient(MONGO_URI, serverSelectionTimeoutMS=20_000)
    client.admin.command("ping")
    log.info("MongoDB connected.")
    return client


def _bulk_upsert_with_retry(
    collection,
    operations: List[UpdateOne],
    batch_size: int = MONGO_BATCH_SIZE,
) -> int:
    """Execute bulk_write in batches with exponential back-off retry.

    Returns total number of documents upserted/modified.
    """
    total_written = 0
    for i in range(0, len(operations), batch_size):
        batch = operations[i : i + batch_size]
        for attempt in range(1, UPSERT_RETRIES + 1):
            try:
                result = collection.bulk_write(batch, ordered=False)
                total_written += result.upserted_count + result.modified_count
                break
            except BulkWriteError as exc:
                n_ok = exc.details.get("nInserted", 0) + exc.details.get("nUpserted", 0) + exc.details.get("nModified", 0)
                total_written += n_ok
                log.warning(
                    "[%s] BulkWriteError on batch %d/%d (attempt %d/%d): %d errors",
                    collection.name,
                    i // batch_size + 1,
                    math.ceil(len(operations) / batch_size),
                    attempt,
                    UPSERT_RETRIES,
                    len(exc.details.get("writeErrors", [])),
                )
                break  # Partial success; no point retrying the same batch
            except PyMongoError as exc:
                if attempt == UPSERT_RETRIES:
                    log.error("[%s] Failed after %d retries: %s", collection.name, UPSERT_RETRIES, exc)
                    raise
                sleep_secs = RETRY_BASE_SLEEP ** attempt
                log.warning(
                    "[%s] Retry %d/%d in %.1fs: %s",
                    collection.name, attempt, UPSERT_RETRIES, sleep_secs, exc,
                )
                time.sleep(sleep_secs)

    return total_written


# ===========================================================================
# Technical indicators (Pandas)
# ===========================================================================

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
          ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast   = _ema(series, fast)
    ema_slow   = _ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0
                     ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Upper band, middle band (SMA), lower band."""
    mid   = series.rolling(window).mean()
    std   = series.rolling(window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday VWAP (cumulative within each date).

    For daily+ timeframes this returns NaN for all rows.
    """
    if "volume" not in df.columns or df["volume"].isna().all():
        return pd.Series(np.nan, index=df.index)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    tpv     = typical * df["volume"]
    # Group by date for intraday reset
    date_grp = df.index.date if hasattr(df.index, "date") else pd.Series(df.index).dt.date.values
    result   = pd.Series(np.nan, index=df.index)
    for dt in np.unique(date_grp):
        mask = np.array(date_grp) == dt
        cum_tpv = tpv[mask].cumsum()
        cum_vol = df["volume"][mask].cumsum()
        result[mask] = cum_tpv / cum_vol.replace(0, np.nan)
    return result


def compute_indicators(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators for one (symbol, timeframe) group.

    This function is designed to be used as the ``func`` argument of
    ``DataFrame.groupBy(...).applyInPandas(func, schema)``.

    Parameters
    ----------
    pdf:
        Pandas DataFrame for a single (symbol, exchange, timeframe) group,
        sorted by ts.

    Returns
    -------
    pd.DataFrame
        Same rows with indicator columns appended.
    """
    pdf = pdf.sort_values("ts").reset_index(drop=True)
    close  = pdf["close"].astype(float)
    high   = pdf["high"].astype(float)
    low    = pdf["low"].astype(float)
    volume = pdf["volume"].astype(float)

    pdf["ema9"]  = _ema(close, 9)
    pdf["ema21"] = _ema(close, 21)
    pdf["ema50"] = _ema(close, 50)
    pdf["ema200"]= _ema(close, 200)
    pdf["rsi14"] = _rsi(close, 14)

    macd_line, macd_sig, macd_hist = _macd(close)
    pdf["macd"]        = macd_line
    pdf["macd_signal"] = macd_sig
    pdf["macd_hist"]   = macd_hist

    bb_upper, bb_mid, bb_lower = _bollinger_bands(close)
    pdf["bb_upper"] = bb_upper
    pdf["bb_mid"]   = bb_mid
    pdf["bb_lower"] = bb_lower

    pdf["atr14"] = _atr(high, low, close, 14)

    # VWAP: meaningful only for intraday bars
    timeframe = pdf["timeframe"].iloc[0] if len(pdf) > 0 else ""
    if timeframe in ("1m", "5m", "15m", "1h"):
        pdf.set_index("ts", inplace=True)
        vwap_series = _vwap(
            pd.DataFrame({"high": high.values, "low": low.values,
                          "close": close.values, "volume": volume.values},
                         index=pdf.index)
        )
        pdf["vwap"] = vwap_series.values
        pdf.reset_index(inplace=True)
    else:
        pdf["vwap"] = np.nan

    # Replace all NaN with None for MongoDB compatibility
    float_cols = ["ema9", "ema21", "ema50", "ema200", "rsi14",
                  "macd", "macd_signal", "macd_hist",
                  "bb_upper", "bb_mid", "bb_lower", "atr14", "vwap"]
    for col in float_cols:
        pdf[col] = pdf[col].where(pdf[col].notna(), other=None)

    # Ensure output columns match INDICATOR_SCHEMA
    out_cols = [
        "symbol", "exchange", "timeframe", "ts",
        "open", "high", "low", "close", "volume", "oi", "vwap",
        "ema9", "ema21", "ema50", "ema200", "rsi14",
        "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_mid", "bb_lower", "atr14",
    ]
    for c in out_cols:
        if c not in pdf.columns:
            pdf[c] = None
    return pdf[out_cols]


# ===========================================================================
# Extract: load active symbols from MongoDB
# ===========================================================================

def _load_symbols_from_mongo(db) -> List[str]:
    """Return Yahoo Finance tickers for active NSE symbols in `securities`."""
    col   = db["securities"]
    count = col.count_documents({"is_active": True, "exchange": "NSE"})
    if count == 0:
        log.info(
            "No active NSE securities found in MongoDB. Using bootstrap list (%d symbols).",
            len(BOOTSTRAP_NSE_SYMBOLS),
        )
        return list(BOOTSTRAP_NSE_SYMBOLS)

    docs    = col.find({"is_active": True, "exchange": "NSE"}, {"symbol": 1})
    symbols = []
    for doc in docs:
        sym = doc.get("symbol", "")
        # Append .NS if not already present
        yf_sym = sym if sym.endswith(".NS") else sym + ".NS"
        symbols.append(yf_sym)
    log.info("Loaded %d active NSE symbols from securities collection.", len(symbols))
    return symbols


# ===========================================================================
# Fetch: yfinance batch download
# ===========================================================================

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample a daily OHLCV DataFrame to weekly ('W') or monthly ('ME')."""
    df = df.sort_index()
    resampled = df.resample(rule).agg(
        {
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }
    ).dropna(how="all")
    return resampled


def _fetch_symbol_batch(
    symbols: List[str],
    yf_interval: str,
    yf_period: str,
    resample_rule: Optional[str],
    timeframe_label: str,
) -> Dict[str, pd.DataFrame]:
    """Fetch a batch of symbols from yfinance and return {yf_ticker: df}."""
    ticker_str = " ".join(symbols)
    log.info(
        "  yfinance fetch: interval=%s  period=%s  symbols=%d  (batch)",
        yf_interval, yf_period, len(symbols),
    )
    try:
        raw = yf.download(
            ticker_str,
            period=yf_period,
            interval=yf_interval,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
    except Exception as exc:
        log.error("  yfinance download failed: %s", exc)
        return {}

    results: Dict[str, pd.DataFrame] = {}

    # When a single symbol is passed yfinance returns a flat DataFrame
    if len(symbols) == 1:
        sym = symbols[0]
        df  = raw.copy()
        if df.empty:
            return results
        df.index = pd.to_datetime(df.index, utc=True)
        if resample_rule:
            df = _resample_ohlcv(df, resample_rule)
            df.index = pd.to_datetime(df.index, utc=True)
        results[sym] = df
        return results

    # Multi-symbol: top-level columns are tickers
    for sym in symbols:
        try:
            df = raw[sym].copy() if sym in raw.columns.get_level_values(0) else pd.DataFrame()
        except (KeyError, AttributeError):
            df = pd.DataFrame()
        if df.empty or df.dropna(how="all").empty:
            log.debug("  No data for %s @ %s", sym, yf_interval)
            continue
        df.index = pd.to_datetime(df.index, utc=True)
        if resample_rule:
            df = _resample_ohlcv(df, resample_rule)
            df.index = pd.to_datetime(df.index, utc=True)
        results[sym] = df

    log.info(
        "  Fetched %d/%d symbols for interval=%s",
        len(results), len(symbols), yf_interval,
    )
    return results


def _fetch_all_symbols_concurrent(
    all_symbols: List[str],
    yf_interval: str,
    yf_period: str,
    resample_rule: Optional[str],
    timeframe_label: str,
) -> Dict[str, pd.DataFrame]:
    """Download all symbols in parallel batches using ThreadPoolExecutor."""
    batches = [
        all_symbols[i : i + SYMBOL_BATCH_SIZE]
        for i in range(0, len(all_symbols), SYMBOL_BATCH_SIZE)
    ]
    log.info(
        "Fetching %d symbols in %d batches for %s ...",
        len(all_symbols), len(batches), timeframe_label,
    )
    all_results: Dict[str, pd.DataFrame] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(
                _fetch_symbol_batch,
                batch, yf_interval, yf_period, resample_rule, timeframe_label,
            ): batch
            for batch in batches
        }
        for future in as_completed(future_map):
            try:
                batch_results = future.result()
                all_results.update(batch_results)
            except Exception as exc:
                batch = future_map[future]
                log.error("Batch fetch error (%s): %s", batch[:3], exc)

    log.info("Total symbols fetched for %s: %d", timeframe_label, len(all_results))
    return all_results


# ===========================================================================
# Transform: build Spark DataFrame and apply indicators
# ===========================================================================

def _df_to_spark_rows(
    symbol_data: Dict[str, pd.DataFrame],
    timeframe_label: str,
) -> List[Dict[str, Any]]:
    """Convert raw {yf_ticker: pandas_df} to a list of row-dicts for Spark."""
    rows: List[Dict[str, Any]] = []
    for yf_sym, df in symbol_data.items():
        # Strip exchange suffix to get base symbol; map to NSE
        base_sym = yf_sym.replace(".NS", "").replace(".BO", "")
        exchange = "BSE" if yf_sym.endswith(".BO") else "NSE"

        for ts, row in df.iterrows():
            open_  = float(row.get("Open",   row.get("open",   np.nan)))
            high   = float(row.get("High",   row.get("high",   np.nan)))
            low_   = float(row.get("Low",    row.get("low",    np.nan)))
            close  = float(row.get("Close",  row.get("close",  np.nan)))
            volume = int(row.get("Volume", row.get("volume", 0)) or 0)

            if any(math.isnan(v) for v in [open_, high, low_, close]):
                continue

            # Normalise timestamp to UTC-aware Python datetime
            if hasattr(ts, "to_pydatetime"):
                ts_dt = ts.to_pydatetime()
            else:
                ts_dt = pd.Timestamp(ts).to_pydatetime()
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)

            rows.append({
                "symbol":    base_sym,
                "exchange":  exchange,
                "timeframe": timeframe_label,
                "ts":        ts_dt,
                "open":      open_,
                "high":      high,
                "low":       low_,
                "close":     close,
                "volume":    volume,
                "oi":        None,
                "vwap":      None,
            })
    return rows


# ===========================================================================
# Load: build pymongo UpdateOne operations and bulk-upsert
# ===========================================================================

def _row_to_update_op(row: Any, ingested_at: datetime) -> UpdateOne:
    """Convert an indicator-enriched Spark Row to a pymongo UpdateOne."""

    def _safe_float(val) -> Optional[float]:
        if val is None:
            return None
        try:
            f = float(val)
            return None if math.isnan(f) or math.isinf(f) else f
        except (TypeError, ValueError):
            return None

    def _safe_int(val) -> Optional[int]:
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    indicators = {
        "ema9":        _safe_float(row.ema9),
        "ema21":       _safe_float(row.ema21),
        "ema50":       _safe_float(row.ema50),
        "ema200":      _safe_float(row.ema200),
        "rsi14":       _safe_float(row.rsi14),
        "macd":        _safe_float(row.macd),
        "macd_signal": _safe_float(row.macd_signal),
        "macd_hist":   _safe_float(row.macd_hist),
        "bb_upper":    _safe_float(row.bb_upper),
        "bb_mid":      _safe_float(row.bb_mid),
        "bb_lower":    _safe_float(row.bb_lower),
        "atr14":       _safe_float(row.atr14),
    }

    ts_val = row.ts
    if hasattr(ts_val, "to_pydatetime"):
        ts_val = ts_val.to_pydatetime()
    if ts_val.tzinfo is None:
        ts_val = ts_val.replace(tzinfo=timezone.utc)

    doc = {
        "symbol":      row.symbol,
        "exchange":    row.exchange,
        "timeframe":   row.timeframe,
        "ts":          ts_val,
        "open":        _safe_float(row.open),
        "high":        _safe_float(row.high),
        "low":         _safe_float(row.low),
        "close":       _safe_float(row.close),
        "volume":      _safe_int(row.volume) or 0,
        "oi":          _safe_int(row.oi),
        "vwap":        _safe_float(row.vwap),
        "indicators":  indicators,
        "ingested_at": ingested_at,
        "source":      "yfinance",
    }

    filter_key = {
        "symbol":    row.symbol,
        "exchange":  row.exchange,
        "timeframe": row.timeframe,
        "ts":        ts_val,
    }
    return UpdateOne(filter_key, {"$set": doc}, upsert=True)


# ===========================================================================
# Securities population
# ===========================================================================

def _upsert_securities(db, yf_tickers: List[str]) -> None:
    """Populate the `securities` collection with basic metadata from yfinance."""
    log.info("Upserting %d securities into 'securities' collection ...", len(yf_tickers))
    col        = db["securities"]
    now        = datetime.now(tz=timezone.utc)
    operations = []

    # Use ThreadPoolExecutor to fetch .info in parallel
    def _fetch_info(yf_sym: str) -> Tuple[str, dict]:
        try:
            ticker = yf.Ticker(yf_sym)
            info   = ticker.info or {}
            return yf_sym, info
        except Exception as exc:
            log.debug("Could not fetch info for %s: %s", yf_sym, exc)
            return yf_sym, {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_info, sym): sym for sym in yf_tickers}
        for future in as_completed(futures):
            yf_sym, info = future.result()
            base_sym = yf_sym.replace(".NS", "").replace(".BO", "")
            exchange = "BSE" if yf_sym.endswith(".BO") else "NSE"

            market_cap = info.get("marketCap", 0) or 0
            if market_cap >= 200_000_000_000:          # >= 200 Bn INR ~ LARGE
                mcc = "LARGE"
            elif market_cap >= 50_000_000_000:          # >= 50 Bn INR  ~ MID
                mcc = "MID"
            elif market_cap > 0:
                mcc = "SMALL"
            else:
                mcc = None

            doc = {
                "symbol":              base_sym,
                "exchange":            exchange,
                "isin":                info.get("isin"),
                "name":                info.get("longName") or info.get("shortName"),
                "sector":              info.get("sector"),
                "industry":            info.get("industry"),
                "series":              "EQ",
                "lot_size":            1,
                "tick_size":           0.05,
                "is_active":           True,
                "listed_date":         None,
                "nse_token":           None,
                "bse_code":            None,
                "market_cap_category": mcc,
                "updated_at":          now,
            }
            operations.append(
                UpdateOne(
                    {"symbol": base_sym, "exchange": exchange},
                    {"$set": doc},
                    upsert=True,
                )
            )

    if operations:
        written = _bulk_upsert_with_retry(col, operations)
        log.info("Securities upserted/modified: %d", written)


# ===========================================================================
# Spark session factory
# ===========================================================================

def _get_spark() -> SparkSession:
    """Create or retrieve the Spark session for ETL."""
    spark = (
        SparkSession.builder
        .appName("TradingMongoDB_ETL")
        .master("local[*]")
        .config("spark.driver.memory",            "6g")
        .config("spark.executor.memory",          "6g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.shuffle.partitions",   "8")
        .config("spark.sql.session.timeZone",     "UTC")
        # Suppress noisy Spark logs
        .config("spark.ui.showConsoleProgress",   "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    log.info("Spark session ready: %s", spark.version)
    return spark


# ===========================================================================
# Main ETL function
# ===========================================================================

def run_mongo_etl() -> None:
    """Entry point for the MongoDB OHLCV ETL pipeline."""
    log.info("=" * 60)
    log.info("Trading System -- MongoDB OHLCV ETL  [%s]",
             datetime.now(tz=timezone.utc).isoformat())
    log.info("=" * 60)

    if not MONGO_URI:
        log.error("MONGO_URI is not set. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Connect to MongoDB
    # ------------------------------------------------------------------
    client = _get_client()
    db     = client[MONGO_DB_NAME]

    # ------------------------------------------------------------------
    # 2. Extract active symbols
    # ------------------------------------------------------------------
    yf_symbols = _load_symbols_from_mongo(db)
    log.info("Total symbols to process: %d", len(yf_symbols))

    # ------------------------------------------------------------------
    # 3. Spark session
    # ------------------------------------------------------------------
    spark = _get_spark()

    # ------------------------------------------------------------------
    # 4. Process each timeframe sequentially
    # ------------------------------------------------------------------
    etl_summary: Dict[str, Any] = {}

    for tf_cfg in TIMEFRAME_CONFIG:
        label      = tf_cfg["label"]
        yf_interval= tf_cfg["yf_interval"]
        yf_period  = tf_cfg["yf_period"]
        coll_name  = tf_cfg["collection"]
        resample   = tf_cfg["resample"]

        log.info("-" * 50)
        log.info("Timeframe: %s  |  yf_interval: %s  |  collection: %s",
                 label, yf_interval, coll_name)

        # ---- FETCH -------------------------------------------------------
        symbol_data = _fetch_all_symbols_concurrent(
            yf_symbols, yf_interval, yf_period, resample, label
        )
        if not symbol_data:
            log.warning("No data fetched for timeframe %s. Skipping.", label)
            etl_summary[label] = {"rows_fetched": 0, "rows_written": 0}
            continue

        # ---- TRANSFORM ---------------------------------------------------
        rows = _df_to_spark_rows(symbol_data, label)
        if not rows:
            log.warning("Empty row list for timeframe %s after conversion.", label)
            etl_summary[label] = {"rows_fetched": 0, "rows_written": 0}
            continue

        log.info("Building Spark DataFrame for %s: %d rows ...", label, len(rows))

        # Convert to pandas first; Spark from-rows can be slow for large lists
        pandas_df = pd.DataFrame(rows)
        pandas_df["ts"] = pd.to_datetime(pandas_df["ts"], utc=True)
        # Ensure correct dtypes
        for fcol in ["open", "high", "low", "close", "vwap"]:
            pandas_df[fcol] = pandas_df[fcol].astype(float)
        pandas_df["volume"] = pandas_df["volume"].fillna(0).astype("int64")
        pandas_df["oi"]     = pandas_df["oi"].where(pandas_df["oi"].notna(), other=None)

        sdf = spark.createDataFrame(pandas_df, schema=SPARK_OHLCV_SCHEMA)

        # Apply indicators grouped by (symbol, exchange, timeframe)
        log.info("Computing indicators for %s ...", label)
        result_sdf = (
            sdf
            .groupBy("symbol", "exchange", "timeframe")
            .applyInPandas(compute_indicators, schema=INDICATOR_SCHEMA)
        )

        # Collect back to driver for MongoDB write
        # For very large datasets consider writing in Spark partitions
        log.info("Collecting %s results to driver ...", label)
        result_rows = result_sdf.collect()
        log.info("Collected %d rows for %s.", len(result_rows), label)

        # ---- LOAD --------------------------------------------------------
        ingested_at = datetime.now(tz=timezone.utc)
        ops = [_row_to_update_op(r, ingested_at) for r in result_rows]

        log.info("Upserting %d documents into '%s' ...", len(ops), coll_name)
        collection   = db[coll_name]
        written      = _bulk_upsert_with_retry(collection, ops)
        log.info("Written %d documents to '%s' for timeframe %s.", written, coll_name, label)

        etl_summary[label] = {
            "rows_fetched": len(rows),
            "rows_written": written,
        }

        # Free Spark cache between timeframes
        sdf.unpersist()
        result_sdf.unpersist()

    # ------------------------------------------------------------------
    # 5. Populate securities metadata
    # ------------------------------------------------------------------
    fetched_symbols = list(symbol_data.keys()) if symbol_data else yf_symbols
    _upsert_securities(db, fetched_symbols)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 60)
    log.info("ETL COMPLETE -- SUMMARY")
    log.info("=" * 60)
    log.info("%-6s  %-14s  %-14s", "TF", "Rows fetched", "Rows written")
    log.info("-" * 40)
    for tf, stats in etl_summary.items():
        log.info("%-6s  %-14d  %-14d", tf, stats["rows_fetched"], stats["rows_written"])
    log.info("=" * 60)
    log.info(
        "ETL finished at %s", datetime.now(tz=timezone.utc).isoformat()
    )

    spark.stop()
    client.close()


# ===========================================================================
# Standalone entry point
# ===========================================================================

if __name__ == "__main__":
    run_mongo_etl()
