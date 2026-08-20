"""
Data ETL Microservice
Wraps pyspark_etl.py and exposes REST endpoints to trigger/monitor the ETL pipeline,
query active tickers, and retrieve precomputed OHLCV indicators.
"""
import os, sys, logging, threading
from datetime import datetime
from typing import Optional

sys.path.insert(0, "/app")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data-etl-service")

app = FastAPI(title="Data ETL Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── ETL Job State ──────────────────────────────────────────────────────────────
_etl_lock = threading.Lock()
_etl_status = {
    "running": False,
    "last_run_at": None,
    "last_run_status": "never_run",
    "last_error": None,
    "duration_seconds": None,
}


def _run_etl_background():
    """Execute the full PySpark ETL pipeline in a background thread."""
    import time
    global _etl_status

    with _etl_lock:
        _etl_status["running"] = True
        _etl_status["last_run_at"] = datetime.utcnow().isoformat()
        _etl_status["last_run_status"] = "running"
        _etl_status["last_error"] = None

    start = time.time()
    try:
        # Import here to avoid loading PySpark at startup (heavy!)
        from src.trading_system.pyspark_etl import run_etl
        run_etl()
        duration = round(time.time() - start, 2)
        with _etl_lock:
            _etl_status["running"] = False
            _etl_status["last_run_status"] = "success"
            _etl_status["duration_seconds"] = duration
        logger.info("ETL completed in %.2fs", duration)
    except Exception as e:
        duration = round(time.time() - start, 2)
        with _etl_lock:
            _etl_status["running"] = False
            _etl_status["last_run_status"] = "failed"
            _etl_status["last_error"] = str(e)
            _etl_status["duration_seconds"] = duration
        logger.exception("ETL pipeline failed")


def _get_db_conn():
    """Return a psycopg2 connection using env-configured DB params."""
    import psycopg2
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME", "trading_db"),
        user=os.getenv("DB_USER", "trading_agent"),
        password=os.getenv("DB_PASSWORD", "zombie612@"),
        host=os.getenv("DB_HOST", "db"),
        port=os.getenv("DB_PORT", "5432"),
    )


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "data-etl"}


@app.post("/run-etl")
def trigger_etl(background_tasks: BackgroundTasks):
    """Trigger the PySpark ETL pipeline asynchronously."""
    if _etl_status["running"]:
        return {"status": "already_running", "message": "ETL pipeline is already running."}
    background_tasks.add_task(_run_etl_background)
    return {"status": "started", "message": "ETL pipeline started in the background."}


@app.get("/etl-status")
def get_etl_status():
    """Return the current ETL job status."""
    with _etl_lock:
        return dict(_etl_status)


@app.get("/tickers")
def get_tickers(market_mode: Optional[str] = None):
    """
    Return active tickers from the database.
    Optionally filter by market_mode: equity | forex | crypto
    """
    try:
        conn = _get_db_conn()
        cur = conn.cursor()
        if market_mode:
            cur.execute(
                "SELECT symbol, name, sector, market_mode FROM tickers_classification "
                "WHERE status = 'Active' AND market_mode = %s ORDER BY symbol;",
                (market_mode,),
            )
        else:
            cur.execute(
                "SELECT symbol, name, sector, market_mode FROM tickers_classification "
                "WHERE status = 'Active' ORDER BY symbol;"
            )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {"symbol": r[0], "name": r[1], "sector": r[2], "market_mode": r[3]}
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/indicators/{symbol}")
def get_indicators(symbol: str, limit: int = 100):
    """
    Return precomputed OHLCV + indicator data for a symbol.
    Falls back to the Parquet cache if the DB table doesn't exist yet.
    """
    # 1. Try DB first
    try:
        conn = _get_db_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT symbol, time, open, high, low, close, volume,
                   ema9, ema21, ema50, ema200, rsi14, macd, signal,
                   bb_upper, bb_mid, bb_lower, precomputed_at
            FROM precomputed_ohlcv_indicators
            WHERE symbol = %s
            ORDER BY time DESC
            LIMIT %s;
            """,
            (symbol, limit),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if rows:
            cols = ["symbol", "time", "open", "high", "low", "close", "volume",
                    "ema9", "ema21", "ema50", "ema200", "rsi14", "macd", "signal",
                    "bb_upper", "bb_mid", "bb_lower", "precomputed_at"]
            return [dict(zip(cols, r)) for r in rows]
    except Exception as db_err:
        logger.warning("DB query failed for %s: %s — trying Parquet fallback", symbol, db_err)

    # 2. Parquet cache fallback
    safe_symbol = symbol.replace(":", "_")
    parquet_path = os.path.join("/app/data/precomputed", f"{safe_symbol}.parquet")
    if os.path.exists(parquet_path):
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            df = df.sort_values("time", ascending=False).head(limit)
            return df.to_dict(orient="records")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Parquet read failed: {e}")

    raise HTTPException(
        status_code=404,
        detail=f"No precomputed data found for symbol '{symbol}'. Run /run-etl first.",
    )


@app.get("/indicators/{symbol}/latest")
def get_latest_indicators(symbol: str):
    """Return only the most recent row of indicators for a symbol (useful for live dashboards)."""
    data = get_indicators(symbol, limit=1)
    if not data:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    return data[0]
