"""
MongoDB ETL Microservice
=========================
FastAPI wrapper around the MongoDB ETL pipeline (src/trading_system/mongo_etl.py).
Exposes REST endpoints to trigger and monitor ETL runs.

Environment Variables:
    MONGO_URI      - MongoDB connection URI (required)
    MONGO_DB_NAME  - MongoDB database name (default: trading_db)
"""

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] service=mongo-etl %(name)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S%z',
)
logger = logging.getLogger("mongo-etl-service")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MONGO_URI: str = os.getenv("MONGO_URI", "")
MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "trading_db")

VALID_TIMEFRAMES = {"1m", "5m", "15m", "1h", "1D", "1W", "1M"}

# ---------------------------------------------------------------------------
# ETL state  (protected by a threading.Lock)
# ---------------------------------------------------------------------------

_etl_lock = threading.Lock()

_etl_state: Dict[str, Any] = {
    "running": False,
    "last_run_at": None,
    "last_run_status": "never_run",   # "never_run" | "success" | "failed"
    "last_error": None,
    "duration_seconds": None,
    "symbols_processed": 0,
    "records_written": 0,
}

# ---------------------------------------------------------------------------
# MongoDB client (module-level, initialised at startup)
# ---------------------------------------------------------------------------

_mongo_client: Optional[MongoClient] = None


def _get_db():
    """Return the MongoDB database handle, or raise if not connected."""
    if _mongo_client is None:
        raise RuntimeError("MongoDB client is not initialised.")
    return _mongo_client[MONGO_DB_NAME]


# ---------------------------------------------------------------------------
# ETL runner helpers
# ---------------------------------------------------------------------------

def _run_etl_background(timeframe: Optional[str] = None) -> None:
    """
    Execute the ETL pipeline in a background thread.

    Imports are deferred to avoid import-time failures when the ETL
    dependencies (PySpark, etc.) are not available during development.

    Args:
        timeframe: If provided, run ETL for this timeframe only.
                   If None, run the full pipeline.
    """
    global _etl_state

    start_ts = datetime.now(timezone.utc)
    logger.info(
        "ETL run started. timeframe=%s", timeframe if timeframe else "all"
    )

    symbols_processed = 0
    records_written = 0

    try:
        # Deferred import to keep startup fast when ETL deps may not yet be installed
        from src.trading_system.mongo_etl import run_mongo_etl  # type: ignore

        result = run_mongo_etl(timeframe=timeframe) if timeframe else run_mongo_etl()

        # run_mongo_etl may return a dict with run statistics; handle both cases
        if isinstance(result, dict):
            symbols_processed = result.get("symbols_processed", 0)
            records_written = result.get("records_written", 0)

        duration = (datetime.now(timezone.utc) - start_ts).total_seconds()

        with _etl_lock:
            _etl_state["running"] = False
            _etl_state["last_run_at"] = start_ts.isoformat()
            _etl_state["last_run_status"] = "success"
            _etl_state["last_error"] = None
            _etl_state["duration_seconds"] = round(duration, 3)
            _etl_state["symbols_processed"] = symbols_processed
            _etl_state["records_written"] = records_written

        logger.info(
            "ETL run completed successfully. duration=%.3fs symbols=%d records=%d",
            duration, symbols_processed, records_written,
        )

    except Exception as exc:
        duration = (datetime.now(timezone.utc) - start_ts).total_seconds()
        logger.exception("ETL run failed: %s", exc)

        with _etl_lock:
            _etl_state["running"] = False
            _etl_state["last_run_at"] = start_ts.isoformat()
            _etl_state["last_run_status"] = "failed"
            _etl_state["last_error"] = str(exc)
            _etl_state["duration_seconds"] = round(duration, 3)
            _etl_state["symbols_processed"] = symbols_processed
            _etl_state["records_written"] = records_written


def _start_etl_thread(timeframe: Optional[str] = None) -> bool:
    """
    Attempt to start the ETL background thread.

    Returns:
        True  if the thread was started successfully.
        False if ETL is already running.
    """
    with _etl_lock:
        if _etl_state["running"]:
            return False
        _etl_state["running"] = True

    thread = threading.Thread(
        target=_run_etl_background,
        args=(timeframe,),
        name=f"mongo-etl-{'all' if timeframe is None else timeframe}",
        daemon=True,
    )
    thread.start()
    logger.info("ETL thread started. timeframe=%s", timeframe if timeframe else "all")
    return True


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise MongoDB connection on startup; close on shutdown."""
    global _mongo_client

    logger.info("Starting mongo-etl service...")

    if MONGO_URI:
        try:
            _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
            _mongo_client.admin.command("ping")
            logger.info("MongoDB connection established. DB=%s", MONGO_DB_NAME)
        except Exception as exc:
            logger.error("MongoDB connection failed: %s", exc)
            _mongo_client = None
    else:
        logger.warning("MONGO_URI is not set; MongoDB-dependent endpoints will fail.")

    yield  # Application is running

    logger.info("Shutting down mongo-etl service...")
    if _mongo_client is not None:
        try:
            _mongo_client.close()
            logger.info("MongoDB connection closed.")
        except Exception as exc:
            logger.error("Error closing MongoDB connection: %s", exc)

    logger.info("mongo-etl service stopped.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MongoDB ETL Service",
    description="Wraps the MongoDB ETL pipeline and exposes REST endpoints for triggering and monitoring runs.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health() -> JSONResponse:
    """Return service health status."""
    return JSONResponse({"status": "ok", "service": "mongo-etl"})


@app.post("/run-etl", summary="Trigger full ETL run")
async def run_etl() -> JSONResponse:
    """
    Trigger the full ETL pipeline in a background thread.

    Returns:
        {status: "started"} if the run was queued.
        {status: "already_running"} if a run is already in progress.
    """
    started = _start_etl_thread(timeframe=None)
    if started:
        return JSONResponse({"status": "started"})
    return JSONResponse({"status": "already_running"})


@app.post("/run-etl/{timeframe}", summary="Trigger ETL for a specific timeframe")
async def run_etl_timeframe(timeframe: str) -> JSONResponse:
    """
    Trigger ETL for a single timeframe (1m / 5m / 15m / 1h / 1D / 1W / 1M).

    Returns:
        {status: "started"} if the run was queued.
        {status: "already_running"} if a run is already in progress.
        HTTP 422 if the timeframe is invalid.
    """
    if timeframe not in VALID_TIMEFRAMES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid timeframe '{timeframe}'. Valid values: {sorted(VALID_TIMEFRAMES)}",
        )
    started = _start_etl_thread(timeframe=timeframe)
    if started:
        return JSONResponse({"status": "started", "timeframe": timeframe})
    return JSONResponse({"status": "already_running"})


@app.get("/etl-status", summary="Get ETL run status")
async def etl_status() -> JSONResponse:
    """Return the current or last ETL run status."""
    with _etl_lock:
        snapshot = {
            "running": _etl_state["running"],
            "last_run_at": _etl_state["last_run_at"],
            "last_run_status": _etl_state["last_run_status"],
            "last_error": _etl_state["last_error"],
            "duration_seconds": _etl_state["duration_seconds"],
            "symbols_processed": _etl_state["symbols_processed"],
            "records_written": _etl_state["records_written"],
        }
    return JSONResponse(snapshot)


@app.post("/init-schema", summary="Initialise MongoDB schema (collections + indexes)")
async def init_schema() -> JSONResponse:
    """
    Call mongo_schema.init_mongo_db() to create collections and indexes.
    This is idempotent and safe to call multiple times.
    """
    try:
        from src.trading_system.mongo_schema import init_mongo_db  # type: ignore

        uri = os.getenv("MONGO_URI")
        db_name = os.getenv("MONGO_DB_NAME", "trading_db")
        if not uri:
            raise HTTPException(status_code=500, detail="MONGO_URI environment variable is not set.")

        init_mongo_db(uri, db_name)
        logger.info("MongoDB schema initialised successfully.")
        return JSONResponse({"status": "ok", "detail": f"Schema initialised for db '{db_name}'."})
    except ImportError as exc:
        logger.error("Could not import mongo_schema: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Could not import mongo_schema module: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Schema initialisation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Schema initialisation failed: {exc}",
        ) from exc


@app.get("/securities/count", summary="Count of securities in MongoDB")
async def securities_count() -> JSONResponse:
    """Return the number of documents in the 'securities' collection."""
    try:
        db = _get_db()
        count = db["securities"].count_documents({})
        return JSONResponse({"securities": count})
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PyMongoError as exc:
        logger.error("MongoDB error in /securities/count: %s", exc)
        raise HTTPException(status_code=500, detail=f"MongoDB error: {exc}") from exc


@app.get("/ohlcv/count", summary="Count of OHLCV and related documents")
async def ohlcv_count() -> JSONResponse:
    """
    Return document counts for all OHLCV and market data collections:
    ohlcv, ohlcv_5m, ohlcv_1m, orderbook_snapshots, trade_prints.
    """
    collections = ["ohlcv", "ohlcv_5m", "ohlcv_1m", "orderbook_snapshots", "trade_prints"]
    try:
        db = _get_db()
        counts: Dict[str, int] = {}
        for col_name in collections:
            try:
                counts[col_name] = db[col_name].count_documents({})
            except PyMongoError as exc:
                logger.warning("Could not count '%s': %s", col_name, exc)
                counts[col_name] = -1
        return JSONResponse(counts)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Entry point (for local development)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        timeout_keep_alive=300,
    )
