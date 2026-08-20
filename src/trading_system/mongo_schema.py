"""
mongo_schema.py
===============
Initializes all MongoDB collections, indexes, TTL policies, and JSON Schema
validation rules for the trading system.

Target: MongoDB Atlas cluster at URI = env[MONGO_URI]
        Database                  = env[MONGO_DB_NAME]

Collections
-----------
  securities          – Master NSE/BSE ticker registry
  ohlcv               – OHLCV bars for timeframes 15m and above (no TTL)
  ohlcv_5m            – 5-minute bars with 90-day TTL
  ohlcv_1m            – 1-minute bars with 30-day TTL
  orderbook_snapshots – L2 full-depth order book snapshots (7-day TTL)
  trade_prints        – Tick-by-tick trade executions (7-day TTL)

Usage
-----
  python mongo_schema.py           # reads MONGO_URI / MONGO_DB_NAME from env
  from mongo_schema import init_mongo_db
"""

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

from pymongo import ASCENDING, DESCENDING, MongoClient, IndexModel
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import (
    CollectionInvalid,
    ConnectionFailure,
    OperationFailure,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("mongo_schema")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECONDS_PER_DAY = 86_400

TTL_1M_DAYS   = 30   # ohlcv_1m  -> 30-day expiry
TTL_5M_DAYS   = 90   # ohlcv_5m  -> 90-day expiry
TTL_OB_DAYS   = 7    # orderbook_snapshots -> 7-day expiry
TTL_TICK_DAYS = 7    # trade_prints -> 7-day expiry


# ===========================================================================
# Client / DB helpers
# ===========================================================================

def get_mongo_client(uri: str) -> MongoClient:
    """Return a MongoClient connected to *uri*.

    Raises ``ConnectionFailure`` if the server is unreachable.
    """
    log.info("Connecting to MongoDB ...")
    client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=15_000)
    # Force a round-trip to verify connectivity
    client.admin.command("ping")
    log.info("MongoDB connection established.")
    return client


def get_db(client: MongoClient, db_name: str) -> Database:
    """Return the ``Database`` object for *db_name*."""
    log.info("Using database: %s", db_name)
    return client[db_name]


# ===========================================================================
# Helpers
# ===========================================================================

def _ensure_collection(
    db: Database, name: str, validator: Optional[dict] = None
) -> Collection:
    """Create *name* if it does not exist; optionally attach a JSON Schema validator."""
    existing = db.list_collection_names()
    if name not in existing:
        kwargs = {}
        if validator:
            kwargs["validator"] = validator
            kwargs["validationLevel"] = "moderate"  # warn but never block writes
            kwargs["validationAction"] = "warn"
        try:
            db.create_collection(name, **kwargs)
            log.info("  [CREATE]  collection '%s'", name)
        except CollectionInvalid:
            log.warning("  [EXISTS]  collection '%s' already exists (race?)", name)
    else:
        log.info("  [EXISTS]  collection '%s'", name)
        # Still (re-)apply / update the validator if supplied
        if validator:
            try:
                db.command(
                    "collMod",
                    name,
                    validator=validator,
                    validationLevel="moderate",
                    validationAction="warn",
                )
                log.info("  [VALIDATOR UPDATED] '%s'", name)
            except OperationFailure as exc:
                log.warning("  Could not update validator for '%s': %s", name, exc)
    return db[name]


def _apply_indexes(collection: Collection, index_models: list) -> None:
    """Create indexes on *collection*, skipping any that already exist."""
    if not index_models:
        return
    try:
        result = collection.create_indexes(index_models)
        for name in result:
            log.info("    [INDEX]  %s.%s", collection.name, name)
    except OperationFailure as exc:
        log.error("    [INDEX ERROR] %s: %s", collection.name, exc)
        raise


# ===========================================================================
# JSON Schema validators
# ===========================================================================

def _securities_validator() -> dict:
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["symbol", "exchange", "is_active"],
            "properties": {
                "symbol":              {"bsonType": "string",  "description": "Ticker symbol"},
                "exchange":            {"bsonType": "string",  "enum": ["NSE", "BSE"]},
                "isin":                {"bsonType": ["string", "null"]},
                "name":                {"bsonType": ["string", "null"]},
                "sector":              {"bsonType": ["string", "null"]},
                "industry":            {"bsonType": ["string", "null"]},
                "series":              {"bsonType": ["string", "null"],
                                        "enum": ["EQ", "BE", "SM", None]},
                "lot_size":            {"bsonType": ["int",    "null"]},
                "tick_size":           {"bsonType": ["double", "null"]},
                "is_active":           {"bsonType": "bool"},
                "listed_date":         {"bsonType": ["date",   "null"]},
                "nse_token":           {"bsonType": ["int",    "null"]},
                "bse_code":            {"bsonType": ["string", "null"]},
                "market_cap_category": {
                    "bsonType": ["string", "null"],
                    "enum": ["LARGE", "MID", "SMALL", None],
                },
                "updated_at":          {"bsonType": ["date",   "null"]},
            },
        }
    }


def _ohlcv_validator(timeframe_enum: Optional[list] = None) -> dict:
    """Shared OHLCV validator; *timeframe_enum* restricts allowed timeframe values."""
    tf_rule: dict = {"bsonType": "string"}
    if timeframe_enum:
        tf_rule["enum"] = timeframe_enum
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["symbol", "exchange", "timeframe", "ts",
                         "open", "high", "low", "close", "volume"],
            "properties": {
                "symbol":     {"bsonType": "string"},
                "exchange":   {"bsonType": "string", "enum": ["NSE", "BSE"]},
                "timeframe":  tf_rule,
                "ts":         {"bsonType": "date",   "description": "Bar-open UTC timestamp"},
                "open":       {"bsonType": "double"},
                "high":       {"bsonType": "double"},
                "low":        {"bsonType": "double"},
                "close":      {"bsonType": "double"},
                "volume":     {"bsonType": "int"},
                "oi":         {"bsonType": ["int",    "null"]},
                "vwap":       {"bsonType": ["double", "null"]},
                "indicators": {
                    "bsonType": ["object", "null"],
                    "properties": {
                        "ema9":        {"bsonType": ["double", "null"]},
                        "ema21":       {"bsonType": ["double", "null"]},
                        "ema50":       {"bsonType": ["double", "null"]},
                        "ema200":      {"bsonType": ["double", "null"]},
                        "rsi14":       {"bsonType": ["double", "null"]},
                        "macd":        {"bsonType": ["double", "null"]},
                        "macd_signal": {"bsonType": ["double", "null"]},
                        "macd_hist":   {"bsonType": ["double", "null"]},
                        "bb_upper":    {"bsonType": ["double", "null"]},
                        "bb_mid":      {"bsonType": ["double", "null"]},
                        "bb_lower":    {"bsonType": ["double", "null"]},
                        "atr14":       {"bsonType": ["double", "null"]},
                    },
                },
                "ingested_at": {"bsonType": ["date", "null"]},
                "source":      {
                    "bsonType": ["string", "null"],
                    "enum": ["yfinance", "kite", "nse_direct", None],
                },
            },
        }
    }


def _orderbook_validator() -> dict:
    level_schema = {
        "bsonType": "object",
        "properties": {
            "price":  {"bsonType": "double"},
            "qty":    {"bsonType": "int"},
            "orders": {"bsonType": ["int", "null"]},
        },
    }
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["symbol", "exchange", "ts"],
            "properties": {
                "symbol":        {"bsonType": "string"},
                "exchange":      {"bsonType": "string", "enum": ["NSE", "BSE"]},
                "ts":            {"bsonType": "date"},
                "seq":           {"bsonType": ["int",    "null"]},
                "bids":          {
                    "bsonType": ["array", "null"],
                    "items": level_schema,
                    "maxItems": 20,
                },
                "asks":          {
                    "bsonType": ["array", "null"],
                    "items": level_schema,
                    "maxItems": 20,
                },
                "ltp":           {"bsonType": ["double", "null"]},
                "ltq":           {"bsonType": ["int",    "null"]},
                "total_bid_qty": {"bsonType": ["int",    "null"]},
                "total_ask_qty": {"bsonType": ["int",    "null"]},
                "source":        {
                    "bsonType": ["string", "null"],
                    "enum": ["kite_ws", "nse_direct", None],
                },
            },
        }
    }


def _trade_prints_validator() -> dict:
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["symbol", "exchange", "ts", "price", "qty"],
            "properties": {
                "symbol":   {"bsonType": "string"},
                "exchange": {"bsonType": "string", "enum": ["NSE", "BSE"]},
                "ts":       {"bsonType": "date"},
                "price":    {"bsonType": "double"},
                "qty":      {"bsonType": "int"},
                "side":     {"bsonType": ["string", "null"], "enum": ["B", "S", None]},
                "trade_id": {"bsonType": ["string", "null"]},
                "order_id": {"bsonType": ["string", "null"]},
                "source":   {"bsonType": ["string", "null"]},
            },
        }
    }


# ===========================================================================
# Per-collection initializers
# ===========================================================================

def _init_securities(db: Database) -> Collection:
    """Create and configure the *securities* collection."""
    log.info("Initializing 'securities' ...")
    col = _ensure_collection(db, "securities", _securities_validator())

    indexes = [
        # Unique master key
        IndexModel(
            [("symbol", ASCENDING), ("exchange", ASCENDING)],
            unique=True,
            name="ux_symbol_exchange",
        ),
        IndexModel([("isin",      ASCENDING)], name="ix_isin"),
        IndexModel([("is_active", ASCENDING)], name="ix_is_active"),
        IndexModel([("sector",    ASCENDING)], name="ix_sector"),
    ]
    _apply_indexes(col, indexes)
    return col


def _init_ohlcv(db: Database) -> Collection:
    """Create and configure the main *ohlcv* collection (15m and above, no TTL)."""
    log.info("Initializing 'ohlcv' (15m+) ...")
    allowed_tf = ["15m", "1h", "1D", "1W", "1M"]
    col = _ensure_collection(db, "ohlcv", _ohlcv_validator(allowed_tf))

    indexes = [
        # Primary unique key for upserts
        IndexModel(
            [("symbol", ASCENDING), ("exchange", ASCENDING),
             ("timeframe", ASCENDING), ("ts", ASCENDING)],
            unique=True,
            name="ux_symbol_exchange_tf_ts",
        ),
        # Fast per-symbol queries across timeframes
        IndexModel(
            [("symbol", ASCENDING), ("timeframe", ASCENDING), ("ts", DESCENDING)],
            name="ix_symbol_tf_ts",
        ),
        # Market-wide scans by exchange + timeframe + time-range
        IndexModel(
            [("exchange", ASCENDING), ("timeframe", ASCENDING), ("ts", DESCENDING)],
            name="ix_exchange_tf_ts",
        ),
    ]
    _apply_indexes(col, indexes)
    return col


def _init_ohlcv_5m(db: Database) -> Collection:
    """Create and configure the *ohlcv_5m* collection with 90-day TTL."""
    log.info("Initializing 'ohlcv_5m' (5-minute bars, 90-day TTL) ...")
    col = _ensure_collection(db, "ohlcv_5m", _ohlcv_validator(["5m"]))

    indexes = [
        IndexModel(
            [("symbol", ASCENDING), ("exchange", ASCENDING),
             ("timeframe", ASCENDING), ("ts", ASCENDING)],
            unique=True,
            name="ux_symbol_exchange_tf_ts",
        ),
        IndexModel(
            [("symbol", ASCENDING), ("timeframe", ASCENDING), ("ts", DESCENDING)],
            name="ix_symbol_tf_ts",
        ),
        IndexModel(
            [("exchange", ASCENDING), ("timeframe", ASCENDING), ("ts", DESCENDING)],
            name="ix_exchange_tf_ts",
        ),
        # TTL index on ts field – expire documents 90 days after bar timestamp
        IndexModel(
            [("ts", ASCENDING)],
            expireAfterSeconds=TTL_5M_DAYS * SECONDS_PER_DAY,
            name="ttl_ts_90d",
        ),
    ]
    _apply_indexes(col, indexes)
    return col


def _init_ohlcv_1m(db: Database) -> Collection:
    """Create and configure the *ohlcv_1m* collection with 30-day TTL."""
    log.info("Initializing 'ohlcv_1m' (1-minute bars, 30-day TTL) ...")
    col = _ensure_collection(db, "ohlcv_1m", _ohlcv_validator(["1m"]))

    indexes = [
        IndexModel(
            [("symbol", ASCENDING), ("exchange", ASCENDING),
             ("timeframe", ASCENDING), ("ts", ASCENDING)],
            unique=True,
            name="ux_symbol_exchange_tf_ts",
        ),
        IndexModel(
            [("symbol", ASCENDING), ("timeframe", ASCENDING), ("ts", DESCENDING)],
            name="ix_symbol_tf_ts",
        ),
        IndexModel(
            [("exchange", ASCENDING), ("timeframe", ASCENDING), ("ts", DESCENDING)],
            name="ix_exchange_tf_ts",
        ),
        # TTL index on ts field – expire documents 30 days after bar timestamp
        IndexModel(
            [("ts", ASCENDING)],
            expireAfterSeconds=TTL_1M_DAYS * SECONDS_PER_DAY,
            name="ttl_ts_30d",
        ),
    ]
    _apply_indexes(col, indexes)
    return col


def _init_orderbook_snapshots(db: Database) -> Collection:
    """Create and configure the *orderbook_snapshots* collection with 7-day TTL."""
    log.info("Initializing 'orderbook_snapshots' ...")
    col = _ensure_collection(db, "orderbook_snapshots", _orderbook_validator())

    indexes = [
        IndexModel(
            [("symbol", ASCENDING), ("exchange", ASCENDING), ("ts", DESCENDING)],
            name="ix_symbol_exchange_ts",
        ),
        # Standalone ts index for TTL
        IndexModel(
            [("ts", ASCENDING)],
            expireAfterSeconds=TTL_OB_DAYS * SECONDS_PER_DAY,
            name="ttl_ts_7d",
        ),
    ]
    _apply_indexes(col, indexes)
    return col


def _init_trade_prints(db: Database) -> Collection:
    """Create and configure the *trade_prints* collection with 7-day TTL."""
    log.info("Initializing 'trade_prints' ...")
    col = _ensure_collection(db, "trade_prints", _trade_prints_validator())

    indexes = [
        IndexModel(
            [("symbol", ASCENDING), ("exchange", ASCENDING), ("ts", DESCENDING)],
            name="ix_symbol_exchange_ts",
        ),
        # Unique trade identifier (sparse to allow many null trade_ids)
        IndexModel(
            [("trade_id", ASCENDING)],
            unique=True,
            sparse=True,
            name="ux_trade_id",
        ),
        # TTL index
        IndexModel(
            [("ts", ASCENDING)],
            expireAfterSeconds=TTL_TICK_DAYS * SECONDS_PER_DAY,
            name="ttl_ts_7d",
        ),
    ]
    _apply_indexes(col, indexes)
    return col


# ===========================================================================
# Main initializer
# ===========================================================================

def init_mongo_db(uri: str, db_name: str) -> Database:
    """Create all collections, indexes, and validators.

    Parameters
    ----------
    uri:
        MongoDB connection URI (e.g. ``mongodb+srv://user:pass@cluster/``).
    db_name:
        Target database name.

    Returns
    -------
    pymongo.database.Database
        The configured database object.
    """
    log.info("=" * 60)
    log.info("Trading System -- MongoDB Schema Initializer")
    log.info("Database : %s", db_name)
    log.info("=" * 60)

    client = get_mongo_client(uri)
    db     = get_db(client, db_name)

    collections_created = {}

    # ---- securities --------------------------------------------------------
    col = _init_securities(db)
    collections_created["securities"] = {
        "purpose": "Master NSE/BSE ticker registry",
        "ttl":     "none",
        "indexes": list(col.index_information().keys()),
    }

    # ---- ohlcv (15m+) ------------------------------------------------------
    col = _init_ohlcv(db)
    collections_created["ohlcv"] = {
        "purpose": "OHLCV bars: 15m, 1h, 1D, 1W, 1M -- no TTL",
        "ttl":     "none",
        "indexes": list(col.index_information().keys()),
    }

    # ---- ohlcv_5m ----------------------------------------------------------
    col = _init_ohlcv_5m(db)
    collections_created["ohlcv_5m"] = {
        "purpose": "5-minute OHLCV bars",
        "ttl":     "%d days" % TTL_5M_DAYS,
        "indexes": list(col.index_information().keys()),
    }

    # ---- ohlcv_1m ----------------------------------------------------------
    col = _init_ohlcv_1m(db)
    collections_created["ohlcv_1m"] = {
        "purpose": "1-minute OHLCV bars",
        "ttl":     "%d days" % TTL_1M_DAYS,
        "indexes": list(col.index_information().keys()),
    }

    # ---- orderbook_snapshots -----------------------------------------------
    col = _init_orderbook_snapshots(db)
    collections_created["orderbook_snapshots"] = {
        "purpose": "L2 full-depth order book snapshots",
        "ttl":     "%d days" % TTL_OB_DAYS,
        "indexes": list(col.index_information().keys()),
    }

    # ---- trade_prints -------------------------------------------------------
    col = _init_trade_prints(db)
    collections_created["trade_prints"] = {
        "purpose": "Tick-by-tick trade executions",
        "ttl":     "%d days" % TTL_TICK_DAYS,
        "indexes": list(col.index_information().keys()),
    }

    # ---- Summary -----------------------------------------------------------
    log.info("")
    log.info("=" * 60)
    log.info("SCHEMA INITIALIZATION COMPLETE -- SUMMARY")
    log.info("=" * 60)
    log.info("%-26s  %-6s  %s", "Collection", "TTL", "Indexes")
    log.info("-" * 60)
    for cname, meta in collections_created.items():
        idx_str = ", ".join(meta["indexes"])
        log.info("%-26s  %-6s  [%s]", cname, meta["ttl"], idx_str)
        log.info("  -> %s", meta["purpose"])
    log.info("=" * 60)
    log.info(
        "Initialized %d collections in database '%s' at %s",
        len(collections_created),
        db_name,
        datetime.now(tz=timezone.utc).isoformat(),
    )

    return db


# ===========================================================================
# Standalone entry point
# ===========================================================================

if __name__ == "__main__":
    _uri     = os.environ.get("MONGO_URI")
    _db_name = os.environ.get("MONGO_DB_NAME", "trading_db")

    if not _uri:
        log.error("Environment variable MONGO_URI is not set. Aborting.")
        sys.exit(1)

    try:
        init_mongo_db(_uri, _db_name)
        log.info("All done. Exiting.")
    except ConnectionFailure as _exc:
        log.error("Could not connect to MongoDB: %s", _exc)
        sys.exit(2)
    except Exception as _exc:  # noqa: BLE001
        log.exception("Unexpected error during schema initialization: %s", _exc)
        sys.exit(3)
