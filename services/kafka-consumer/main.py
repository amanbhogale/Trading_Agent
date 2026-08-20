"""
Kafka Consumer Microservice
============================
Consumes messages from Kafka topics (nse.ohlcv, nse.orderbook, nse.trades)
and writes them to MongoDB Atlas.

Environment Variables:
    KAFKA_BOOTSTRAP_SERVERS  - Kafka broker address (default: kafka:9092)
    MONGO_URI                - MongoDB connection URI (required)
    MONGO_DB_NAME            - MongoDB database name (default: trading_db)
    KAFKA_GROUP_ID           - Kafka consumer group ID (default: trading-mongo-consumer)
    KAFKA_AUTO_OFFSET_RESET  - Kafka offset reset policy (default: earliest)
"""

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError, DuplicateKeyError, PyMongoError
from confluent_kafka import Consumer, KafkaError, KafkaException

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] service=kafka-consumer %(name)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S%z',
)
logger = logging.getLogger("kafka-consumer")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
MONGO_URI: str = os.getenv("MONGO_URI", "")
MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "trading_db")
KAFKA_GROUP_ID: str = os.getenv("KAFKA_GROUP_ID", "trading-mongo-consumer")
KAFKA_AUTO_OFFSET_RESET: str = os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest")

TOPIC_OHLCV: str = "nse.ohlcv"
TOPIC_ORDERBOOK: str = "nse.orderbook"
TOPIC_TRADES: str = "nse.trades"
ALL_TOPICS: List[str] = [TOPIC_OHLCV, TOPIC_ORDERBOOK, TOPIC_TRADES]

BATCH_SIZE: int = 100          # max messages per topic before flushing to MongoDB
KAFKA_POLL_TIMEOUT: float = 1.0
KAFKA_STARTUP_RETRIES: int = 5
KAFKA_STARTUP_RETRY_DELAY: float = 5.0

# ---------------------------------------------------------------------------
# Shared state (thread-safe via locks)
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()

_state: Dict[str, Any] = {
    "running": False,
    "paused": False,
    "messages_processed": {"ohlcv": 0, "orderbook": 0, "trades": 0},
    "last_message_at": None,
    "kafka_connected": False,
    "mongo_connected": False,
    "errors": 0,
}

# Batch buffers per topic  {topic: [msg_dict, ...]}
_buffers: Dict[str, List[Dict[str, Any]]] = {
    TOPIC_OHLCV: [],
    TOPIC_ORDERBOOK: [],
    TOPIC_TRADES: [],
}

_consumer: Optional[Consumer] = None
_mongo_client: Optional[MongoClient] = None
_poll_thread: Optional[threading.Thread] = None

# ---------------------------------------------------------------------------
# MongoDB helpers
# ---------------------------------------------------------------------------

def _ohlcv_collection_name(timeframe: str) -> str:
    """Return the MongoDB collection name for a given OHLCV timeframe."""
    mapping = {
        "1m": "ohlcv_1m",
        "5m": "ohlcv_5m",
    }
    return mapping.get(timeframe, "ohlcv")


def _flush_ohlcv(db, messages: List[Dict[str, Any]]) -> None:
    """Bulk-upsert OHLCV messages into MongoDB."""
    if not messages:
        return

    # Group by target collection
    by_collection: Dict[str, List[Dict[str, Any]]] = {}
    for msg in messages:
        col_name = _ohlcv_collection_name(msg.get("timeframe", ""))
        by_collection.setdefault(col_name, []).append(msg)

    for col_name, docs in by_collection.items():
        col = db[col_name]
        operations = [
            UpdateOne(
                filter={
                    "symbol": doc["symbol"],
                    "exchange": doc["exchange"],
                    "timeframe": doc["timeframe"],
                    "ts": doc["ts"],
                },
                update={"$set": doc},
                upsert=True,
            )
            for doc in docs
        ]
        try:
            result = col.bulk_write(operations, ordered=False)
            logger.debug(
                "OHLCV bulk_write to '%s': upserted=%d modified=%d",
                col_name, result.upserted_count, result.modified_count,
            )
        except BulkWriteError as bwe:
            logger.error("OHLCV bulk_write error to '%s': %s", col_name, bwe.details)
            with _state_lock:
                _state["errors"] += 1
        except PyMongoError as exc:
            logger.error("OHLCV MongoDB error to '%s': %s", col_name, exc)
            with _state_lock:
                _state["errors"] += 1


def _flush_orderbook(db, messages: List[Dict[str, Any]]) -> None:
    """Insert orderbook snapshots; silently suppress duplicate-key errors."""
    if not messages:
        return
    col = db["orderbook_snapshots"]
    for doc in messages:
        try:
            col.insert_one(doc)
        except DuplicateKeyError:
            # Duplicate seq — already recorded, skip silently
            logger.debug(
                "Duplicate orderbook seq=%s for %s, skipping.",
                doc.get("seq"), doc.get("symbol"),
            )
        except PyMongoError as exc:
            logger.error("Orderbook insert error: %s", exc)
            with _state_lock:
                _state["errors"] += 1


def _flush_trades(db, messages: List[Dict[str, Any]]) -> None:
    """Bulk-upsert trade prints keyed on trade_id."""
    if not messages:
        return
    col = db["trade_prints"]
    operations = [
        UpdateOne(
            filter={"trade_id": doc["trade_id"]},
            update={"$set": doc},
            upsert=True,
        )
        for doc in messages
    ]
    try:
        result = col.bulk_write(operations, ordered=False)
        logger.debug(
            "Trades bulk_write: upserted=%d modified=%d",
            result.upserted_count, result.modified_count,
        )
    except BulkWriteError as bwe:
        logger.error("Trades bulk_write error: %s", bwe.details)
        with _state_lock:
            _state["errors"] += 1
    except PyMongoError as exc:
        logger.error("Trades MongoDB error: %s", exc)
        with _state_lock:
            _state["errors"] += 1


def _flush_all_buffers(db) -> None:
    """Flush all topic buffers to MongoDB."""
    global _buffers

    ohlcv_batch = _buffers[TOPIC_OHLCV][:]
    orderbook_batch = _buffers[TOPIC_ORDERBOOK][:]
    trades_batch = _buffers[TOPIC_TRADES][:]

    # Clear buffers before I/O so new messages can accumulate
    _buffers[TOPIC_OHLCV] = []
    _buffers[TOPIC_ORDERBOOK] = []
    _buffers[TOPIC_TRADES] = []

    if ohlcv_batch:
        _flush_ohlcv(db, ohlcv_batch)
        with _state_lock:
            _state["messages_processed"]["ohlcv"] += len(ohlcv_batch)

    if orderbook_batch:
        _flush_orderbook(db, orderbook_batch)
        with _state_lock:
            _state["messages_processed"]["orderbook"] += len(orderbook_batch)

    if trades_batch:
        _flush_trades(db, trades_batch)
        with _state_lock:
            _state["messages_processed"]["trades"] += len(trades_batch)


# ---------------------------------------------------------------------------
# Kafka poll loop (runs in background thread)
# ---------------------------------------------------------------------------

def _poll_loop(db) -> None:
    """Main Kafka poll loop. Runs in a background thread."""
    global _consumer, _buffers

    logger.info("Kafka poll loop started.")

    with _state_lock:
        _state["running"] = True

    try:
        while True:
            with _state_lock:
                paused = _state["paused"]
                running = _state["running"]

            if not running:
                logger.info("Poll loop: running=False, exiting.")
                break

            if paused:
                time.sleep(0.5)
                continue

            try:
                msg = _consumer.poll(timeout=KAFKA_POLL_TIMEOUT)
            except KafkaException as exc:
                logger.error("Kafka poll exception: %s", exc)
                with _state_lock:
                    _state["errors"] += 1
                time.sleep(1.0)
                continue

            if msg is None:
                # No message — check if any buffer needs flushing
                _flush_all_buffers(db)
                continue

            if msg.error():
                err = msg.error()
                if err.code() == KafkaError._PARTITION_EOF:
                    logger.debug(
                        "End of partition: topic=%s partition=%d offset=%d",
                        msg.topic(), msg.partition(), msg.offset(),
                    )
                else:
                    logger.error("Kafka message error: %s", err)
                    with _state_lock:
                        _state["errors"] += 1
                continue

            # Deserialize message
            topic = msg.topic()
            try:
                payload = json.loads(msg.value().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as exc:
                logger.error(
                    "Deserialization error on topic=%s offset=%d: %s",
                    topic, msg.offset(), exc,
                )
                with _state_lock:
                    _state["errors"] += 1
                continue

            # Buffer the message
            if topic in _buffers:
                _buffers[topic].append(payload)

            with _state_lock:
                _state["last_message_at"] = datetime.now(timezone.utc).isoformat()

            # Flush if any buffer is at capacity
            max_buf = max(len(_buffers[t]) for t in ALL_TOPICS)
            if max_buf >= BATCH_SIZE:
                _flush_all_buffers(db)

    except Exception as exc:
        logger.exception("Unexpected error in poll loop: %s", exc)
        with _state_lock:
            _state["errors"] += 1
    finally:
        # Final flush before exit
        try:
            _flush_all_buffers(db)
        except Exception:
            pass
        with _state_lock:
            _state["running"] = False
        logger.info("Kafka poll loop exited.")


# ---------------------------------------------------------------------------
# Startup / shutdown helpers
# ---------------------------------------------------------------------------

def _connect_mongo() -> MongoClient:
    """Create and verify MongoDB connection."""
    if not MONGO_URI:
        raise RuntimeError("MONGO_URI environment variable is not set.")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    # Verify connection
    client.admin.command("ping")
    logger.info("MongoDB connection established. DB=%s", MONGO_DB_NAME)
    return client


def _create_consumer() -> Consumer:
    """Create a Confluent Kafka Consumer with retry logic."""
    conf = {
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": KAFKA_GROUP_ID,
        "auto.offset.reset": KAFKA_AUTO_OFFSET_RESET,
        "enable.auto.commit": True,
        "auto.commit.interval.ms": 5000,
        "session.timeout.ms": 30000,
        "max.poll.interval.ms": 300000,
    }

    for attempt in range(1, KAFKA_STARTUP_RETRIES + 1):
        try:
            consumer = Consumer(conf)
            consumer.subscribe(ALL_TOPICS)
            logger.info(
                "Kafka consumer created and subscribed to topics: %s", ALL_TOPICS
            )
            return consumer
        except KafkaException as exc:
            logger.warning(
                "Kafka connection attempt %d/%d failed: %s",
                attempt, KAFKA_STARTUP_RETRIES, exc,
            )
            if attempt < KAFKA_STARTUP_RETRIES:
                time.sleep(KAFKA_STARTUP_RETRY_DELAY)
            else:
                raise RuntimeError(
                    f"Failed to connect to Kafka after {KAFKA_STARTUP_RETRIES} attempts."
                ) from exc

    # Should never reach here
    raise RuntimeError("Unexpected error creating Kafka consumer.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown logic."""
    global _consumer, _mongo_client, _poll_thread

    logger.info("Starting kafka-consumer service...")

    # --- MongoDB ---
    try:
        _mongo_client = _connect_mongo()
        with _state_lock:
            _state["mongo_connected"] = True
    except Exception as exc:
        logger.error("MongoDB startup failed: %s", exc)
        with _state_lock:
            _state["mongo_connected"] = False

    # --- Kafka ---
    try:
        _consumer = _create_consumer()
        with _state_lock:
            _state["kafka_connected"] = True
    except Exception as exc:
        logger.error("Kafka startup failed: %s", exc)
        with _state_lock:
            _state["kafka_connected"] = False

    # --- Background poll thread ---
    if _state["kafka_connected"] and _state["mongo_connected"]:
        db = _mongo_client[MONGO_DB_NAME]
        _poll_thread = threading.Thread(
            target=_poll_loop,
            args=(db,),
            name="kafka-poll-loop",
            daemon=True,
        )
        _poll_thread.start()
        logger.info("Background Kafka poll thread started.")
    else:
        logger.warning(
            "Skipping poll thread start due to connection failures. "
            "kafka_connected=%s, mongo_connected=%s",
            _state["kafka_connected"],
            _state["mongo_connected"],
        )

    yield  # Application is running

    # --- Shutdown ---
    logger.info("Shutting down kafka-consumer service...")

    with _state_lock:
        _state["running"] = False

    if _poll_thread and _poll_thread.is_alive():
        logger.info("Waiting for poll thread to exit...")
        _poll_thread.join(timeout=15)
        if _poll_thread.is_alive():
            logger.warning("Poll thread did not exit cleanly within timeout.")

    if _consumer is not None:
        try:
            _consumer.close()
            logger.info("Kafka consumer closed.")
        except Exception as exc:
            logger.error("Error closing Kafka consumer: %s", exc)

    if _mongo_client is not None:
        try:
            _mongo_client.close()
            logger.info("MongoDB connection closed.")
        except Exception as exc:
            logger.error("Error closing MongoDB connection: %s", exc)

    logger.info("kafka-consumer service stopped.")


app = FastAPI(
    title="Kafka Consumer Service",
    description="Consumes NSE market data from Kafka and persists to MongoDB Atlas.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health() -> JSONResponse:
    """Return service health status."""
    return JSONResponse({"status": "ok", "service": "kafka-consumer"})


@app.get("/status", summary="Consumer status")
async def status() -> JSONResponse:
    """Return detailed consumer status."""
    with _state_lock:
        snapshot = {
            "running": _state["running"],
            "paused": _state["paused"],
            "messages_processed": dict(_state["messages_processed"]),
            "last_message_at": _state["last_message_at"],
            "kafka_connected": _state["kafka_connected"],
            "mongo_connected": _state["mongo_connected"],
            "errors": _state["errors"],
        }
    return JSONResponse(snapshot)


@app.post("/pause", summary="Pause Kafka consumption")
async def pause_consumer() -> JSONResponse:
    """Pause the Kafka poll loop without stopping the thread."""
    with _state_lock:
        if not _state["running"]:
            return JSONResponse(
                {"status": "error", "detail": "Consumer is not running."},
                status_code=400,
            )
        _state["paused"] = True
        logger.info("Kafka consumer paused via /pause endpoint.")
    return JSONResponse({"status": "paused"})


@app.post("/resume", summary="Resume Kafka consumption")
async def resume_consumer() -> JSONResponse:
    """Resume a paused Kafka poll loop."""
    with _state_lock:
        if not _state["running"]:
            return JSONResponse(
                {"status": "error", "detail": "Consumer is not running."},
                status_code=400,
            )
        _state["paused"] = False
        logger.info("Kafka consumer resumed via /resume endpoint.")
    return JSONResponse({"status": "resumed"})


# ---------------------------------------------------------------------------
# Entry point (for local development)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
