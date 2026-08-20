import os
import json
import time
import logging
from datetime import datetime
from kiteconnect import KiteTicker
from confluent_kafka import Producer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kite_producer")

KITE_API_KEY = os.getenv("KITE_API_KEY")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")
# Connect to kafka from the host on localhost:9092, or from inside docker via kafka:9092
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

if not KITE_API_KEY or not KITE_ACCESS_TOKEN:
    logger.error("Missing Kite API credentials in .env")
    exit(1)

# Initialize Kafka Producer
producer = Producer({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS})

def delivery_report(err, msg):
    if err is not None:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")

kws = KiteTicker(KITE_API_KEY, KITE_ACCESS_TOKEN)

# Map instrument tokens to symbols
TOKEN_MAP = {
    738561: ("RELIANCE", "NSE"),
    408065: ("INFY", "NSE"),
    341249: ("HDFCBANK", "NSE"),
    969473: ("WIPRO", "NSE")
}

def on_ticks(ws, ticks):
    for tick in ticks:
        token = tick.get("instrument_token")
        symbol_info = TOKEN_MAP.get(token)
        if not symbol_info:
            continue
            
        symbol, exchange = symbol_info
        
        # We need full mode for depth (L2 Orderbook)
        if "depth" in tick:
            depth = tick["depth"]
            
            bids = [{"price": b["price"], "qty": b["quantity"], "orders": b["orders"]} for b in depth["buy"]]
            asks = [{"price": a["price"], "qty": a["quantity"], "orders": a["orders"]} for a in depth["sell"]]
            
            msg = {
                "symbol": symbol,
                "exchange": exchange,
                "ts": datetime.utcnow().isoformat(),
                "seq": int(time.time() * 1000),
                "bids": bids,
                "asks": asks,
                "ltp": tick.get("last_price", 0),
                "ltq": tick.get("last_traded_quantity", 0),
                "total_bid_qty": tick.get("total_buy_quantity", 0),
                "total_ask_qty": tick.get("total_sell_quantity", 0),
                "source": "kite_ws"
            }
            
            producer.produce(
                "nse.orderbook",
                key=f"{symbol}:{exchange}",
                value=json.dumps(msg),
                callback=delivery_report
            )
            producer.poll(0)
            logger.info(f"Published live orderbook for {symbol} to Kafka")

def on_connect(ws, response):
    logger.info("Successfully connected to Kite WebSocket")
    tokens = list(TOKEN_MAP.keys())
    ws.subscribe(tokens)
    ws.set_mode(ws.MODE_FULL, tokens)

def on_close(ws, code, reason):
    logger.info(f"WebSocket closed: {code} - {reason}")
    
def on_error(ws, code, reason):
    logger.error(f"WebSocket error: {code} - {reason}")

kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close
kws.on_error = on_error

logger.info(f"Starting Kite Producer, connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS}...")
kws.connect()
