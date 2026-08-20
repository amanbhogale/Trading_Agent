import os
import json
import time
import logging
import websocket
from datetime import datetime
from confluent_kafka import Producer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("binance_producer")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

# Initialize Kafka Producer
producer = Producer({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS})

def delivery_report(err, msg):
    if err is not None:
        logger.error(f"Message delivery failed: {err}")

def on_message(ws, message):
    data = json.loads(message)
    
    # Binance depth payload
    bids_raw = data.get("bids", [])
    asks_raw = data.get("asks", [])
    
    if not bids_raw or not asks_raw:
        return
        
    bids = [{"price": float(b[0]), "qty": float(b[1]), "orders": 1} for b in bids_raw]
    asks = [{"price": float(a[0]), "qty": float(a[1]), "orders": 1} for a in asks_raw]
    
    # Calculate LTP as Mid Price for demo purposes
    best_bid = bids[0]["price"]
    best_ask = asks[0]["price"]
    ltp = (best_bid + best_ask) / 2
    
    msg = {
        "symbol": "BTCUSDT",
        "exchange": "crypto",
        "ts": datetime.utcnow().isoformat(),
        "seq": int(time.time() * 1000),
        "bids": bids,
        "asks": asks,
        "ltp": ltp,
        "ltq": 0,
        "total_bid_qty": sum(b["qty"] for b in bids),
        "total_ask_qty": sum(a["qty"] for a in asks),
        "source": "binance_ws"
    }
    
    producer.produce(
        "nse.orderbook",
        key="BTCUSDT:crypto",
        value=json.dumps(msg),
        callback=delivery_report
    )
    producer.poll(0)
    logger.info(f"Published LIVE Binance orderbook for BTCUSDT to Kafka")

def on_error(ws, error):
    logger.error(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.info("WebSocket Closed")

def on_open(ws):
    logger.info("Connected to Binance WebSocket - Streaming LIVE BTCUSDT depth")

if __name__ == "__main__":
    # Binance provides 20-level deep orderbook at 100ms interval for free without authentication
    socket = "wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms"
    ws = websocket.WebSocketApp(socket,
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)
    ws.run_forever()
