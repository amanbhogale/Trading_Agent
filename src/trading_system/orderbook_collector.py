import os
import sys
import time
import json
import threading
import datetime
import logging
import websocket
import psycopg2
from psycopg2.extras import Json

# Setup paths
sys.path.append("/home/zombie/Documents/Trading_Agent")
sys.path.append("/home/zombie/Documents/Trading_Agent/src")

from trading_system.memory import DB_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("/home/zombie/Documents/Trading_Agent/orderbook_collector.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("OrderBookCollector")

# Memory structures to hold current order book states
# Format: { symbol: { 'bids': { price: qty }, 'asks': { price: qty } } }
order_books = {
    "BTCUSDT": {"bids": {}, "asks": {}},
    "ETHUSDT": {"bids": {}, "asks": {}}
}
book_lock = threading.Lock()

# Queue to hold snapshots for batch insertion
save_queue = []
queue_lock = threading.Lock()

def update_book(symbol, bids_updates, asks_updates, action):
    with book_lock:
        if symbol not in order_books:
            return
        
        # Clear book on snapshot
        if action == "snapshot":
            order_books[symbol]["bids"] = {}
            order_books[symbol]["asks"] = {}
            
        # Update bids
        for price_str, qty_str in bids_updates:
            try:
                price = float(price_str)
                qty = float(qty_str)
                if qty == 0.0:
                    order_books[symbol]["bids"].pop(price, None)
                else:
                    order_books[symbol]["bids"][price] = qty
            except Exception as e:
                logger.error(f"Error parsing bid update: {e}")
                
        # Update asks
        for price_str, qty_str in asks_updates:
            try:
                price = float(price_str)
                qty = float(qty_str)
                if qty == 0.0:
                    order_books[symbol]["asks"].pop(price, None)
                else:
                    order_books[symbol]["asks"][price] = qty
            except Exception as e:
                logger.error(f"Error parsing ask update: {e}")

def get_top_10(symbol):
    with book_lock:
        if symbol not in order_books:
            return [], []
        
        # Sort bids descending (highest buy prices first)
        sorted_bids = sorted(order_books[symbol]["bids"].items(), key=lambda x: x[0], reverse=True)[:10]
        # Sort asks ascending (lowest sell prices first)
        sorted_asks = sorted(order_books[symbol]["asks"].items(), key=lambda x: x[0])[:10]
        
        # Map back to list of [price, quantity]
        bids_list = [[float(p), float(q)] for p, q in sorted_bids]
        asks_list = [[float(p), float(q)] for p, q in sorted_asks]
        
        return bids_list, asks_list

# WebSocket message handlers
def on_message(ws, message):
    try:
        data = json.loads(message)
        
        # We handle the 'l2_updates' channel messages
        if data.get('type') == 'l2_updates':
            symbol = data.get('symbol')
            action = data.get('action') # 'snapshot' or 'update'
            
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            
            update_book(symbol, bids, asks, action)
    except Exception as e:
        logger.error(f"WebSocket on_message error: {e}")

def on_error(ws, error):
    logger.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.info(f"WebSocket closed (status={close_status_code}, msg={close_msg}). Reconnecting in 5s...")

def on_open(ws):
    logger.info("WebSocket connection established. Subscribing to l2_updates...")
    subscribe_msg = {
        "type": "subscribe",
        "payload": {
            "channels": [
                {
                    "name": "l2_updates",
                    "symbols": ["BTCUSDT", "ETHUSDT"]
                }
            ]
        }
    }
    ws.send(json.dumps(subscribe_msg))

def start_websocket():
    while True:
        try:
            logger.info("Connecting to wss://socket.delta.exchange...")
            ws = websocket.WebSocketApp(
                "wss://socket.delta.exchange",
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever()
        except Exception as e:
            logger.error(f"WebSocket loop crash: {e}")
        time.sleep(5)

# Thread to periodically capture book state and queue it
def snapshot_generator_loop():
    logger.info("Starting snapshot generator loop (once every 1 minute)...")
    while True:
        # Align to the next minute boundary for consistent time tracking
        now = time.time()
        sleep_time = 60 - (now % 60)
        time.sleep(sleep_time)
        
        timestamp = datetime.datetime.utcnow()
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            bids, asks = get_top_10(symbol)
            if bids or asks:
                with queue_lock:
                    save_queue.append({
                        "symbol": symbol,
                        "exchange": "delta",
                        "timestamp": timestamp,
                        "bids": bids,
                        "asks": asks
                    })

# Thread to periodically write batch to DB and prune old records
def db_writer_loop():
    logger.info("Starting database writer and retention loop...")
    last_prune_time = 0
    
    while True:
        time.sleep(10) # check queue every 10 seconds
        
        # 1. Write queued records to PostgreSQL
        records_to_write = []
        with queue_lock:
            if save_queue:
                records_to_write = list(save_queue)
                save_queue.clear()
                
        if records_to_write:
            conn = None
            try:
                conn = psycopg2.connect(**DB_CONFIG)
                conn.autocommit = True
                with conn.cursor() as cur:
                    for rec in records_to_write:
                        cur.execute(
                            """
                            INSERT INTO order_books (symbol, exchange, timestamp, bids, asks)
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (rec["symbol"], rec["exchange"], rec["timestamp"], Json(rec["bids"]), Json(rec["asks"]))
                        )
                logger.info(f"Successfully saved {len(records_to_write)} order book snapshots to PostgreSQL.")
            except Exception as e:
                logger.error(f"Failed to batch write order books to DB: {e}")
                # Put them back in queue so we don't lose data
                with queue_lock:
                    save_queue.extend(records_to_write)
            finally:
                if conn:
                    conn.close()
                    
        # 2. Prune data older than 7 days (once every hour)
        now = time.time()
        if now - last_prune_time >= 3600:
            conn = None
            try:
                conn = psycopg2.connect(**DB_CONFIG)
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM order_books WHERE timestamp < NOW() - INTERVAL '7 days';")
                logger.info("Retention cleanup completed successfully. Pruned order book logs older than 7 days.")
                last_prune_time = now
            except Exception as e:
                logger.error(f"Failed to run database prune: {e}")
            finally:
                if conn:
                    conn.close()

if __name__ == "__main__":
    logger.info("Starting Multi-Exchange Order Book Capture service...")
    
    # Start snapshot generator thread
    t_snapshot = threading.Thread(target=snapshot_generator_loop, daemon=True)
    t_snapshot.start()
    
    # Start database writing thread
    t_db = threading.Thread(target=db_writer_loop, daemon=True)
    t_db.start()
    
    # Start WebSocket reader in main thread
    start_websocket()
