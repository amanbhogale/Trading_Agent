import asyncio
import json
import logging
import threading
from kafka import KafkaConsumer
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(name)s) %(message)s'
)
logger = logging.getLogger("WebSocketServer")

RAW_TICK_TOPIC = 'market.raw.ticks'

# Set of connected WebSocket clients
connected_clients = set()
clients_lock = threading.Lock()

# Main async event loop reference
loop = None

async def register(websocket):
    with clients_lock:
        connected_clients.add(websocket)
    logger.info(f"Client connected: {websocket.remote_address}. Active clients: {len(connected_clients)}")
    try:
        await websocket.wait_closed()
    finally:
        with clients_lock:
            connected_clients.remove(websocket)
        logger.info(f"Client disconnected: {websocket.remote_address}. Active clients: {len(connected_clients)}")

async def broadcast_tick(payload):
    if not connected_clients:
        return
    message_str = json.dumps(payload)
    # Gather all broadcast coroutines
    with clients_lock:
        clients = list(connected_clients)
    if clients:
        await asyncio.gather(*[client.send(message_str) for client in clients], return_exceptions=True)

def kafka_consumer_thread():
    """Runs synchronous Kafka consumer and forwards to active WS clients."""
    logger.info("Kafka consumer thread starting...")
    try:
        consumer = KafkaConsumer(
            RAW_TICK_TOPIC,
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest'
        )
        logger.info(f"Kafka consumer subscribed to {RAW_TICK_TOPIC}")
    except Exception as e:
        logger.error(f"Kafka consumer connection failed: {e}")
        return

    try:
        for message in consumer:
            tick_data = message.value
            if loop:
                # Schedule broadcast in the main asyncio event loop thread-safely
                asyncio.run_coroutine_threadsafe(broadcast_tick(tick_data), loop)
    except Exception as e:
        logger.error(f"Kafka consumer loop crashed: {e}")
    finally:
        consumer.close()

async def main():
    global loop
    loop = asyncio.get_running_loop()
    
    # Start Kafka consumer thread
    t = threading.Thread(target=kafka_consumer_thread, daemon=True)
    t.start()
    
    logger.info("Starting WebSocket Server on ws://0.0.0.0:8765 ...")
    async with websockets.serve(register, "0.0.0.0", 8765):
        await asyncio.Future() # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("WebSocket Server shut down.")
