import os
import json
import logging
import urllib.parse
from kafka import KafkaProducer
from yliveticker import YLiveTicker
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(name)s) %(message)s'
)
logger = logging.getLogger("YahooStreamer")

load_dotenv()

# Topics definitions
RAW_TICK_TOPIC = 'market.raw.ticks'

# Initialize Kafka Producer
producer = None
try:
    # Set timeout parameters for connection
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks=1,
        retries=5,
        request_timeout_ms=10000
    )
    logger.info("Kafka Producer initialized successfully ✅")
except Exception as e:
    logger.error("Failed to connect to Kafka Broker: %s. The streamer will print ticks to console.", e)

def on_ticker_update(message):
    """Callback fired on every price tick from Yahoo Finance."""
    try:
        symbol = message.get('id')
        if not symbol:
            return
            
        payload = {
            'symbol': symbol,
            'price': float(message.get('price', 0.0)),
            'timestamp': int(message.get('timestamp', 0)),
            'volume': int(message.get('dayVolume', 0)),
            'change_pct': float(message.get('changePercent', 0.0))
        }
        
        logger.info(f"Tick: {symbol} | Price: {payload['price']} | Vol: {payload['volume']}")
        
        if producer:
            producer.send(RAW_TICK_TOPIC, key=symbol.encode('utf-8'), value=payload)
    except Exception as e:
        logger.error(f"Error processing tick callback: {e}")

if __name__ == '__main__':
    # Default list of symbols
    tickers = ["AAPL", "MSFT", "BTC-USD", "EURUSD=X", "^NSEI"]
    
    logger.info(f"Starting Yahoo Live Streamer subscribing to: {tickers}")
    try:
        # yliveticker connects to wss://streamer.finance.yahoo.com/ and decodes Protobuf
        ticker_stream = YLiveTicker(on_ticker=on_ticker_update, ticker_names=tickers)
        ticker_stream.start()
    except KeyboardInterrupt:
        logger.info("Stopping Yahoo Live Streamer...")
    except Exception as e:
        logger.error(f"Streamer encountered an error: {e}")
