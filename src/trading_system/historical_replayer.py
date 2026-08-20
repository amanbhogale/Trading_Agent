import os
import time
import json
import logging
import yfinance as yf
from datetime import datetime, timedelta
from kafka import KafkaProducer
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(name)s) %(message)s'
)
logger = logging.getLogger("HistoricalReplayer")

load_dotenv()

RAW_TICK_TOPIC = 'market.raw.ticks'

def start_replay(symbol, days=5, delay_sec=0.5):
    """
    Downloads 1-minute historical data for a symbol and streams it to Kafka
    to simulate a live real-time price feed.
    """
    logger.info(f"Initializing Kafka Producer...")
    try:
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        logger.info("Kafka Producer initialized successfully ✅")
    except Exception as e:
        logger.error(f"Failed to connect to Kafka Broker: {e}")
        return

    # Convert Kite symbols if needed
    from trading_system.tools import _kite_to_yf
    yf_symbol = _kite_to_yf(symbol)
    
    logger.info(f"Downloading historical 1-minute candles for {yf_symbol}...")
    try:
        ticker = yf.Ticker(yf_symbol)
        # 1-minute interval only supports up to last 30 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days))
        
        df = ticker.history(start=start_date, end=end_date, interval="1m")
        if df.empty:
            logger.warning(f"No historical 1-minute data found for {yf_symbol}.")
            return
            
        logger.info(f"Downloaded {len(df)} historical rows. Starting streaming replay...")
    except Exception as e:
        logger.error(f"Failed to fetch historical data: {e}")
        return

    try:
        for timestamp, row in df.iterrows():
            payload = {
                'symbol': symbol, # Maintain original symbol notation
                'price': float(row['Close']),
                'timestamp': int(timestamp.timestamp()),
                'volume': float(row['Volume']),
                'change_pct': 0.0 # Heuristic placeholder
            }
            
            logger.info(f"Replaying Tick: {symbol} | Time: {timestamp} | Price: {payload['price']} | Vol: {payload['volume']}")
            producer.send(RAW_TICK_TOPIC, key=symbol.encode('utf-8'), value=payload)
            
            # Throttle the streaming rate
            time.sleep(delay_sec)
            
        logger.info("Replay completed successfully! ✅")
    except KeyboardInterrupt:
        logger.info("Replay interrupted by user.")
    except Exception as e:
        logger.error(f"Replayer encountered an error: {e}")
    finally:
        producer.close()

if __name__ == '__main__':
    # Replay NSE:INFY 1-minute candles with a 0.2 second delay per candle
    start_replay("NSE:INFY", days=1, delay_sec=0.2)
