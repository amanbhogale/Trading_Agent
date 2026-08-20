import os
import json
import logging
from datetime import datetime, timedelta
from kafka import KafkaConsumer
from dotenv import load_dotenv
from trading_system.memory import db_pool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(name)s) %(message)s'
)
logger = logging.getLogger("StreamConsumer")

load_dotenv()

RAW_TICK_TOPIC = 'market.raw.ticks'

# Dictionary to hold the state of current active candles (1-minute aggregations)
# Structure: { symbol: { 'start_time': datetime, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float } }
active_candles = {}

def save_candle_to_db(symbol, candle):
    """Insert aggregated 1-minute candle into PostgreSQL ohlcv_data table."""
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ohlcv_data (symbol, timestamp, open, high, low, close, volume, interval)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                symbol,
                candle['start_time'],
                candle['open'],
                candle['high'],
                candle['low'],
                candle['close'],
                candle['volume'],
                '1m'
            ))
            conn.commit()
            logger.info(f"💾 Saved Candle: {symbol} | Time: {candle['start_time'].strftime('%H:%M:%S')} | O:{candle['open']} H:{candle['high']} L:{candle['low']} C:{candle['close']} V:{candle['volume']}")
    except Exception as e:
        logger.error(f"Failed to save candle for {symbol}: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            db_pool.putconn(conn)

def process_tick(symbol, price, volume, tick_time):
    """Aggregate incoming tick data into 1-minute OHLCV windows."""
    # Round timestamp to the start of the minute
    candle_minute = tick_time.replace(second=0, microsecond=0)
    
    if symbol not in active_candles:
        # Start a new candle window
        active_candles[symbol] = {
            'start_time': candle_minute,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': volume
        }
        return
        
    current_candle = active_candles[symbol]
    
    # Check if the tick falls into the current window
    if candle_minute == current_candle['start_time']:
        # Update existing candle
        current_candle['high'] = max(current_candle['high'], price)
        current_candle['low'] = min(current_candle['low'], price)
        current_candle['close'] = price
        current_candle['volume'] += volume
    else:
        # The minute has changed! Save the completed candle and start a new one
        save_candle_to_db(symbol, current_candle)
        
        # Initialize new candle window
        active_candles[symbol] = {
            'start_time': candle_minute,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': volume
        }

def start_consumer():
    """Start Kafka consumer loop to read raw ticks."""
    logger.info("Initializing Kafka Consumer...")
    
    try:
        consumer = KafkaConsumer(
            RAW_TICK_TOPIC,
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            group_id='ohlcv-aggregator-group'
        )
        logger.info(f"Subscribed to topic: {RAW_TICK_TOPIC}. Awaiting messages...")
    except Exception as e:
        logger.error(f"Failed to start Kafka Consumer: {e}")
        return

    try:
        for message in consumer:
            tick_data = message.value
            symbol = tick_data.get('symbol')
            price = tick_data.get('price')
            # Volume can be tick volume or accumulated volume
            volume = tick_data.get('volume', 1.0)
            ts = tick_data.get('timestamp')
            
            # Map epoch ms/s to datetime
            if ts:
                # Yahoo Finance timestamps can be in seconds
                if ts < 1e11: 
                    tick_time = datetime.fromtimestamp(ts)
                else:
                    tick_time = datetime.fromtimestamp(ts / 1000.0)
            else:
                tick_time = datetime.now()
                
            process_tick(symbol, price, volume, tick_time)
            
    except KeyboardInterrupt:
        logger.info("Stopping Kafka Stream Consumer...")
    except Exception as e:
        logger.error(f"Consumer loop crashed: {e}")
    finally:
        consumer.close()

if __name__ == '__main__':
    start_consumer()
