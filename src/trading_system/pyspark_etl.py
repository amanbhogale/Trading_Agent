import os
import sys
import time
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

# Ensure src/ is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, TimestampType

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PARAMS = {
    'dbname': 'trading_db',
    'user': 'trading_agent',
    'password': 'zombie612@',
    'host': 'localhost'
}

def get_active_tickers():
    """Fetch active tickers from PostgreSQL database."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("SELECT symbol, name, market_mode FROM tickers_classification WHERE status = 'Active';")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"symbol": r[0], "name": r[1], "market_mode": r[2]} for r in rows]

def fetch_ticker_data_yf(symbol, days=365):
    """Fetch historical data from Yahoo Finance as a Pandas DataFrame."""
    import yfinance as yf
    from src.trading_system.tools import _kite_to_yf
    
    yf_sym = _kite_to_yf(symbol)
    logger.info(f"Fetching YFinance data for {symbol} (mapped to {yf_sym})...")
    try:
        ticker = yf.Ticker(yf_sym)
        # Fetch 1 year of daily data
        df = ticker.history(period="1y", interval="1d", auto_adjust=True)
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return None
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        time_col = 'datetime' if 'datetime' in df.columns else 'date'
        
        # Format columns
        df = df.rename(columns={time_col: 'time'})
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        df['symbol'] = symbol
        return df[['symbol', 'time', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None

def compute_indicators_group(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Pandas Grouped Map function applied on Spark.
    Calculates technical indicators for a single symbol partition.
    """
    if pdf.empty:
        return pdf
    
    # Sort by time
    pdf = pdf.sort_values('time').copy()
    close = pdf['close']
    
    # EMAs
    pdf['ema9'] = close.ewm(span=9, adjust=False).mean()
    pdf['ema21'] = close.ewm(span=21, adjust=False).mean()
    pdf['ema50'] = close.ewm(span=50, adjust=False).mean()
    pdf['ema200'] = close.ewm(span=200, adjust=False).mean()
    
    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    pdf['rsi14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    pdf['macd'] = ema12 - ema26
    pdf['signal'] = pdf['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    pdf['bb_upper'] = sma20 + 2 * std20
    pdf['bb_mid'] = sma20
    pdf['bb_lower'] = sma20 - 2 * std20
    
    # Handle NaNs (Spark doesn't like NaN double type, prefers None/Null)
    cols_to_clean = ['ema9', 'ema21', 'ema50', 'ema200', 'rsi14', 'macd', 'signal', 'bb_upper', 'bb_mid', 'bb_lower']
    for col in cols_to_clean:
        pdf[col] = pdf[col].replace({np.nan: None})
        
    return pdf

def save_to_postgres(pdf: pd.DataFrame):
    """Save the precomputed data to PostgreSQL database in bulk."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    # Create target table if not exists
    cur.execute("""
    CREATE TABLE IF NOT EXISTS precomputed_ohlcv_indicators (
        symbol VARCHAR(50),
        time TIMESTAMP,
        open DOUBLE PRECISION,
        high DOUBLE PRECISION,
        low DOUBLE PRECISION,
        close DOUBLE PRECISION,
        volume BIGINT,
        ema9 DOUBLE PRECISION,
        ema21 DOUBLE PRECISION,
        ema50 DOUBLE PRECISION,
        ema200 DOUBLE PRECISION,
        rsi14 DOUBLE PRECISION,
        macd DOUBLE PRECISION,
        signal DOUBLE PRECISION,
        bb_upper DOUBLE PRECISION,
        bb_mid DOUBLE PRECISION,
        bb_lower DOUBLE PRECISION,
        precomputed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, time)
    );
    """)
    conn.commit()
    
    # Bulk upsert using execute_values
    columns = [
        'symbol', 'time', 'open', 'high', 'low', 'close', 'volume',
        'ema9', 'ema21', 'ema50', 'ema200', 'rsi14', 'macd', 'signal',
        'bb_upper', 'bb_mid', 'bb_lower'
    ]
    
    # Convert dataframe to tuples list
    # Replace NaNs with None for psycopg2
    data_tuples = []
    for _, r in pdf.iterrows():
        # Ensure timestamp is string format for Postgres insertion
        time_str = r['time'].strftime('%Y-%m-%d %H:%M:%S')
        data_tuples.append((
            r['symbol'], time_str, float(r['open']), float(r['high']), float(r['low']), float(r['close']), int(r['volume']),
            None if pd.isna(r['ema9']) else float(r['ema9']),
            None if pd.isna(r['ema21']) else float(r['ema21']),
            None if pd.isna(r['ema50']) else float(r['ema50']),
            None if pd.isna(r['ema200']) else float(r['ema200']),
            None if pd.isna(r['rsi14']) else float(r['rsi14']),
            None if pd.isna(r['macd']) else float(r['macd']),
            None if pd.isna(r['signal']) else float(r['signal']),
            None if pd.isna(r['bb_upper']) else float(r['bb_upper']),
            None if pd.isna(r['bb_mid']) else float(r['bb_mid']),
            None if pd.isna(r['bb_lower']) else float(r['bb_lower'])
        ))
        
    query = """
    INSERT INTO precomputed_ohlcv_indicators (
        symbol, time, open, high, low, close, volume,
        ema9, ema21, ema50, ema200, rsi14, macd, signal, bb_upper, bb_mid, bb_lower
    ) VALUES %s
    ON CONFLICT (symbol, time) DO UPDATE SET
        open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume,
        ema9 = EXCLUDED.ema9, ema21 = EXCLUDED.ema21, ema50 = EXCLUDED.ema50, ema200 = EXCLUDED.ema200,
        rsi14 = EXCLUDED.rsi14, macd = EXCLUDED.macd, signal = EXCLUDED.signal,
        bb_upper = EXCLUDED.bb_upper, bb_mid = EXCLUDED.bb_mid, bb_lower = EXCLUDED.bb_lower,
        precomputed_at = CURRENT_TIMESTAMP;
    """
    
    logger.info(f"Upserting {len(data_tuples)} records into precomputed_ohlcv_indicators table...")
    execute_values(cur, query, data_tuples)
    conn.commit()
    cur.close()
    conn.close()


def write_partition_to_postgres(iterator):
    """Write a Spark partition to PostgreSQL using bulk upsert."""
    import psycopg2
    from psycopg2.extras import execute_values
    import pandas as pd
    
    rows = list(iterator)
    if not rows:
        return
    
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    columns = [
        'symbol', 'time', 'open', 'high', 'low', 'close', 'volume',
        'ema9', 'ema21', 'ema50', 'ema200', 'rsi14', 'macd', 'signal',
        'bb_upper', 'bb_mid', 'bb_lower'
    ]
    
    data_tuples = []
    for row in rows:
        time_str = row['time'].strftime('%Y-%m-%d %H:%M:%S')
        data_tuples.append((
            row['symbol'], time_str, float(row['open']), float(row['high']), float(row['low']), float(row['close']), int(row['volume']),
            None if pd.isna(row['ema9']) else float(row['ema9']),
            None if pd.isna(row['ema21']) else float(row['ema21']),
            None if pd.isna(row['ema50']) else float(row['ema50']),
            None if pd.isna(row['ema200']) else float(row['ema200']),
            None if pd.isna(row['rsi14']) else float(row['rsi14']),
            None if pd.isna(row['macd']) else float(row['macd']),
            None if pd.isna(row['signal']) else float(row['signal']),
            None if pd.isna(row['bb_upper']) else float(row['bb_upper']),
            None if pd.isna(row['bb_mid']) else float(row['bb_mid']),
            None if pd.isna(row['bb_lower']) else float(row['bb_lower'])
        ))
    
    query = """
    INSERT INTO precomputed_ohlcv_indicators (
        symbol, time, open, high, low, close, volume,
        ema9, ema21, ema50, ema200, rsi14, macd, signal, bb_upper, bb_mid, bb_lower
    ) VALUES %s
    ON CONFLICT (symbol, time) DO UPDATE SET
        open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume,
        ema9 = EXCLUDED.ema9, ema21 = EXCLUDED.ema21, ema50 = EXCLUDED.ema50, ema200 = EXCLUDED.ema200,
        rsi14 = EXCLUDED.rsi14, macd = EXCLUDED.macd, signal = EXCLUDED.signal,
        bb_upper = EXCLUDED.bb_upper, bb_mid = EXCLUDED.bb_mid, bb_lower = EXCLUDED.bb_lower,
        precomputed_at = CURRENT_TIMESTAMP;
    """
    
    execute_values(cur, query, data_tuples)
    conn.commit()
    cur.close()
    conn.close()

def run_etl():
    start_time = time.time()
    logger.info("Initializing PySpark session...")
    
    spark = SparkSession.builder \
        .appName("TradingAgentPrecomputationETL") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.driver.memory", "6g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.executor.memory", "6g") \
        .master("local[*]") \
        .getOrCreate()
        
    try:
        # Step 1: Extract (Get active symbols from Postgres)
        tickers = get_active_tickers()
        logger.info(f"Loaded {len(tickers)} active symbols from PostgreSQL.")
        
        if not tickers:
            logger.warning("No active tickers to process.")
            return
            
        # Step 2: Fetch data (Extract phase)
        pandas_dfs = []
        import yfinance as yf
        from src.trading_system.tools import _kite_to_yf
        
        kite_to_yf_map = {}
        for ticker in tickers:
            sym = ticker['symbol']
            yf_sym = _kite_to_yf(sym)
            kite_to_yf_map[yf_sym] = sym
            
        yf_symbols = list(kite_to_yf_map.keys())
        
        # Batch download in chunks of 500
        chunk_size = 500
        for i in range(0, len(yf_symbols), chunk_size):
            chunk = yf_symbols[i:i+chunk_size]
            logger.info(f"Fetching YFinance data for chunk {i//chunk_size + 1}/{(len(yf_symbols)+chunk_size-1)//chunk_size} ({len(chunk)} symbols)...")
            try:
                data = yf.download(chunk, period="1y", interval="1d", group_by="ticker", threads=True, progress=False)
                
                if len(chunk) == 1:
                    if not data.empty:
                        df = data.dropna(how='all').reset_index()
                        df.columns = [c.lower() for c in df.columns]
                        time_col = 'datetime' if 'datetime' in df.columns else 'date'
                        df = df.rename(columns={time_col: 'time'})
                        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
                        df['symbol'] = kite_to_yf_map[chunk[0]]
                        req_cols = ['symbol', 'time', 'open', 'high', 'low', 'close', 'volume']
                        if all(c in df.columns for c in req_cols):
                            pandas_dfs.append(df[req_cols].dropna())
                else:
                    for yf_sym in chunk:
                        if yf_sym in data.columns.levels[0]:
                            df = data[yf_sym].dropna(how='all')
                            if df.empty:
                                continue
                            df = df.reset_index()
                            df.columns = [c.lower() for c in df.columns]
                            time_col = 'datetime' if 'datetime' in df.columns else 'date'
                            df = df.rename(columns={time_col: 'time'})
                            df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
                            df['symbol'] = kite_to_yf_map[yf_sym]
                            
                            req_cols = ['symbol', 'time', 'open', 'high', 'low', 'close', 'volume']
                            if all(c in df.columns for c in req_cols):
                                pandas_dfs.append(df[req_cols].dropna())
            except Exception as e:
                logger.error(f"Error fetching chunk: {e}")
                
        if not pandas_dfs:
            logger.error("Could not fetch data for any active symbols.")
            return
            
        # Combine all raw pandas dataframes
        combined_pdf = pd.concat(pandas_dfs, ignore_index=True)
        
        # Step 3: Load into Spark DataFrame
        spark_schema = StructType([
            StructField("symbol", StringType(), False),
            StructField("time", TimestampType(), False),
            StructField("open", DoubleType(), True),
            StructField("high", DoubleType(), True),
            StructField("low", DoubleType(), True),
            StructField("close", DoubleType(), True),
            StructField("volume", LongType(), True)
        ])
        
        spark_raw_df = spark.createDataFrame(combined_pdf, schema=spark_schema)
        
        # Step 4: Transform (Calculate indicators using applyInPandas Grouped Map)
        output_schema = StructType([
            StructField("symbol", StringType(), False),
            StructField("time", TimestampType(), False),
            StructField("open", DoubleType(), True),
            StructField("high", DoubleType(), True),
            StructField("low", DoubleType(), True),
            StructField("close", DoubleType(), True),
            StructField("volume", LongType(), True),
            StructField("ema9", DoubleType(), True),
            StructField("ema21", DoubleType(), True),
            StructField("ema50", DoubleType(), True),
            StructField("ema200", DoubleType(), True),
            StructField("rsi14", DoubleType(), True),
            StructField("macd", DoubleType(), True),
            StructField("signal", DoubleType(), True),
            StructField("bb_upper", DoubleType(), True),
            StructField("bb_mid", DoubleType(), True),
            StructField("bb_lower", DoubleType(), True)
        ])
        
        logger.info("Running parallel technical indicators precomputation using PySpark UDF & Apache Arrow...")
        precomputed_spark_df = spark_raw_df.groupBy("symbol").applyInPandas(compute_indicators_group, schema=output_schema)
        
        # Trigger Spark execution and collect back to Pandas for database write and local Parquet cache
        precomputed_pdf = precomputed_spark_df.toPandas()
        
        logger.info("Saving precomputed data locally in Arrow/Parquet files...")
        precomputed_dir = os.path.abspath("data/precomputed")
        os.makedirs(precomputed_dir, exist_ok=True)
        
        # Save partitioned by symbol locally for instant loading
        for symbol, group in precomputed_pdf.groupby('symbol'):
            safe_symbol = symbol.replace(':', '_')
            target_path = os.path.join(precomputed_dir, f"{safe_symbol}.parquet")
            # Write to parquet
            group.to_parquet(target_path, index=False)
            logger.info(f"Cached precomputed data to: {target_path}")
            
        # Step 5: Save to Postgres
        logger.info("Upserting precomputed data to PostgreSQL database...")
        save_to_postgres(precomputed_pdf)
        
        duration = time.time() - start_time
        logger.info(f"PySpark Precomputation ETL completed successfully in {duration:.2f} seconds!")
        
    except Exception as e:
        logger.exception("Error during PySpark ETL run:")
        raise e
    finally:
        spark.stop()

if __name__ == "__main__":
    run_etl()
