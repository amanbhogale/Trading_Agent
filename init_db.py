import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def init_db():
    dbname = os.getenv("DB_NAME", "trading_db")
    user = os.getenv("DB_USER", "trading_agent")
    password = os.getenv("DB_PASSWORD", "zombie612@")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    conn.autocommit = True
    cur = conn.cursor()

    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS agent_memory (
        agent_id VARCHAR(255) PRIMARY KEY,
        created_at TIMESTAMP,
        updated_at TIMESTAMP,
        context JSONB DEFAULT '{}'::jsonb
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id SERIAL PRIMARY KEY,
        agent_id VARCHAR(255) REFERENCES agent_memory(agent_id),
        role VARCHAR(50),
        content TEXT,
        timestamp TIMESTAMP
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS decisions (
        id SERIAL PRIMARY KEY,
        agent_id VARCHAR(255) REFERENCES agent_memory(agent_id),
        data JSONB,
        timestamp TIMESTAMP
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS strategies (
        name VARCHAR(255) PRIMARY KEY,
        data JSONB,
        saved_at TIMESTAMP
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS trade_logs (
        id SERIAL PRIMARY KEY,
        date DATE,
        data JSONB,
        logged_at TIMESTAMP,
        market_mode VARCHAR(50) DEFAULT 'equity'
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS visualizations (
        name VARCHAR(255) PRIMARY KEY,
        meta JSONB,
        saved_at TIMESTAMP
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS hedging_positions (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(50) NOT NULL,
        option_type VARCHAR(10) NOT NULL,
        strike NUMERIC NOT NULL,
        expiry DATE,
        quantity INTEGER NOT NULL,
        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS hedge_trades (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(50) NOT NULL,
        trade_type VARCHAR(10) NOT NULL,
        quantity INTEGER NOT NULL,
        price NUMERIC NOT NULL,
        net_delta_before NUMERIC,
        executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # --- New Tickers Classification Table ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tickers_classification (
        symbol VARCHAR(50) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        sector VARCHAR(100),
        status VARCHAR(50),
        demerger_details TEXT,
        market_mode VARCHAR(50) DEFAULT 'equity'
    );
    """)

    # Seed tickers
    tickers = [
        ("NSE:TMCV", "Tata Motors Commercial Vehicles", "Automotive - Commercial", "Active", "Demerged entity representing the Commercial Vehicles business of Tata Motors Ltd.", "equity"),
        ("NSE:TMPV", "Tata Motors Passenger Vehicles", "Automotive - Passenger & EV", "Active", "Demerged entity representing Passenger Vehicles, Electric Vehicles, and JLR business of Tata Motors Ltd.", "equity"),
        ("NSE:RELIANCE", "Reliance Industries Ltd.", "Conglomerate / Energy", "Active", None, "equity"),
        ("NSE:TCS", "Tata Consultancy Services Ltd.", "IT Services", "Active", None, "equity"),
        ("NSE:INFY", "Infosys Ltd.", "IT Services", "Active", None, "equity"),
        ("NSE:HDFCBANK", "HDFC Bank Ltd.", "Banking & Financials", "Active", None, "equity"),
        ("NSE:ICICIBANK", "ICICI Bank Ltd.", "Banking & Financials", "Active", None, "equity"),
        ("NSE:WIPRO", "Wipro Ltd.", "IT Services", "Active", None, "equity"),
        ("NSE:BAJFINANCE", "Bajaj Finance Ltd.", "Banking & Financials", "Active", None, "equity"),
        ("NSE:SBIN", "State Bank of India", "Banking & Financials", "Active", None, "equity"),
        ("NSE:ADANIENT", "Adani Enterprises Ltd.", "Conglomerate", "Active", None, "equity"),
        ("NSE:ITC", "ITC Ltd.", "Consumer Goods / Tobacco", "Active", None, "equity"),
        
        # Forex
        ("EURUSD=X", "EUR/USD Exchange Rate", "Forex", "Active", None, "forex"),
        ("GBPUSD=X", "GBP/USD Exchange Rate", "Forex", "Active", None, "forex"),
        ("USDJPY=X", "USD/JPY Exchange Rate", "Forex", "Active", None, "forex"),
        ("AUDUSD=X", "AUD/USD Exchange Rate", "Forex", "Active", None, "forex"),
        ("USDCAD=X", "USD/CAD Exchange Rate", "Forex", "Active", None, "forex"),
        ("USDCHF=X", "USD/CHF Exchange Rate", "Forex", "Active", None, "forex"),
        ("NZDUSD=X", "NZD/USD Exchange Rate", "Forex", "Active", None, "forex"),
        ("EURGBP=X", "EUR/GBP Exchange Rate", "Forex", "Active", None, "forex"),
        ("EURJPY=X", "EUR/JPY Exchange Rate", "Forex", "Active", None, "forex"),
        ("GBPJPY=X", "GBP/JPY Exchange Rate", "Forex", "Active", None, "forex"),
        
        # Crypto
        ("BTCUSDT", "Bitcoin / USDT", "Crypto", "Active", None, "crypto"),
        ("ETHUSDT", "Ethereum / USDT", "Crypto", "Active", None, "crypto"),
        ("SOLUSDT", "Solana / USDT", "Crypto", "Active", None, "crypto"),
        ("XRPUSDT", "Ripple / USDT", "Crypto", "Active", None, "crypto"),
        ("DOGEUSDT", "Dogecoin / USDT", "Crypto", "Active", None, "crypto"),
        ("PAXGUSDT", "Pax Gold / USDT", "Crypto", "Active", None, "crypto"),
    ]

    for sym, name, sector, status, details, mm in tickers:
        cur.execute("""
        INSERT INTO tickers_classification (symbol, name, sector, status, demerger_details, market_mode)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol) DO UPDATE 
        SET name = EXCLUDED.name, sector = EXCLUDED.sector, status = EXCLUDED.status, 
            demerger_details = EXCLUDED.demerger_details, market_mode = EXCLUDED.market_mode;
        """, (sym, name, sector, status, details, mm))

    # ── DB Modernization Migrations ──────────────────────────────────────────
    print("Applying schema migrations (indexes, constraints, triggers)...")
    
    # 1. Add normalized columns to trade_logs if they don't exist
    cur.execute("""
    ALTER TABLE trade_logs 
        ADD COLUMN IF NOT EXISTS symbol VARCHAR(50),
        ADD COLUMN IF NOT EXISTS trade_type VARCHAR(10),
        ADD COLUMN IF NOT EXISTS quantity INTEGER,
        ADD COLUMN IF NOT EXISTS price NUMERIC,
        ADD COLUMN IF NOT EXISTS pnl NUMERIC;
    """)

    # 1.5 Create ohlcv_data table for streaming OHLCV indicators
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ohlcv_data (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(50) NOT NULL,
        timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        open NUMERIC NOT NULL,
        high NUMERIC NOT NULL,
        low NUMERIC NOT NULL,
        close NUMERIC NOT NULL,
        volume NUMERIC NOT NULL,
        interval VARCHAR(10) DEFAULT '1m'
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_ts ON ohlcv_data(symbol, timestamp DESC);")


    # 2. Add domain constraints
    cur.execute("""
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_option_type') THEN
            ALTER TABLE hedging_positions ADD CONSTRAINT chk_option_type CHECK (LOWER(option_type) IN ('call', 'put', 'future'));
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_trade_type') THEN
            ALTER TABLE hedge_trades ADD CONSTRAINT chk_trade_type CHECK (LOWER(trade_type) IN ('buy', 'sell'));
        END IF;
    END;
    $$;
    """)

    # 3. Create Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_agent_timestamp ON conversations (agent_id, timestamp ASC);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_agent_timestamp ON decisions (agent_id, timestamp ASC);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_logs_date_mode ON trade_logs (date, market_mode);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hedging_positions_symbol ON hedging_positions (symbol);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hedge_trades_symbol_executed ON hedge_trades (symbol, executed_at DESC);")

    # 4. Automate updated_at triggers
    cur.execute("""
    CREATE OR REPLACE FUNCTION update_modified_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = now();
        RETURN NEW;
    END;
    $$ language 'plpgsql';
    """)
    
    cur.execute("""
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_agent_memory_modtime') THEN
            CREATE TRIGGER update_agent_memory_modtime
                BEFORE UPDATE ON agent_memory
                FOR EACH ROW
                EXECUTE FUNCTION update_modified_column();
        END IF;
    END;
    $$;
    """)

    print("Tables created, seeded, and migrated successfully.")
    cur.close()
    conn.close()


if __name__ == '__main__':
    init_db()

