import psycopg2

def init_db():
    conn = psycopg2.connect(dbname='trading_db', user='trading_agent', password='zombie612@', host='localhost')
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

    print("Tables created and seeded successfully.")
    cur.close()
    conn.close()

if __name__ == '__main__':
    init_db()

