import psycopg2

def init_db():
    conn = psycopg2.connect(dbname='trading_db', user='trading_agent', password='secretpassword', host='localhost')
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
        logged_at TIMESTAMP
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS visualizations (
        name VARCHAR(255) PRIMARY KEY,
        meta JSONB,
        saved_at TIMESTAMP
    );
    """)

    # --- New Tickers Classification Table ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tickers_classification (
        symbol VARCHAR(50) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        sector VARCHAR(100),
        status VARCHAR(50),
        demerger_details TEXT
    );
    """)

    # Seed tickers
    tickers = [
        ("NSE:TMCV", "Tata Motors Commercial Vehicles", "Automotive - Commercial", "Active", "Demerged entity representing the Commercial Vehicles business of Tata Motors Ltd."),
        ("NSE:TMPV", "Tata Motors Passenger Vehicles", "Automotive - Passenger & EV", "Active", "Demerged entity representing Passenger Vehicles, Electric Vehicles, and JLR business of Tata Motors Ltd."),
        ("NSE:RELIANCE", "Reliance Industries Ltd.", "Conglomerate / Energy", "Active", None),
        ("NSE:TCS", "Tata Consultancy Services Ltd.", "IT Services", "Active", None),
        ("NSE:INFY", "Infosys Ltd.", "IT Services", "Active", None),
        ("NSE:HDFCBANK", "HDFC Bank Ltd.", "Banking & Financials", "Active", None),
        ("NSE:ICICIBANK", "ICICI Bank Ltd.", "Banking & Financials", "Active", None),
        ("NSE:WIPRO", "Wipro Ltd.", "IT Services", "Active", None),
        ("NSE:BAJFINANCE", "Bajaj Finance Ltd.", "Banking & Financials", "Active", None),
        ("NSE:SBIN", "State Bank of India", "Banking & Financials", "Active", None),
        ("NSE:ADANIENT", "Adani Enterprises Ltd.", "Conglomerate", "Active", None),
        ("NSE:ITC", "ITC Ltd.", "Consumer Goods / Tobacco", "Active", None),
    ]

    for sym, name, sector, status, details in tickers:
        cur.execute("""
        INSERT INTO tickers_classification (symbol, name, sector, status, demerger_details)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (symbol) DO UPDATE 
        SET name = EXCLUDED.name, sector = EXCLUDED.sector, status = EXCLUDED.status, demerger_details = EXCLUDED.demerger_details;
        """, (sym, name, sector, status, details))

    print("Tables created and seeded successfully.")
    cur.close()
    conn.close()

if __name__ == '__main__':
    init_db()

