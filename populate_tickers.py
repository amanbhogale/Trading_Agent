import pandas as pd
import psycopg2
import sys

DB_CONFIG = {
    'dbname': 'trading_db',
    'user': 'trading_agent',
    'password': 'zombie612@',
    'host': 'localhost'
}

def main():
    print("Fetching NSE & BSE instruments from Kite API...")
    try:
        df = pd.read_csv('https://api.kite.trade/instruments')
        # Filter only Equities from NSE and BSE
        eq_df = df[(df['exchange'].isin(['NSE', 'BSE'])) & (df['segment'].isin(['BSE', 'NSE']))].copy()
        print(f"Found {len(eq_df)} NSE/BSE equity instruments.")
    except Exception as e:
        print("Failed to fetch Kite instruments:", e)
        return

    print("Fetching S&P 500 instruments for US Stocks...")
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        print(f"Found {len(sp500)} US instruments.")
    except Exception as e:
        print("Failed to fetch US instruments:", e)
        sp500 = pd.DataFrame()

    print("Connecting to DB to insert tickers...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    count = 0
    # Insert NSE/BSE
    for _, row in eq_df.iterrows():
        symbol = f"{row['exchange']}:{row['tradingsymbol']}"
        name = row['name'] if pd.notnull(row['name']) else row['tradingsymbol']
        sector = 'Equity'
        
        try:
            cur.execute("""
                INSERT INTO tickers_classification (symbol, name, sector, status, demerger_details)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (symbol) DO NOTHING;
            """, (symbol, name, sector, 'Active', None))
            count += 1
        except Exception as e:
            pass

    # Insert US S&P 500
    if not sp500.empty:
        for _, row in sp500.iterrows():
            symbol = f"US:{row['Symbol']}"
            name = row['Security']
            sector = row['GICS Sector']
            try:
                cur.execute("""
                    INSERT INTO tickers_classification (symbol, name, sector, status, demerger_details)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (symbol) DO NOTHING;
                """, (symbol, name, sector, 'Active', None))
                count += 1
            except Exception as e:
                pass

    print(f"Successfully inserted/verified {count} tickers into the database!")
    cur.close()
    conn.close()

if __name__ == '__main__':
    main()
