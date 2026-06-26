import urllib.request
import pandas as pd
import psycopg2

DB_CONFIG = {
    'dbname': 'trading_db',
    'user': 'trading_agent',
    'password': 'zombie612@',
    'host': 'localhost'
}

def main():
    print("Fetching S&P 500 instruments for US Stocks...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html = urllib.request.urlopen(req).read()
        sp500 = pd.read_html(html)[0]
        print(f"Found {len(sp500)} US instruments.")
    except Exception as e:
        print("Failed to fetch US instruments:", e)
        return

    print("Connecting to DB to insert US tickers...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    count = 0
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

    print(f"Successfully inserted {count} US tickers into the database!")
    cur.close()
    conn.close()

if __name__ == '__main__':
    main()
