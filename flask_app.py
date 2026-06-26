import os
# Disable multi-threaded OpenMP/MKL pooling to prevent PyTorch segmentation faults under threads/forks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import json
import logging
from pathlib import Path
from typing import Optional

# Bootstrapper to resolve src/ directory automatically
sys.path.insert(0, str(Path(__file__).parent / "src"))

from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv
import markdown

from trading_system.main_agent import build_orchestrator, OrchestratorAgent
from trading_system.memory import MemoryManager
import trading_system.tools as T
from trading_system.news_classifier import NewsClassifier

news_classifier = NewsClassifier()

load_dotenv(override=True)

# ── In-memory log capture (last 500 lines) ────────────────────────────────────
from collections import deque

class _DequeHandler(logging.Handler):
    def __init__(self, maxlen=500):
        super().__init__()
        self.records: deque = deque(maxlen=maxlen)

    def emit(self, record):
        from datetime import datetime
        self.records.append({
            'time':    datetime.now().strftime('%H:%M:%S'),
            'level':   record.levelname,
            'name':    record.name,
            'message': self.format(record),
        })

_log_handler = _DequeHandler(maxlen=500)
_log_handler.setFormatter(logging.Formatter('%(message)s'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s  %(message)s',
    handlers=[logging.StreamHandler(), _log_handler],
)
logger = logging.getLogger(__name__)
logging.getLogger('trading_system').addHandler(_log_handler)
logging.getLogger('httpx').addHandler(_log_handler)
logging.getLogger('werkzeug').addHandler(_log_handler)
# ─────────────────────────────────────────────────────────────────────────────

from flask_cors import CORS
from flask.json.provider import DefaultJSONProvider
import numpy as np

class _NumpyJSONProvider(DefaultJSONProvider):
    """Extend Flask's JSON provider to handle numpy scalars gracefully."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app = Flask(__name__)
app.json_provider_class = _NumpyJSONProvider
app.json = _NumpyJSONProvider(app)
CORS(app)
memory = MemoryManager()

# Global orchestrator
_orchestrator: Optional[OrchestratorAgent] = None

def get_orchestrator() -> OrchestratorAgent:
    if _orchestrator is None:
        raise ValueError("Not connected — go to Config tab and click Connect.")
    return _orchestrator

def human_approval_popup(order: dict) -> bool:
    return False

import time
import requests
import xml.etree.ElementTree as ET
import re
import urllib.request

def fetch_single_feed(source_name, url, region):
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        )
        with urllib.request.urlopen(req, timeout=3) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        items = []
        for item in root.findall('.//item')[:10]:
            title = item.find('title')
            link = item.find('link')
            pub_date = item.find('pubDate')
            description = item.find('description')
            
            title_text = title.text if title is not None else 'No Title'
            link_text = link.text if link is not None else '#'
            pub_date_text = pub_date.text if pub_date is not None else ''
            desc_text = description.text if description is not None else ''
            
            if desc_text:
                desc_text = re.sub('<[^<]+?>', '', desc_text)
                if len(desc_text) > 200:
                    desc_text = desc_text[:200] + '...'
            
            # Smart classification using local Naive Bayes ML model + keyword rules
            item_region = region
            method = "Default Source"
            confidence = 1.0
            title_lower = title_text.lower()
            desc_lower = desc_text.lower()
            if any(kw in title_lower or kw in desc_lower for kw in ['nifty', 'sensex', 'nse', 'bse', 'rbi', 'rupee', 'narendra modi', 'india', 'indian market']):
                item_region = 'Indian'
                method = 'Rule Override'
            elif any(kw in title_lower or kw in desc_lower for kw in ['nasdaq', 's&p 500', 'dow jones', 'federal reserve', 'fomc', 'ecb', 'boj', 'boe']):
                item_region = 'Global'
                method = 'Rule Override'
            else:
                pred_res = news_classifier.predict_with_confidence(title_text, desc_text)
                item_region = pred_res["region"]
                method = pred_res["method"]
                confidence = pred_res["confidence"]
                    
            items.append({
                'title': title_text,
                'link': link_text,
                'pub_date': pub_date_text,
                'description': desc_text,
                'source': source_name,
                'region': item_region,
                'method': method,
                'confidence': confidence
            })
        return items
    except Exception as e:
        logger.error(f"Error fetching RSS from {source_name}: {e}")
        return []

def fetch_rss_feed(url):
    return fetch_single_feed('News', url, 'Global')

_delta_products_cache = {
    'data': [],
    'expires_at': 0
}

def get_delta_products():
    now = time.time()
    if _delta_products_cache['expires_at'] < now or not _delta_products_cache['data']:
        try:
            r = requests.get("https://api.delta.exchange/v2/tickers", timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get('success') and data.get('result'):
                    products = []
                    for x in data['result']:
                        products.append({
                            'symbol': x['symbol'],
                            'name': x.get('description') or x['symbol'],
                            'sector': x.get('contract_type', 'crypto'),
                            'status': x.get('product_trading_status', 'operational')
                        })
                    _delta_products_cache['data'] = products
                    _delta_products_cache['expires_at'] = now + 120
        except Exception as e:
            logger.error("Failed to fetch Delta tickers: %s", e)
    return _delta_products_cache['data']

def get_mock_portfolio(mode: str):
    """Build a mock portfolio from DB trade logs for Forex or Crypto."""
    try:
        import psycopg2, psycopg2.extras
        from trading_system.memory import DB_CONFIG
        
        trades = []
        try:
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute("SELECT data, logged_at FROM trade_logs WHERE market_mode = %s ORDER BY logged_at ASC", (mode,))
                    for row in cur.fetchall():
                        trade_data = dict(row['data'])
                        trade_data['logged_at'] = row['logged_at']
                        trades.append(trade_data)
        except Exception as e:
            logger.error("DB mock trade logs fetch failed: %s", e)
            return "### 💼 Mock Portfolio\n\nNo trades logged in the database yet."

        filtered_trades = []
        for t in trades:
            sym = t.get('symbol', '').upper()
            is_crypto_sym = sym.endswith("USDT") or sym.startswith("P-") or sym.startswith("C-") or sym.startswith("F-")
            is_forex_sym = sym.endswith("=X")
            
            if mode == 'crypto' and is_crypto_sym:
                filtered_trades.append(t)
            elif mode == 'forex' and is_forex_sym:
                filtered_trades.append(t)

        if not filtered_trades:
            m_name = "Crypto" if mode == 'crypto' else "Forex"
            return f"### 💼 Mock {m_name} Portfolio\n\nNo {m_name.lower()} trades found in trade logs. Place a trade in the chat or analysis section to see it here!"

        positions = {}
        for t in filtered_trades:
            sym = t['symbol']
            qty = int(t.get('quantity', 0))
            price = float(t.get('price', 0.0))
            txn = t.get('transaction_type') or t.get('transaction', 'BUY')
            
            if price <= 0:
                price = float(t.get('trigger_price', 0.0))
                
            if price <= 0:
                continue
                
            if sym not in positions:
                positions[sym] = {'qty': 0, 'total_cost': 0.0}
                
            if txn.upper() == 'BUY':
                positions[sym]['qty'] += qty
                positions[sym]['total_cost'] += qty * price
            else:
                positions[sym]['qty'] -= qty
                positions[sym]['total_cost'] -= qty * price

        active_positions = {}
        for sym, data in positions.items():
            if data['qty'] != 0:
                qty = data['qty']
                if qty > 0:
                    avg_price = data['total_cost'] / qty
                else:
                    avg_price = abs(data['total_cost'] / qty) if qty != 0 else 0.0
                active_positions[sym] = {
                    'quantity': qty,
                    'average_price': round(avg_price, 4),
                    'total_invested': round(qty * avg_price, 4) if qty > 0 else 0.0
                }

        if not active_positions:
            m_name = "Crypto" if mode == 'crypto' else "Forex"
            return f"### 💼 Mock {m_name} Portfolio\n\nAll positions have been squared off (net quantity is zero)."

        symbols_to_fetch = list(active_positions.keys())
        ltps = {}
        if mode == 'crypto':
            try:
                r = requests.get("https://api.delta.exchange/v2/tickers", timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    if data.get('success') and data.get('result'):
                        tickers = {x['symbol']: x for x in data['result']}
                        for sym in symbols_to_fetch:
                            t = tickers.get(sym)
                            price = t.get('close') or t.get('mark_price') or t.get('spot_price') if t else None
                            if price:
                                ltps[sym] = float(price)
            except Exception as e:
                logger.error("Delta tickers fetch failed for portfolio LTPs: %s", e)
        else:
            for sym in symbols_to_fetch:
                try:
                    q = T._yf_quote(sym)
                    if q.get('last_price'):
                        ltps[sym] = q.get('last_price')
                except Exception as e:
                    logger.error("Forex yfinance quote failed for portfolio LTP %s: %s", sym, e)

        m_name = "Crypto" if mode == 'crypto' else "Forex"
        md = f"### 💼 Mock {m_name} Portfolio\n\n"
        md += "| Symbol | Position | Quantity | Avg Price | Current Price | Total Invested | Current Value | P&L | P&L % |\n"
        md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        
        total_inv = 0.0
        total_val = 0.0
        
        for sym, data in active_positions.items():
            qty = data['quantity']
            avg = data['average_price']
            ltp = ltps.get(sym, avg)
            
            direction = "🟢 Long" if qty > 0 else "🔴 Short"
            invested = data['total_invested']
            
            if qty > 0:
                current_value = qty * ltp
                pnl = current_value - invested
            else:
                invested = abs(qty) * avg
                current_value = invested + (avg - ltp) * abs(qty)
                pnl = (avg - ltp) * abs(qty)

            pnl_pct = (pnl / invested * 100) if invested > 0 else 0.0
            
            total_inv += invested
            total_val += current_value
            
            color_prefix = "🟢" if pnl >= 0 else "🔴"
            
            md += f"| **{sym}** | {direction} | {abs(qty)} | {avg:.4f} | {ltp:.4f} | {invested:.2f} | {current_value:.2f} | {color_prefix} {pnl:.2f} | {color_prefix} {pnl_pct:.2f}% |\n"

        net_pnl = total_val - total_inv
        net_pnl_pct = (net_pnl / total_inv * 100) if total_inv > 0 else 0.0
        net_color = "🟢" if net_pnl >= 0 else "🔴"
        
        md += f"| **TOTAL** | | | | | **{total_inv:.2f}** | **{total_val:.2f}** | **{net_color} {net_pnl:.2f}** | **{net_color} {net_pnl_pct:.2f}%** |\n\n"
        md += f"> Live valuation using Delta Exchange API (Crypto) and Yahoo Finance (Forex). Sorted by date."
        
        return md
    except Exception as e:
        logger.exception("Failed to build mock portfolio")
        return f"### 💼 Mock Portfolio Error\n\nFailed to build mock portfolio: {e}"

# --- UI Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/backtest')
def backtest():
    return render_template('backtest.html')

@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

@app.route('/news')
def news():
    return render_template('news.html')

FEEDS = {
    'equity': [
        ('CNBC', 'https://www.cnbc.com/id/10000664/device/rss/rss.html', 'Global'),
        ('Wall Street Journal', 'https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml', 'Global'),
        ('Bloomberg', 'https://news.google.com/rss/search?q=when:24h+markets+site:bloomberg.com&hl=en-US&gl=US&ceid=US:en', 'Global'),
        ('Reuters', 'https://news.google.com/rss/search?q=when:24h+business+site:reuters.com&hl=en-US&gl=US&ceid=US:en', 'Global'),
        ('Financial Times', 'https://news.google.com/rss/search?q=when:24h+markets+site:ft.com&hl=en-US&gl=US&ceid=US:en', 'Global'),
        ('Economic Times', 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms', 'Indian'),
        ('LiveMint', 'https://www.livemint.com/rss/markets', 'Indian'),
        ('Business Standard', 'https://www.business-standard.com/rss/markets-104.rss', 'Indian'),
        ('MoneyControl', 'https://www.moneycontrol.com/rss/marketoutlook.xml', 'Indian'),
    ],
    'forex': [
        ('DailyFX', 'https://www.dailyfx.com/feeds/forex-market-news', 'Global'),
        ('Investing.com', 'https://www.investing.com/rss/forex.rss', 'Global'),
        ('Reuters', 'https://news.google.com/rss/search?q=when:24h+forex+site:reuters.com&hl=en-US&gl=US&ceid=US:en', 'Global'),
        ('Bloomberg', 'https://news.google.com/rss/search?q=when:24h+forex+site:bloomberg.com&hl=en-US&gl=US&ceid=US:en', 'Global'),
        ('Economic Times Forex', 'https://economictimes.indiatimes.com/markets/forex/rssfeeds/11833503.cms', 'Indian'),
    ],
    'crypto': [
        ('Cointelegraph', 'https://cointelegraph.com/rss', 'Global'),
        ('Coindesk', 'https://www.coindesk.com/arc/outboundfeeds/rss/', 'Global'),
        ('Bloomberg', 'https://news.google.com/rss/search?q=when:24h+crypto+site:bloomberg.com&hl=en-US&gl=US&ceid=US:en', 'Global'),
        ('Reuters', 'https://news.google.com/rss/search?q=when:24h+crypto+site:reuters.com&hl=en-US&gl=US&ceid=US:en', 'Global'),
    ]
}

@app.route('/api/news', methods=['GET'])
def api_news():
    mode = request.args.get('mode', 'equity').lower()
    feeds_list = FEEDS.get(mode, FEEDS['equity'])
    
    import concurrent.futures
    all_items = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(feeds_list)) as executor:
        futures = [executor.submit(fetch_single_feed, name, url, region) for name, url, region in feeds_list]
        for future in concurrent.futures.as_completed(futures):
            all_items.extend(future.result())
            
    # Deduplicate
    seen = set()
    deduped_items = []
    for item in all_items:
        key = item['title'].lower().strip()
        if key not in seen:
            seen.add(key)
            deduped_items.append(item)
            
    # Sort by date
    def parse_pub_date(item):
        date_str = item.get('pub_date', '')
        try:
            import email.utils
            return email.utils.parsedate_to_datetime(date_str)
        except Exception:
            from datetime import datetime, timezone
            return datetime.fromtimestamp(0, tz=timezone.utc)
            
    deduped_items.sort(key=parse_pub_date, reverse=True)
    items = deduped_items[:25]
    
    # 1. Fetch VIX score from yfinance
    vix_score = 15.0  # default fallback
    try:
        import yfinance as yf
        vix_sym = "^INDIAVIX" if mode == 'equity' else "^VIX"
        ticker = yf.Ticker(vix_sym)
        history = ticker.history(period="1d")
        if not history.empty:
            vix_score = float(history['Close'].iloc[-1])
    except Exception as e:
        logger.error("Failed to fetch VIX score from yfinance: %s", e)
        
    # 2. Compute Sentiment lexicon analysis
    pos_words = {'bullish', 'rise', 'gain', 'rally', 'growth', 'up', 'soar', 'record', 'surge', 'optimistic', 'positive', 'higher', 'advance', 'profit', 'stimulus', 'rebound', 'jump'}
    neg_words = {'bearish', 'fall', 'drop', 'slump', 'decline', 'fear', 'loss', 'down', 'crash', 'plunge', 'worry', 'slide', 'dread', 'recession', 'correction', 'inflation', 'panic'}
    
    total_pos = 0
    total_neg = 0
    import re
    for item in items:
        text = (item.get('title', '') + ' ' + item.get('description', '')).lower()
        words = set(re.findall(r'\b\w+\b', text))
        total_pos += len(words.intersection(pos_words))
        total_neg += len(words.intersection(neg_words))
        
    total = total_pos + total_neg
    if total > 0:
        sentiment_score = (total_pos - total_neg) / total
    else:
        sentiment_score = 0.0
        
    if sentiment_score >= 0.15:
        sentiment_label = "BULLISH"
    elif sentiment_score <= -0.15:
        sentiment_label = "BEARISH"
    else:
        sentiment_label = "NEUTRAL"
        
    return jsonify({
        'news': items,
        'vix_score': round(vix_score, 2),
        'sentiment': {
            'score': round(sentiment_score, 2),
            'label': sentiment_label
        }
    })

# --- Static File Serving for Charts ---

@app.route('/data/visualizations/<path:filename>')
def serve_visualizations(filename):
    return send_from_directory(os.path.join(app.root_path, 'data', 'visualizations'), filename)


@app.route('/api/watchlist', methods=['GET'])
def api_watchlist():
    """Return watchlist with live LTP based on the active mode."""
    mode = request.args.get('mode', 'equity').lower()
    
    if mode == 'crypto':
        default_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'PAXGUSDT']
        result = []
        try:
            r = requests.get("https://api.delta.exchange/v2/tickers", timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get('success') and data.get('result'):
                    tickers = {x['symbol']: x for x in data['result']}
                    for sym in default_symbols:
                        t = tickers.get(sym, {})
                        price = t.get('close') or t.get('mark_price') or t.get('spot_price')
                        change_pct = t.get('ltp_change_24h')
                        result.append({
                            'symbol':           sym,
                            'name':             t.get('description') or sym,
                            'ltp':              float(price) if price else None,
                            'change':           None,
                            'change_pct':       float(change_pct) if change_pct else None,
                            'sector':           t.get('contract_type', 'crypto'),
                            'demerger_details': None,
                        })
        except Exception as e:
            logger.warning("Delta watchlist fetch failed: %s", e)
            for sym in default_symbols:
                result.append({
                    'symbol':           sym,
                    'name':             sym,
                    'ltp':              None,
                    'change':           None,
                    'change_pct':       None,
                    'sector':           'crypto',
                    'demerger_details': None,
                })
        return jsonify({'watchlist': result})

    elif mode == 'forex':
        default_symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X']
        result = []
        from src.trading_system.api_manager import api_manager
        for sym in default_symbols:
            try:
                # Check WebSocket cache first
                cached = api_manager.yahoo_live_prices.get(sym)
                if cached and (time.time() - cached['updated_at'] < 60):
                    ltp = cached['price']
                    change = cached['change']
                    pct = cached['change_percent']
                else:
                    q = T._yf_quote(sym)
                    ltp = q.get('last_price')
                    prev = q.get('previous_close')
                    change = ltp - prev if (ltp and prev) else None
                    pct = (change / prev * 100) if (change and prev) else None
                    # Update cache
                    if ltp is not None:
                        api_manager.yahoo_live_prices[sym] = {
                            'price': ltp,
                            'change': change or 0,
                            'change_percent': pct or 0,
                            'updated_at': time.time()
                        }
                        
                result.append({
                    'symbol':           sym,
                    'name':             f"{sym[:3]}/{sym[3:6]}" if len(sym) >= 6 else sym,
                    'ltp':              ltp,
                    'change':           round(change, 5) if change else None,
                    'change_pct':       round(pct, 4) if pct else None,
                    'sector':           'Forex Rate',
                    'demerger_details': None,
                })
            except Exception as e:
                logger.warning("Forex watchlist fetch failed for %s: %s", sym, e)
                result.append({
                    'symbol':           sym,
                    'name':             sym,
                    'ltp':              None,
                    'change':           None,
                    'change_pct':       None,
                    'sector':           'Forex Rate',
                    'demerger_details': None,
                })
        return jsonify({'watchlist': result})

    else:
        # Original Equity watchlist code
        default_symbols = [
            'NSE:NIFTY 50', 'NSE:NIFTY BANK', 'NSE:RELIANCE', 'NSE:TCS',
            'NSE:INFY', 'NSE:HDFCBANK', 'NSE:ICICIBANK', 'NSE:WIPRO',
            'NSE:BAJFINANCE', 'NSE:TATAMOTORS', 'NSE:TMCV', 'NSE:TMPV',
            'NSE:ADANIENT', 'NSE:SBIN', 'NSE:ITC', 'NSE:AXISBANK',
        ]

        db_meta: dict = {}
        try:
            import psycopg2, psycopg2.extras
            from trading_system.memory import DB_CONFIG
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute(
                        "SELECT symbol, name, sector, demerger_details FROM tickers_classification"
                    )
                    for row in cur.fetchall():
                        db_meta[row['symbol']] = dict(row)
        except Exception as e:
            logger.debug("DB metadata fetch skipped (DB not connected): %s", e)

        db_symbols = list(db_meta.keys())
        all_symbols = list(dict.fromkeys(default_symbols + db_symbols))

        result = []
        try:
            if T.kite_available():
                kite = T.get_kite()
                quotes_raw = kite.quote(all_symbols)
                for sym in all_symbols:
                    q    = quotes_raw.get(sym, {})
                    meta = db_meta.get(sym, {})
                    result.append({
                        'symbol':           sym,
                        'name':             meta.get('name') or sym.split(':')[-1],
                        'ltp':              q.get('last_price'),
                        'change':           q.get('net_change'),
                        'change_pct':       q.get('change'),
                        'sector':           meta.get('sector'),
                        'demerger_details': meta.get('demerger_details'),
                    })
            else:
                from src.trading_system.api_manager import api_manager
                from src.trading_system.tools import _kite_to_yf
                for sym in all_symbols:
                    meta = db_meta.get(sym, {})
                    yf_sym = _kite_to_yf(sym)
                    cached = api_manager.yahoo_live_prices.get(yf_sym)
                    ltp = cached['price'] if cached else None
                    change = cached['change'] if cached else None
                    change_pct = cached['change_percent'] if cached else None
                    result.append({
                        'symbol':           sym,
                        'name':             meta.get('name') or sym.split(':')[-1],
                        'ltp':              ltp,
                        'change':           change,
                        'change_pct':       change_pct,
                        'sector':           meta.get('sector'),
                        'demerger_details': meta.get('demerger_details'),
                    })
        except Exception as e:
            logger.warning("Watchlist LTP fetch failed: %s", e)
            for sym in all_symbols:
                meta = db_meta.get(sym, {})
                result.append({
                    'symbol': sym,
                    'name':   meta.get('name') or sym.split(':')[-1],
                    'sector': meta.get('sector'),
                })
        return jsonify({'watchlist': result})




# --- API Endpoints ---

@app.route('/api/connect', methods=['POST'])
def api_connect():
    global _orchestrator
    try:
        data = request.json
        provider = data.get('provider')
        model = data.get('model', '').strip()
        api_key = data.get('api_key', '').strip()
        base_url = data.get('base_url', '').strip()
        kite_key = data.get('kite_key', '').strip()
        kite_token = data.get('kite_token', '').strip()
        
        # Determine langchain provider
        langchain_provider = {
            "openrouter": "openai",
            "openai": "openai",
            "anthropic": "anthropic",
            "google-gemini": "google-genai",
        }.get(provider, "openai")
        
        # Init Kite
        if kite_key and kite_token:
            T.init_kite(kite_key, kite_token)
            
        _orchestrator = build_orchestrator(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://openrouter.ai/api/v1",
            temperature=0.0,
            max_tokens=4096,
            approval_fn=human_approval_popup,
        )
        
        kite_status = 'Connected ✅' if T.kite_available() else 'Not Connected ❌'
        return jsonify({"message": f"✅ LLM Connected\nModel: {model}\nKite: {kite_status}"})
    except Exception as e:
        logger.exception("Connect failed")
        return jsonify({"error": f"❌ Error: {e}"}), 400


@app.route('/api/test_kite', methods=['POST'])
def api_test_kite():
    try:
        data = request.json
        api_key = data.get('kite_key', '').strip()
        token = data.get('kite_token', '').strip()
        
        if not api_key or not token:
            return jsonify({"message": "🪁 Kite Connection: **False** (Missing credentials ❌)"})
            
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(token)
        kite.profile()
        return jsonify({"message": "🪁 Kite Connection: **True** (Connected ✅)"})
    except Exception as e:
        return jsonify({"message": f"🪁 Kite Connection: **False** (Error: {e} ❌)"})



@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        msg = request.json.get('message')
        mode = request.json.get('mode', 'equity').lower()
        full_msg = f"[System Context: The user is currently in {mode.upper()} trading mode. Respond and query symbols accordingly.]\n\n{msg}"
        orch = get_orchestrator()
        response = orch.run(full_msg)
        html_response = markdown.markdown(response, extensions=['fenced_code', 'tables'])
        return jsonify({"response": html_response})
    except Exception as e:
        return jsonify({"response": f"❌ Error: {e}"}), 400


@app.route('/api/analysis', methods=['POST'])
def api_analysis():
    try:
        symbol = request.json.get('symbol', '').strip().upper()
        results = get_orchestrator().analyse_symbol(symbol)
        
        output = "\n\n".join(f"**{k}**\n{v}" for k, v in results.items())
        html_output = markdown.markdown(output, extensions=['fenced_code', 'tables'])
        
        chart_name = f"{symbol.replace(':', '_')}_60d"
        chart_url = f"/data/visualizations/{chart_name}.html"
        
        return jsonify({
            "output": html_output,
            "chart_url": chart_url
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    try:
        data = request.json
        symbol = data.get('symbol').strip().upper()
        strategy = data.get('strategy')
        params_json = data.get('params')
        days = int(data.get('days', 365))
        
        result = get_orchestrator().strategy.run(
            f"Backtest {strategy} on {symbol} for {days} days with params {params_json}"
        )
        
        strat_name = f"{symbol}_{strategy}_{days}d"
        chart_resp = get_orchestrator().viz.run(f"Create backtest chart for {strat_name}")
        
        html_output = markdown.markdown(result, extensions=['fenced_code', 'tables'])
        chart_url = f"/data/visualizations/{strat_name}_equity.html"
        
        return jsonify({
            "output": html_output,
            "chart_url": chart_url
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/portfolio', methods=['POST'])
def api_portfolio():
    try:
        mode = request.json.get('mode', 'equity').lower() if request.json else 'equity'
        if mode in ('forex', 'crypto'):
            result = get_mock_portfolio(mode)
            html_output = markdown.markdown(result, extensions=['fenced_code', 'tables'])
            chart_url = ""
        else:
            result = get_orchestrator().get_dashboard()
            html_output = markdown.markdown(result, extensions=['fenced_code', 'tables'])
            chart_url = "/data/visualizations/portfolio_dashboard.html"
        
        return jsonify({
            "output": html_output,
            "chart_url": chart_url
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def get_mock_portfolio_stats(mode: str):
    """Calculate aggregate stats for mock Forex/Crypto trades."""
    try:
        import psycopg2, psycopg2.extras
        from trading_system.memory import DB_CONFIG
        import trading_system.tools as T
        
        trades = []
        try:
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute("SELECT data FROM trade_logs WHERE market_mode = %s ORDER BY logged_at ASC", (mode,))
                    for row in cur.fetchall():
                        trades.append(dict(row['data']))
        except Exception as e:
            logger.error("DB mock trade logs fetch failed: %s", e)
            return {"total_invested": 0.0, "total_current": 0.0, "total_pnl": 0.0, "pnl_pct": 0.0, "num_holdings": 0}

        filtered_trades = []
        for t in trades:
            sym = t.get('symbol', '').upper()
            is_crypto_sym = sym.endswith("USDT") or sym.startswith("P-") or sym.startswith("C-") or sym.startswith("F-")
            is_forex_sym = sym.endswith("=X")
            
            if mode == 'crypto' and is_crypto_sym:
                filtered_trades.append(t)
            elif mode == 'forex' and is_forex_sym:
                filtered_trades.append(t)

        positions = {}
        for t in filtered_trades:
            sym = t['symbol']
            qty = int(t.get('quantity', 0))
            price = float(t.get('price', 0.0))
            txn = t.get('transaction_type') or t.get('transaction', 'BUY')
            
            if price <= 0:
                price = float(t.get('trigger_price', 0.0))
                
            if price <= 0:
                continue
                
            if sym not in positions:
                positions[sym] = {'qty': 0, 'total_cost': 0.0}
                
            if txn.upper() == 'BUY':
                positions[sym]['qty'] += qty
                positions[sym]['total_cost'] += qty * price
            else:
                positions[sym]['qty'] -= qty
                positions[sym]['total_cost'] -= qty * price

        active_positions = {}
        for sym, data in positions.items():
            if data['qty'] != 0:
                qty = data['qty']
                if qty > 0:
                    avg_price = data['total_cost'] / qty
                else:
                    avg_price = abs(data['total_cost'] / qty) if qty != 0 else 0.0
                active_positions[sym] = {
                    'quantity': qty,
                    'average_price': avg_price,
                }

        if not active_positions:
            return {"total_invested": 0.0, "total_current": 0.0, "total_pnl": 0.0, "pnl_pct": 0.0, "num_holdings": 0}

        symbols_to_fetch = list(active_positions.keys())
        ltps = {}
        if mode == 'crypto':
            try:
                r = requests.get("https://api.delta.exchange/v2/tickers", timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    if data.get('success') and data.get('result'):
                        tickers = {x['symbol']: x for x in data['result']}
                        for sym in symbols_to_fetch:
                            t = tickers.get(sym)
                            price = t.get('close') or t.get('mark_price') or t.get('spot_price') if t else None
                            if price:
                                ltps[sym] = float(price)
            except Exception as e:
                logger.error("Delta tickers fetch failed for portfolio stats LTPs: %s", e)
        else:
            for sym in symbols_to_fetch:
                try:
                    q = T._yf_quote(sym)
                    if q.get('last_price'):
                        ltps[sym] = q.get('last_price')
                except Exception as e:
                    logger.error("Forex yfinance quote failed for portfolio stats LTP %s: %s", sym, e)

        total_invested = 0.0
        total_current = 0.0
        for sym, pos in active_positions.items():
            qty = pos['quantity']
            avg_price = pos['average_price']
            ltp = ltps.get(sym, avg_price) # fallback to avg_price
            
            if qty > 0:
                inv = qty * avg_price
                curr = qty * ltp
            else:
                inv = abs(qty) * avg_price
                pnl = (avg_price - ltp) * abs(qty)
                curr = inv + pnl
                
            total_invested += inv
            total_current += curr

        total_pnl = total_current - total_invested
        pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0.0
        
        return {
            "total_invested": round(total_invested, 2),
            "total_current": round(total_current, 2),
            "total_pnl": round(total_pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "num_holdings": len(active_positions),
        }
    except Exception as e:
        logger.error("Error in get_mock_portfolio_stats: %s", e)
        return {"total_invested": 0.0, "total_current": 0.0, "total_pnl": 0.0, "pnl_pct": 0.0, "num_holdings": 0}


@app.route('/api/portfolio_stats', methods=['GET'])
def api_portfolio_stats():
    """Return aggregate portfolio stats: total invested, current value, total P&L."""
    try:
        mode = request.args.get('mode', 'equity').lower()
        if mode in ('forex', 'crypto'):
            return jsonify(get_mock_portfolio_stats(mode))

        import trading_system.tools as T
        import json as _json
        raw = _json.loads(T.get_portfolio.invoke({}))
        if "error" in raw:
            return jsonify({"error": raw["error"]}), 400

        holdings = raw.get("holdings", [])
        total_invested = sum(h.get("average_price", 0) * h.get("quantity", 0) for h in holdings)
        total_current  = sum(h.get("last_price", 0)   * h.get("quantity", 0) for h in holdings)
        total_pnl      = total_current - total_invested
        pnl_pct        = (total_pnl / total_invested * 100) if total_invested else 0

        return jsonify({
            "total_invested": round(total_invested, 2),
            "total_current":  round(total_current,  2),
            "total_pnl":      round(total_pnl,       2),
            "pnl_pct":        round(pnl_pct,         2),
            "num_holdings":   len(holdings),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400



@app.route('/api/logs', methods=['GET'])
def api_logs():
    """Return recent captured log records."""
    n = int(request.args.get('n', 200))
    records = list(_log_handler.records)[-n:]
    return jsonify({'logs': records})

_kite_historical_permitted = True
_kite_ltp_permitted = True


@app.route('/api/quotes', methods=['POST'])
def api_quotes():
    """Fetch live/last quotes for multiple symbols via get_quote tool."""
    try:
        symbols = request.json.get('symbols', 'NSE:INFY,NSE:RELIANCE,NSE:TCS')
        raw = T.get_quote.invoke({'symbols': symbols})
        return jsonify(json.loads(raw))
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/ohlcv', methods=['POST'])
def api_ohlcv():
    """
    Return OHLCV candles + EMA9/21/50 for Lightweight Charts.
    Body: { symbol, interval, days }
    interval: '1minute'|'5minute'|'15minute'|'60minute'|'day'|'week'
    """
    import pandas as pd, numpy as np
    global _kite_historical_permitted
    try:
        body     = request.json or {}
        symbol   = body.get('symbol',   'NSE:INFY')
        interval = body.get('interval', 'day')
        days     = int(body.get('days', 180))

        # ── Fetch based on routed API ──────────────────────────────────
        from src.trading_system.api_manager import api_manager
        
        # Determine target API to use based on the routing strategy
        routed_api = api_manager.route_ohlcv_request(symbol, interval, days)
        logger.info(f"Routed OHLCV request for {symbol} ({days} days) to: {routed_api}")
        
        df = None
        
        # 1. Delta
        if routed_api == 'delta':
            try:
                import time
                end_time = int(time.time())
                start_time = int(end_time - days * 86400)
                res_map = {
                    '1minute': '1m', '5minute': '5m', '15minute': '15m',
                    '60minute': '1h', 'day': '1d', 'week': '1w'
                }
                resolution = res_map.get(interval, '1d')
                params = {
                    'symbol': symbol,
                    'resolution': resolution,
                    'start': start_time,
                    'end': end_time
                }
                r = requests.get("https://api.delta.exchange/v2/history/candles", params=params, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    if data.get('success') and data.get('result'):
                        candles_raw = sorted(data['result'], key=lambda c: c['time'])
                        df = pd.DataFrame(candles_raw)
                        if not df.empty:
                            df['time'] = pd.to_datetime(df['time'], unit='s')
                            api_manager.track_call('delta')
            except Exception as e:
                logger.warning("Delta candles fetch failed: %s. Falling back.", e)
                
        # 2. Kite
        elif routed_api == 'kite':
            if T.kite_available() and _kite_historical_permitted:
                try:
                    kite = T.get_kite()
                    ltp_data  = kite.ltp(symbol)
                    token     = ltp_data[symbol]['instrument_token']
                    from_dt   = __import__('datetime').datetime.now() - __import__('datetime').timedelta(days=days)
                    to_dt     = __import__('datetime').datetime.now()
                    raw_candles = kite.historical_data(token, from_dt, to_dt, interval)
                    df = pd.DataFrame(raw_candles)
                    if not df.empty:
                        df = df.rename(columns={'date':'time','open':'open','high':'high',
                                                'low':'low','close':'close','volume':'volume'})
                        api_manager.track_call('kite')
                except Exception as e:
                    logger.warning("Kite OHLCV failed: %s. Falling back.", e)
                    if "permission" in str(e).lower() or "privilege" in str(e).lower():
                        _kite_historical_permitted = False
                        logger.info("Permanently disabling Kite historical data calls due to insufficient permissions. Bypassing to Yahoo Finance fallback directly.")
                    df = None

        # 3. Finnhub
        elif routed_api == 'finnhub':
            global _finnhub_api_key
            if _finnhub_api_key:
                import time
                from src.trading_system.tools import _kite_to_yf
                fh_sym = _kite_to_yf(symbol)
                res_map = {'minute': '1', '1minute': '1', '5minute': '5', '15minute': '15', '30minute': '30', '60minute': '60', 'day': 'D', 'week': 'W', 'month': 'M'}
                fh_res = res_map.get(interval, 'D')
                end_t = int(time.time())
                start_t = end_t - (days * 86400)
                fh_url = f"https://finnhub.io/api/v1/stock/candle?symbol={fh_sym}&resolution={fh_res}&from={start_t}&to={end_t}&token={_finnhub_api_key}"
                try:
                    logger.info(f"Attempting to fetch Finnhub OHLCV for {fh_sym} (res: {fh_res})")
                    r = requests.get(fh_url, timeout=5)
                    if r.status_code == 200:
                        d = r.json()
                        if d.get('s') == 'ok':
                            df = pd.DataFrame({
                                'time': pd.to_datetime(d['t'], unit='s'),
                                'open': d['o'],
                                'high': d['h'],
                                'low': d['l'],
                                'close': d['c'],
                                'volume': d['v']
                            })
                            logger.info(f"Successfully fetched {len(df)} candles from Finnhub for {fh_sym}")
                            api_manager.track_call('finnhub')
                        else:
                            logger.warning(f"Finnhub API returned status '{d.get('s')}' for {fh_sym}")
                    else:
                        logger.warning(f"Finnhub API request failed with status {r.status_code}")
                except Exception as e:
                    logger.warning(f"Finnhub OHLCV failed: {e}")

        # 4. Fallback to Yahoo Finance
        if df is None or df.empty:
            if routed_api != 'yfinance':
                logger.info(f"Routed API ({routed_api}) failed or returned empty. Falling back to Yahoo Finance.")
            
            import yfinance as yf
            from src.trading_system.tools import _kite_to_yf, _yf_interval, _days_to_period
            yf_sym  = _kite_to_yf(symbol)
            yf_int  = _yf_interval(interval)
            period  = _days_to_period(days, yf_int)
            ticker  = yf.Ticker(yf_sym)
            raw_df  = ticker.history(period=period, interval=yf_int, auto_adjust=True)
            if raw_df.empty:
                return jsonify({'error': f'No data for {symbol}'}), 404
            raw_df  = raw_df.reset_index()
            raw_df.columns = [c.lower() for c in raw_df.columns]
            time_col = 'datetime' if 'datetime' in raw_df.columns else 'date'
            df = raw_df.rename(columns={time_col: 'time'})[['time','open','high','low','close','volume']]
            api_manager.track_call('yfinance')

        # ── Clean ───────────────────────────────────────────────────────
        df = df.dropna(subset=['close']).copy()
        df['time'] = pd.to_datetime(df['time'])
        if hasattr(df['time'].dtype, 'tz') and df['time'].dtype.tz is not None:
            df['time'] = df['time'].dt.tz_localize(None)
        df = df.sort_values('time').reset_index(drop=True)

        # ── EMA helper ──────────────────────────────────────────────────
        def ema(series, span):
            return series.ewm(span=span, adjust=False).mean()

        close      = df['close']
        df['ema9']  = ema(close, 9).round(4)
        df['ema21'] = ema(close, 21).round(4)
        df['ema50'] = ema(close, 50).round(4)
        df['ema200']= ema(close, 200).round(4)

        # ── RSI 14 ──────────────────────────────────────────────────────
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / loss.replace(0, float('nan'))
        df['rsi14'] = (100 - 100 / (1 + rs)).round(4)

        # ── Serialise timestamp to Unix seconds (LWC format) ────────────
        def ts(row):
            return int(row['time'].timestamp())

        candles = []
        for _, r in df.iterrows():
            candles.append({
                'time':  ts(r),
                'open':  round(float(r['open']),  4),
                'high':  round(float(r['high']),  4),
                'low':   round(float(r['low']),   4),
                'close': round(float(r['close']), 4),
                'volume':int(r.get('volume', 0) or 0),
                'ema9':   None if np.isnan(r['ema9'])   else round(float(r['ema9']),  4),
                'ema21':  None if np.isnan(r['ema21'])  else round(float(r['ema21']), 4),
                'ema50':  None if np.isnan(r['ema50'])  else round(float(r['ema50']), 4),
                'ema200': None if np.isnan(r['ema200']) else round(float(r['ema200']),4),
                'rsi14':  None if np.isnan(r['rsi14'])  else round(float(r['rsi14']), 4),
            })

        last = candles[-1] if candles else {}
        return jsonify({
            'symbol':   symbol,
            'interval': interval,
            'count':    len(candles),
            'last_close': last.get('close'),
            'candles':  candles,
        })
    except Exception as e:
        logger.exception("Failed to get candles")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ltp', methods=['POST'])
def api_ltp():
    """Return last-traded prices for a list of symbols (Kite, Delta, or Yahoo fallback)."""
    global _kite_ltp_permitted
    try:
        symbols = request.json.get('symbols', [])
        result  = {}
        
        crypto_symbols = [s for s in symbols if s.upper().strip().endswith("USDT") or s.upper().strip().startswith("P-") or s.upper().strip().startswith("C-") or s.upper().strip().startswith("F-")]
        other_symbols = [s for s in symbols if s not in crypto_symbols]
        
        if crypto_symbols:
            try:
                r = requests.get("https://api.delta.exchange/v2/tickers", timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    if data.get('success') and data.get('result'):
                        tickers = {x['symbol']: x for x in data['result']}
                        for sym in crypto_symbols:
                            t = tickers.get(sym)
                            price = t.get('close') or t.get('mark_price') or t.get('spot_price') if t else None
                            result[sym] = {
                                'ltp': float(price) if price else None,
                                'source': 'delta_exchange'
                            }
            except Exception as e:
                logger.warning("Delta LTP fetch failed: %s", e)
                for sym in crypto_symbols:
                    result[sym] = {'ltp': None, 'source': 'error'}

        if other_symbols:
            if T.kite_available() and _kite_ltp_permitted:
                try:
                    kite   = T.get_kite()
                    quotes = kite.ltp(other_symbols)
                    for sym in other_symbols:
                        q = quotes.get(sym, {})
                        result[sym] = {
                            'ltp':    q.get('last_price'),
                            'source': 'kite',
                        }
                except Exception as e:
                    logger.warning("Kite LTP failed: %s", e)
                    if "permission" in str(e).lower() or "privilege" in str(e).lower():
                        _kite_ltp_permitted = False
                        logger.info("Permanently disabling Kite LTP calls due to insufficient permissions. Bypassing to Yahoo Finance fallback directly.")
            
            # Yahoo fallback for remaining other symbols
            from src.trading_system.api_manager import api_manager
            from src.trading_system.tools import _kite_to_yf
            
            # Find symbols that need to be fetched/subscribed
            missing_yf_symbols = []
            for sym in other_symbols:
                if sym not in result:
                    yf_sym = _kite_to_yf(sym)
                    cached = api_manager.yahoo_live_prices.get(yf_sym)
                    if cached and (time.time() - cached['updated_at'] < 60):
                        result[sym] = {'ltp': cached['price'], 'source': 'yahoo_ws'}
                        api_manager.track_call('yfinance')
                    else:
                        missing_yf_symbols.append(sym)
                        
            # Fetch missing symbols
            if missing_yf_symbols:
                for sym in missing_yf_symbols:
                    try:
                        q = T._yf_quote(sym)
                        ltp = q.get('last_price')
                        result[sym] = {'ltp': ltp, 'source': 'yahoo'}
                        api_manager.track_call('yfinance')
                        
                        # Cache and dynamically subscribe to it
                        yf_sym = _kite_to_yf(sym)
                        if ltp is not None:
                            api_manager.yahoo_live_prices[yf_sym] = {
                                'price': ltp,
                                'change': q.get('change', 0),
                                'change_percent': q.get('change_percent', 0),
                                'updated_at': time.time()
                            }
                            api_manager.start_yahoo_websocket([yf_sym])
                    except:
                        result[sym] = {'ltp': None, 'source': 'error'}
                        
        return jsonify({'quotes': result, 'source': 'mixed'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/market_status', methods=['GET'])
def api_market_status():
    """Return whether market is currently open based on the active mode."""
    mode = request.args.get('mode', 'equity').lower()
    from datetime import datetime, time as dtime
    import zoneinfo
    tz = zoneinfo.ZoneInfo('Asia/Kolkata')
    now = datetime.now(tz)
    weekday = now.weekday()          # Mon=0, Fri=4, Sat=5, Sun=6
    t = now.time()
    
    if mode == 'crypto':
        return jsonify({
            'is_open': True,
            'current_time_ist': now.strftime('%H:%M:%S'),
            'day': now.strftime('%A'),
            'next_open': None,
            'message': 'Crypto markets are open 24/7'
        })
    elif mode == 'forex':
        # Forex closed on Saturday and Sunday (Asia/Kolkata zone timezone check)
        is_open = weekday < 5
        return jsonify({
            'is_open': is_open,
            'current_time_ist': now.strftime('%H:%M:%S'),
            'day': now.strftime('%A'),
            'next_open': 'Monday 03:30 IST' if not is_open else None,
            'message': 'Forex markets are open 24/5'
        })
    else:
        market_open  = dtime(9, 15)
        market_close = dtime(15, 30)
        is_open = (weekday < 5) and (market_open <= t <= market_close)
        return jsonify({
            'is_open': is_open,
            'current_time_ist': now.strftime('%H:%M:%S'),
            'day': now.strftime('%A'),
            'next_open': '09:15 IST' if not is_open else None,
            'message': 'NSE market is open 09:15 - 15:30 IST'
        })


@app.route('/api/public_ip', methods=['GET'])
def api_public_ip():
    """Return the server's current public IP (needed for Kite IP whitelisting)."""
    try:
        import urllib.request
        ip = urllib.request.urlopen('https://api.ipify.org', timeout=5).read().decode()
        return jsonify({
            'ip': ip,
            'kite_console': 'https://developers.kite.trade/apps',
            'message': f'Add {ip} to Allowed IPs in your Kite app settings.',
        })
    except Exception as e:
        return jsonify({'ip': 'unknown', 'error': str(e)}), 500



@app.route('/api/connect_llm', methods=['POST'])
def api_connect_llm():
    """Connect the LLM and fire a real test ping to verify credentials + model."""
    global _orchestrator
    try:
        data     = request.json
        provider = data.get('provider')
        model    = data.get('model', '').strip()
        api_key  = data.get('api_key', '').strip()
        base_url = (data.get('base_url') or 'https://openrouter.ai/api/v1').strip()

        if not model:
            return jsonify({'error': '❌ Model name is required', 'connected': False}), 400
        if not api_key:
            return jsonify({'error': '❌ API key is required', 'connected': False}), 400

        # ── Step 1: Check key validity via lightweight GET request (saves tokens) ──
        logger.info("Testing connection to %s to verify API key", base_url)
        import requests
        headers = {"Authorization": f"Bearer {api_key}"}
        clean_url = base_url.rstrip('/')
        
        if "generativelanguage.googleapis.com" in base_url:
            test_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            response = requests.get(test_url, timeout=10)
        elif "openrouter.ai" in base_url:
            test_url = "https://openrouter.ai/api/v1/auth/key"
            response = requests.get(test_url, headers=headers, timeout=10)
        else:
            test_url = f"{clean_url}/models"
            response = requests.get(test_url, headers=headers, timeout=10)

        if response.status_code != 200:
            err_msg = f"API returned status code {response.status_code}"
            try:
                err_data = response.json()
                if "error" in err_data:
                    err_msg = err_data["error"].get("message", err_msg)
            except Exception:
                pass
            raise Exception(f"Authentication failed: {err_msg}")

        logger.info("LLM connection test success (status 200). Verified key without consuming tokens.")
        test_response_content = "Key verified successfully (no tokens consumed)"

        # ── Step 2: Build the full orchestrator ──────────────────────────────
        _orchestrator = build_orchestrator(
            model       = model,
            api_key     = api_key,
            base_url    = base_url,
            temperature = 0.0,
            max_tokens  = 4096,
            approval_fn = human_approval_popup,
        )

        kite_status = '🪁 Kite Connected' if T.kite_available() else '📡 Yahoo Finance (no Kite)'
        return jsonify({
            'message'  : f'✅ LLM Connected\nModel: {model}\nData: {kite_status}\nPing: {test_response_content}',
            'connected': True,
            'ping'     : test_response_content,
        })

    except Exception as e:
        err = str(e)
        # Surface the exact OpenRouter error message
        if 'invalid model' in err.lower() or '404' in err or '400' in err:
            msg = f'❌ Model not found or invalid: {model}\nCheck the model ID at openrouter.ai/models'
        elif '401' in err or 'unauthorized' in err.lower() or 'invalid api' in err.lower():
            msg = '❌ Invalid API key — check your OpenRouter key'
        elif '429' in err:
            msg = '❌ Rate limit exceeded — try again in a moment'
        else:
            msg = f'❌ {err}'
        logger.error("LLM connect failed: %s", err)
        return jsonify({'error': msg, 'connected': False}), 400


@app.route('/api/connect_kite', methods=['POST'])
def api_connect_kite():
    """Connect only Kite broker."""
    try:
        data       = request.json
        kite_key   = data.get('kite_key', '').strip()
        kite_token = data.get('kite_token', '').strip()
        if not kite_key or not kite_token:
            return jsonify({'error': 'Missing API key or access token', 'connected': False}), 400
        T.init_kite(kite_key, kite_token)
        # Verify by fetching profile
        from kiteconnect import KiteConnect
        kite = T.get_kite()
        profile = kite.profile()
        return jsonify({
            'message': f"✅ Kite Connected — {profile.get('user_name', 'User')}",
            'connected': True,
            'user': profile.get('user_name', ''),
        })
    except Exception as e:
        err_str = str(e)
        if 'No IPs configured' in err_str or 'static-ip' in err_str or 'allowed IPs' in err_str:
            try:
                import urllib.request
                my_ip = urllib.request.urlopen('https://api.ipify.org', timeout=3).read().decode()
            except Exception:
                my_ip = 'unknown'
            return jsonify({
                'error': (
                    f'\u26a0\ufe0f IP not whitelisted in Kite developer console.\n'
                    f'Your current public IP is: {my_ip}\n'
                    f'Add it at: https://developers.kite.trade/apps'
                ),
                'error_code': 'KITE_IP_BLOCKED',
                'your_ip': my_ip,
                'connected': False,
            }), 403
        return jsonify({'error': f'\u274c {err_str}', 'connected': False}), 400


_delta_api_key = None
_delta_api_secret = None

def delta_request(method, path, params=None, json_data=None, api_key=None, api_secret=None):
    """Make an authenticated request to Delta Exchange API using HMAC-SHA256 signature."""
    import hmac
    import hashlib
    import time
    
    url = f"https://api.delta.exchange{path}"
    timestamp = str(int(time.time() * 1000))
    query_string = ""
    if params:
        from urllib.parse import urlencode
        query_string = urlencode(sorted(params.items()))
    
    body = ""
    if json_data:
        import json
        body = json.dumps(json_data)
        
    payload = method.upper() + timestamp + path + ("?" + query_string if query_string else "") + body
    signature = hmac.new(api_secret.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
    
    headers = {
        "api-key": api_key,
        "signature": signature,
        "timestamp": timestamp,
        "Content-Type": "application/json"
    }
    
    if method.upper() == 'GET':
        return requests.get(url, params=params, headers=headers, timeout=5)
    elif method.upper() == 'POST':
        return requests.post(url, json=json_data, headers=headers, timeout=5)
    return None


@app.route('/api/connect_delta', methods=['POST'])
def api_connect_delta():
    """Verify and connect Delta Exchange broker."""
    global _delta_api_key, _delta_api_secret
    try:
        data       = request.json or {}
        api_key    = data.get('api_key', '').strip()
        api_secret = data.get('api_secret', '').strip()
        if not api_key or not api_secret:
            return jsonify({'error': 'Missing API key or API secret', 'connected': False}), 400
            
        res = delta_request('GET', '/v2/profile', api_key=api_key, api_secret=api_secret)
        if res and res.status_code == 200 and res.json().get('success') == True:
            _delta_api_key = api_key
            _delta_api_secret = api_secret
            from src.trading_system.api_manager import api_manager
            api_manager.set_delta_credentials(api_key, api_secret)
            email = res.json().get('result', {}).get('email') or 'Delta User'
            return jsonify({
                'message': f"✅ Delta Connected — {email}",
                'connected': True,
                'user': email,
            })
        else:
            err_msg = 'Request failed'
            if res:
                try:
                    err_msg = res.json().get('error', {}).get('message') or res.text
                except Exception:
                    err_msg = res.text
            return jsonify({'error': f'Connection failed: {err_msg}', 'connected': False}), 400
    except Exception as e:
        return jsonify({'error': f'❌ {str(e)}', 'connected': False}), 500

_finnhub_api_key = None

@app.route('/api/connect_finnhub', methods=['POST'])
def api_connect_finnhub():
    global _finnhub_api_key
    try:
        data = request.json or {}
        api_key = data.get('api_key', '').strip()
        if not api_key:
            return jsonify({'error': 'Missing API key', 'connected': False}), 400
        
        # Test connection
        r = requests.get(f"https://finnhub.io/api/v1/stock/profile2?symbol=AAPL&token={api_key}")
        if r.status_code == 200 and 'ticker' in r.json():
            _finnhub_api_key = api_key
            from src.trading_system.api_manager import api_manager
            api_manager.set_finnhub_key(api_key)
            return jsonify({'message': '✅ Finnhub Connected', 'connected': True})
        else:
            return jsonify({'error': 'Invalid API Key', 'connected': False}), 400
    except Exception as e:
        return jsonify({'error': f'❌ {str(e)}', 'connected': False}), 500

@app.route('/api/api_status', methods=['GET'])
def api_status():
    from src.trading_system.api_manager import api_manager
    # Keep key state in sync just in case
    api_manager.finnhub_api_key = _finnhub_api_key
    api_manager.delta_api_key = _delta_api_key
    api_manager.delta_api_secret = _delta_api_secret
    return jsonify(api_manager.get_api_status())

# --- Classified Tickers Endpoint ---
@app.route('/api/tickers', methods=['GET'])
def api_tickers():
    try:
        mode = request.args.get('mode', 'equity').lower()
        query = request.args.get('q', '').strip()
        
        import psycopg2
        import psycopg2.extras
        from trading_system.memory import DB_CONFIG
        
        db_tickers = []
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                if query:
                    exact_q = query
                    prefix_q = f"{query}%"
                    contains_q = f"%{query}%"
                    cur.execute(
                        "SELECT symbol, name, sector, status, demerger_details FROM tickers_classification "
                        "WHERE market_mode = %s AND (symbol ILIKE %s OR name ILIKE %s) "
                        "ORDER BY ( "
                        "  CASE "
                        "    WHEN symbol ILIKE %s OR symbol ILIKE %s THEN 1 "
                        "    WHEN symbol ILIKE %s OR symbol ILIKE %s THEN 2 "
                        "    WHEN name ILIKE %s THEN 3 "
                        "    WHEN name ILIKE %s THEN 4 "
                        "    WHEN symbol ILIKE %s THEN 5 "
                        "    ELSE 6 "
                        "  END "
                        ") ASC, LENGTH(symbol) ASC, symbol ASC LIMIT 50",
                        (mode, contains_q, contains_q, exact_q, f"NSE:{exact_q}", prefix_q, f"NSE:{prefix_q}", exact_q, prefix_q, contains_q)
                    )
                else:
                    cur.execute(
                        "SELECT symbol, name, sector, status, demerger_details FROM tickers_classification "
                        "WHERE market_mode = %s "
                        "ORDER BY ( "
                        "  CASE "
                        "    WHEN symbol IN ('NSE:TMCV', 'NSE:TMPV', 'NSE:RELIANCE', 'NSE:TCS', 'NSE:INFY', 'NSE:HDFCBANK', 'NSE:NIFTY 50') THEN 1 "
                        "    WHEN symbol LIKE 'NSE:%%' THEN 2 "
                        "    ELSE 3 "
                        "  END "
                        ") ASC, symbol ASC LIMIT 50",
                        (mode,)
                    )
                db_tickers = [dict(row) for row in cur.fetchall()]
        
        # Fallback to API cache for crypto dynamic search if database doesn't have it
        if mode == 'crypto' and query and not db_tickers:
            products = get_delta_products()
            q_upper = query.upper()
            filtered = [
                p for p in products
                if q_upper in p['symbol'].upper() or q_upper in p['name'].upper()
            ]
            return jsonify({"tickers": filtered[:50]})
            
        return jsonify({"tickers": db_tickers})
    except Exception as e:
        logger.exception("Failed to fetch tickers")
        return jsonify({"error": str(e)}), 500


# --- Place/Execute Trade Endpoint ---
@app.route('/api/place_order', methods=['POST'])
def api_place_order():
    try:
        data = request.json or {}
        symbol = data.get('symbol', '').strip().upper()
        exchange = data.get('exchange', 'NSE').strip().upper()
        transaction = data.get('transaction', 'BUY').strip().upper()
        quantity = int(data.get('quantity', 1))
        order_type = data.get('order_type', 'MARKET').strip().upper()
        price = float(data.get('price', 0.0))
        product = data.get('product', 'CNC').strip().upper()
        variety = data.get('variety', 'regular').strip().lower() # regular, intraday, gtt
        trigger_price = float(data.get('trigger_price', 0.0))
        client_ltp = float(data.get('ltp', 0.0))

        # Check if option details are provided to build the option symbol
        is_option = data.get('is_option', False)
        if is_option:
            option_type = data.get('option_type', 'CE').strip().upper() # CE or PE
            strike_price = data.get('strike_price')
            expiry = data.get('expiry', '') # e.g. "26JUN" or "2026-06-25"
            if strike_price and expiry:
                clean_expiry = str(expiry).replace('-', '').upper()
                symbol = f"{symbol}{clean_expiry}{strike_price}{option_type}"
                product = "NRML"

        # Check if symbol requires mock order routing or if Kite is offline
        is_crypto = symbol.upper().strip().endswith("USDT") or symbol.upper().strip().startswith("P-") or symbol.upper().strip().startswith("C-") or symbol.upper().strip().startswith("F-")
        is_forex = symbol.upper().strip().endswith("=X")
        
        if is_crypto or is_forex or not T.kite_available():
            mock_order_id = f"MOCK_ORD_{__import__('random').randint(100000, 999999)}"
            trade = {
                "symbol": symbol,
                "exchange": "CRYPTO" if is_crypto else ("FOREX" if is_forex else exchange),
                "transaction": transaction,
                "quantity": quantity,
                "order_type": order_type,
                "price": price,
                "product": product,
                "variety": variety,
                "trigger_price": trigger_price,
                "order_id": mock_order_id,
                "status": "placed",
                "mock": True
            }
            memory.log_trade(trade)
            m_type = "Crypto" if is_crypto else ("Forex" if is_forex else "Mock")
            return jsonify({"status": "SUCCESS", "order_id": mock_order_id, "message": f"✅ Mock Order Placed ({m_type}): {transaction} {quantity} shares of {symbol}."})

        kite = T.get_kite()
        if variety == 'gtt':
            # Use frontend LTP to avoid Kite Connect permission errors on live quote fetching
            actual_ltp = client_ltp
            if actual_ltp <= 0:
                try:
                    instrument = f"{exchange}:{symbol}"
                    ltp_resp = kite.ltp(instrument)
                    actual_ltp = ltp_resp[instrument]['last_price']
                except Exception as e:
                    return jsonify({"error": f"Failed to fetch LTP for GTT and no client LTP provided: {str(e)}"}), 400
                
            if trigger_price == actual_ltp:
                return jsonify({"error": f"Trigger cannot be created with trigger price ({trigger_price}) equal to the last price ({actual_ltp}). A GTT must cross the current price to trigger (either above or below)."}), 400

            from kiteconnect import KiteConnect
            trigger_type = kite.GTT_TYPE_SINGLE
            trigger_values = [trigger_price]
            gtt_order = {
                "exchange": exchange,
                "tradingsymbol": symbol,
                "transaction_type": transaction,
                "quantity": quantity,
                "order_type": "LIMIT" if price > 0 else "MARKET",
                "product": product,
                "price": price if price > 0 else trigger_price
            }
            gtt_id = kite.place_gtt(
                trigger_type=trigger_type,
                tradingsymbol=symbol,
                exchange=exchange,
                trigger_values=trigger_values,
                last_price=actual_ltp,
                orders=[gtt_order]
            )
            trade = {**gtt_order, "gtt_id": gtt_id, "variety": "gtt", "status": "placed"}
            memory.log_trade(trade)
            return jsonify({"status": "SUCCESS", "order_id": gtt_id, "message": f"✅ GTT Order Placed. ID: {gtt_id}"})
        else:
            if variety == 'intraday':
                product = 'MIS'
            params = dict(
                tradingsymbol=symbol,
                exchange=exchange,
                transaction_type=transaction,
                quantity=quantity,
                order_type=order_type,
                product=product
            )
            if order_type == "LIMIT":
                params["price"] = price
            
            # Determine if we should use AMO (After Market Order) based on time
            from datetime import datetime, time as dtime
            import zoneinfo
            tz = zoneinfo.ZoneInfo('Asia/Kolkata')
            now = datetime.now(tz)
            weekday = now.weekday()
            t = now.time()
            is_open = (weekday < 5) and (dtime(9, 15) <= t <= dtime(15, 30))
            
            kite_variety = kite.VARIETY_AMO if not is_open else kite.VARIETY_REGULAR
            
            order_id = kite.place_order(variety=kite_variety, **params)
            trade = {**params, "order_id": order_id, "variety": variety, "kite_variety": kite_variety, "status": "placed"}
            memory.log_trade(trade)
            amo_msg = " (AMO)" if kite_variety == kite.VARIETY_AMO else ""
            return jsonify({"status": "SUCCESS", "order_id": order_id, "message": f"✅ Order Placed{amo_msg}. ID: {order_id}"})

    except Exception as e:
        err_str = str(e)
        logger.exception("Order execution failed")
        # ── Surface Kite-specific errors with actionable messages ──────────────────
        if 'No IPs configured' in err_str or 'static-ip' in err_str or 'allowed IPs' in err_str:
            try:
                import urllib.request
                my_ip = urllib.request.urlopen('https://api.ipify.org', timeout=3).read().decode()
            except Exception:
                my_ip = 'unknown'
            return jsonify({
                'error': (
                    f'\u26a0\ufe0f Kite API requires IP whitelisting.\n'
                    f'Your current public IP: {my_ip}\n\n'
                    f'Fix: Go to https://developers.kite.trade/apps \u2192 select your app \u2192 '
                    f'add {my_ip} to \'Allowed IP addresses\' \u2192 Save.\n\n'
                    f'Note: If your IP is dynamic (changes daily), contact your ISP for a static IP or use a VPN with fixed exit IP.'
                ),
                'error_code': 'KITE_IP_BLOCKED',
                'your_ip': my_ip,
            }), 403
        elif 'InputException' in err_str or 'Invalid' in err_str:
            return jsonify({'error': f'\u274c Invalid order parameters: {err_str}'}), 400
        elif 'TokenException' in err_str or 'Unauthorized' in err_str:
            return jsonify({'error': '\u274c Kite session expired. Please reconnect in the Config tab.'}), 401
        elif 'NetworkException' in err_str or 'ConnectionError' in err_str:
            return jsonify({'error': '\u274c Network error connecting to Kite. Check your internet connection.'}), 503
        else:
            return jsonify({'error': f'\u274c Order failed: {err_str}'}), 400


# --- Model Prediction Endpoint ---
@app.route('/api/predict_model', methods=['POST'])
def api_predict_model():
    try:
        body = request.json or {}
        symbol = body.get('symbol', 'NSE:INFY')
        interval = body.get('interval', 'day')
        days = int(body.get('days', 120))

        import pandas as pd
        df = None
        if T.kite_available():
            try:
                kite = T.get_kite()
                ltp_data = kite.ltp(symbol)
                token = ltp_data[symbol]['instrument_token']
                from_dt = __import__('datetime').datetime.now() - __import__('datetime').timedelta(days=days)
                to_dt = __import__('datetime').datetime.now()
                raw_candles = kite.historical_data(token, from_dt, to_dt, interval)
                df = pd.DataFrame(raw_candles)
                if not df.empty:
                    df = df.rename(columns={'date':'time','open':'open','high':'high',
                                            'low':'low','close':'close','volume':'volume'})
            except Exception as e:
                logger.warning("Kite predict OHLCV failed, fallback to Yahoo: %s", e)
                df = None

        if df is None or df.empty:
            import yfinance as yf
            from src.trading_system.tools import _kite_to_yf, _yf_interval, _days_to_period
            yf_sym = _kite_to_yf(symbol)
            yf_int = _yf_interval(interval)
            period = _days_to_period(days, yf_int)
            ticker = yf.Ticker(yf_sym)
            raw_df = ticker.history(period=period, interval=yf_int, auto_adjust=True)
            if raw_df.empty:
                return jsonify({'error': f'No data for {symbol}'}), 404
            raw_df = raw_df.reset_index()
            raw_df.columns = [c.lower() for c in raw_df.columns]
            time_col = 'datetime' if 'datetime' in raw_df.columns else 'date'
            df = raw_df.rename(columns={time_col: 'time'})[['time','open','high','low','close','volume']]

        df = df.dropna(subset=['close']).copy()
        df = df.sort_values('time').reset_index(drop=True)
        
        candles = []
        for _, r in df.iterrows():
            candles.append({
                'time': r['time'],
                'open': float(r['open']),
                'high': float(r['high']),
                'low': float(r['low']),
                'close': float(r['close']),
                'volume': int(r['volume'] or 0)
            })

        from trading_system.model_predictor import predict_signals
        predictions = predict_signals(candles)
        
        return jsonify({
            "symbol": symbol,
            "predictions": predictions
        })
    except Exception as e:
        logger.exception("Model prediction endpoint failed")
        return jsonify({"error": str(e)}), 500


def run_db_migrations():
    """Run database migrations on startup to support mode categorization."""
    try:
        import psycopg2
        from trading_system.memory import DB_CONFIG
        
        forex_cnt = 0
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute("ALTER TABLE trade_logs ADD COLUMN IF NOT EXISTS market_mode VARCHAR(50) DEFAULT 'equity';")
                cur.execute("ALTER TABLE tickers_classification ADD COLUMN IF NOT EXISTS market_mode VARCHAR(50) DEFAULT 'equity';")
                
                cur.execute("SELECT COUNT(*) FROM tickers_classification WHERE market_mode = 'forex'")
                forex_cnt = cur.fetchone()[0]
        finally:
            conn.close()

        if forex_cnt == 0:
            logger.info("Database empty of forex tickers. Seeding database...")
            from init_db import init_db
            init_db()
        else:
            logger.info("Database migration verification passed successfully.")
    except Exception as e:
        logger.error("Failed to run database migrations: %s", e)


# Run database migrations on module import (ensures test/dev environments are migrated)
run_db_migrations()

def init_yahoo_websocket_tickers():
    """Start the Yahoo Finance live WebSocket ticker with default watchlist symbols."""
    try:
        from src.trading_system.api_manager import api_manager
        from src.trading_system.tools import _kite_to_yf
        
        watchlist_equities = [
            'NSE:NIFTY 50', 'NSE:NIFTY BANK', 'NSE:RELIANCE', 'NSE:TCS',
            'NSE:INFY', 'NSE:HDFCBANK', 'NSE:ICICIBANK', 'NSE:WIPRO',
            'NSE:BAJFINANCE', 'NSE:TATAMOTORS', 'NSE:TMCV', 'NSE:TMPV',
            'NSE:ADANIENT', 'NSE:SBIN', 'NSE:ITC', 'NSE:AXISBANK',
        ]
        watchlist_forex = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X']
        
        yf_symbols = [_kite_to_yf(s) for s in watchlist_equities] + watchlist_forex
        api_manager.start_yahoo_websocket(yf_symbols)
        logger.info(f"Initialized Yahoo Finance WebSocket with {len(yf_symbols)} symbols")
    except Exception as e:
        logger.error(f"Failed to initialize Yahoo Finance WebSocket ticker: {e}")

# Run Yahoo Finance WebSocket ticker initialization on import
init_yahoo_websocket_tickers()

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

