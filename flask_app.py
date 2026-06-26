import os
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

# --- Static File Serving for Charts ---

@app.route('/data/visualizations/<path:filename>')
def serve_visualizations(filename):
    return send_from_directory(os.path.join(app.root_path, 'data', 'visualizations'), filename)


@app.route('/api/watchlist', methods=['GET'])
def api_watchlist():
    """Return watchlist with live LTP from Kite (if available).
    Symbol metadata (name, sector, demerger_details) is sourced from DB.
    Falls back to hardcoded list if DB is unavailable.
    """
    # ── Default symbols (always included) ─────────────────────────────────────
    default_symbols = [
        'NSE:NIFTY 50', 'NSE:NIFTY BANK', 'NSE:RELIANCE', 'NSE:TCS',
        'NSE:INFY', 'NSE:HDFCBANK', 'NSE:ICICIBANK', 'NSE:WIPRO',
        'NSE:BAJFINANCE', 'NSE:TATAMOTORS', 'NSE:TMCV', 'NSE:TMPV',
        'NSE:ADANIENT', 'NSE:SBIN', 'NSE:ITC', 'NSE:AXISBANK',
    ]

    # ── Pull metadata from DB ──────────────────────────────────────────────────
    db_meta: dict = {}          # symbol → {name, sector, demerger_details}
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

    # Merge DB symbols into the symbol list (add any extra DB symbols)
    db_symbols = list(db_meta.keys())
    all_symbols = list(dict.fromkeys(default_symbols + db_symbols))  # deduplicate, preserve order

    result = []
    try:
        if T.kite_available():
            # Kite only accepts batches ≤ 500; chunk if needed
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
            for sym in all_symbols:
                meta = db_meta.get(sym, {})
                result.append({
                    'symbol':           sym,
                    'name':             meta.get('name') or sym.split(':')[-1],
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
        orch = get_orchestrator()
        response = orch.run(msg)
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
        result = get_orchestrator().get_dashboard()
        html_output = markdown.markdown(result, extensions=['fenced_code', 'tables'])
        chart_url = "/data/visualizations/portfolio_dashboard.html"
        
        return jsonify({
            "output": html_output,
            "chart_url": chart_url
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/portfolio_stats', methods=['GET'])
def api_portfolio_stats():
    """Return aggregate portfolio stats: total invested, current value, total P&L."""
    try:
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
    try:
        body     = request.json or {}
        symbol   = body.get('symbol',   'NSE:INFY')
        interval = body.get('interval', 'day')
        days     = int(body.get('days', 180))

        # ── Fetch from Kite if available ────────────────────────────────
        df = None
        if T.kite_available():
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
            except Exception as e:
                logger.warning("Kite OHLCV failed, falling back: %s", e)
                df = None

        # ── Yahoo Finance fallback ──────────────────────────────────────
        if df is None or df.empty:
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
        logger.exception("OHLCV error")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ltp', methods=['POST'])
def api_ltp():
    """Return last-traded prices for a list of symbols (Kite or Yahoo fallback)."""
    try:
        symbols = request.json.get('symbols', [])
        result  = {}
        if T.kite_available():
            try:
                kite   = T.get_kite()
                quotes = kite.ltp(symbols)
                for sym in symbols:
                    q = quotes.get(sym, {})
                    result[sym] = {
                        'ltp':    q.get('last_price'),
                        'source': 'kite',
                    }
                return jsonify({'quotes': result, 'source': 'kite'})
            except Exception as e:
                logger.warning("Kite LTP failed: %s", e)

        # Yahoo fallback
        for sym in symbols:
            try:
                q = T._yf_quote(sym)
                result[sym] = {'ltp': q.get('last_price'), 'source': 'yahoo'}
            except:
                result[sym] = {'ltp': None, 'source': 'error'}
        return jsonify({'quotes': result, 'source': 'yahoo'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/market_status', methods=['GET'])
def api_market_status():
    """Return whether NSE market is currently open."""
    from datetime import datetime, time as dtime
    import zoneinfo
    tz = zoneinfo.ZoneInfo('Asia/Kolkata')
    now = datetime.now(tz)
    weekday = now.weekday()          # Mon=0, Fri=4, Sat=5, Sun=6
    t = now.time()
    market_open  = dtime(9, 15)
    market_close = dtime(15, 30)
    is_open = (weekday < 5) and (market_open <= t <= market_close)
    return jsonify({
        'is_open': is_open,
        'current_time_ist': now.strftime('%H:%M:%S'),
        'day': now.strftime('%A'),
        'next_open': '09:15 IST' if not is_open else None,
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

        # ── Step 1: Build the LangChain LLM directly for a cheap test ping ──
        logger.info("Testing connection to %s with model=%s", base_url, model)
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        test_llm = ChatOpenAI(
            model        = model,
            api_key      = api_key,
            base_url     = base_url,
            temperature  = 0.0,
            max_tokens   = 8,          # tiny — just verify the round-trip
        )
        # Real API call — will raise if key is wrong, model not found, quota exceeded
        test_response = test_llm.invoke([HumanMessage(content="Hi")])
        logger.info("OpenRouter ping success. Response: %s", test_response.content[:60])

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
            'message'  : f'✅ LLM Connected\nModel: {model}\nData: {kite_status}\nPing: {test_response.content[:40]}',
            'connected': True,
            'ping'     : test_response.content,
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


# --- Classified Tickers Endpoint ---
@app.route('/api/tickers', methods=['GET'])
def api_tickers():
    try:
        import psycopg2
        import psycopg2.extras
        from trading_system.memory import DB_CONFIG
        
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("SELECT symbol, name, sector, status, demerger_details FROM tickers_classification ORDER BY symbol ASC")
                rows = cur.fetchall()
                tickers = [dict(row) for row in rows]
                return jsonify({"tickers": tickers})
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

        # Check if Kite is available
        if not T.kite_available():
            mock_order_id = f"MOCK_ORD_{__import__('random').randint(100000, 999999)}"
            trade = {
                "symbol": symbol,
                "exchange": exchange,
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
            return jsonify({"status": "SUCCESS", "order_id": mock_order_id, "message": f"✅ Mock Order Placed (Kite Offline): {transaction} {quantity} shares of {symbol}."})

        kite = T.get_kite()
        if variety == 'gtt':
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
                last_price=trigger_price,
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


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

