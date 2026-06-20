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

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

from flask_cors import CORS

app = Flask(__name__)
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

@app.route('/api/quotes', methods=['POST'])
def api_quotes():
    """Fetch live/last quotes for multiple symbols via get_quote tool."""
    try:
        symbols = request.json.get('symbols', 'NSE:INFY,NSE:RELIANCE,NSE:TCS')
        raw = T.get_quote.invoke({'symbols': symbols})
        return jsonify(json.loads(raw))
    except Exception as e:
        return jsonify({'error': str(e)}), 400


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


@app.route('/api/connect_llm', methods=['POST'])
def api_connect_llm():
    """Connect only the LLM/orchestrator."""
    global _orchestrator
    try:
        data     = request.json
        provider = data.get('provider')
        model    = data.get('model', '').strip()
        api_key  = data.get('api_key', '').strip()
        base_url = data.get('base_url', '').strip()
        _orchestrator = build_orchestrator(
            model=model,
            api_key=api_key,
            base_url=base_url or 'https://openrouter.ai/api/v1',
            temperature=0.0,
            max_tokens=4096,
            approval_fn=human_approval_popup,
        )
        return jsonify({'message': f'✅ LLM Connected — Model: {model}', 'connected': True})
    except Exception as e:
        logger.exception('LLM connect failed')
        return jsonify({'error': f'❌ {e}', 'connected': False}), 400


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
        return jsonify({'error': f'❌ {e}', 'connected': False}), 400


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
