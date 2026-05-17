# dashboard.py
import os
import json
import logging
from pathlib import Path
from typing import Optional

import gradio as gr
from dotenv import load_dotenv

from llm_service import LLMService, LLMConfig
from main_agent import build_orchestrator, OrchestratorAgent
from memory import MemoryManager
import tools as T

load_dotenv()
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)
memory = MemoryManager()

# ---------------------------------------------------------------------------
# Global orchestrator
# ---------------------------------------------------------------------------
_orchestrator: Optional[OrchestratorAgent] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_orchestrator() -> OrchestratorAgent:
    """Return active orchestrator or raise Gradio error."""
    if _orchestrator is None:
        raise gr.Error("Not connected — go to ⚙️ Config tab and click Connect.")
    return _orchestrator


def human_approval_popup(order: dict) -> bool:
    """
    Default approval gate — always deny.
    Real approval is handled via the Execute Trade tab checkbox.
    """
    return False


def load_html(path: str) -> str:
    """Read HTML file contents or return fallback."""
    if not path or not path.strip():
        return "<p>No chart path provided.</p>"

    p = Path(path)
    if not p.exists():
        return f"<p>Chart not found: {path}</p>"
    if not p.is_file():                            # ✅ guard against dirs
        return f"<p>Path is not a file: {path}</p>"

    try:
        return p.read_text()
    except Exception as e:
        return f"<p>Error reading chart: {e}</p>"

# ---------------------------------------------------------------------------
# Provider map — display name → LangChain internal provider
# ---------------------------------------------------------------------------
PROVIDER_MAP = {
    "openrouter"   : "openai",        # openrouter is openai-API-compatible
    "openai"       : "openai",
    "anthropic"    : "anthropic",
    "google-gemini": "google-genai",
}

PROVIDER_URLS = {
    "openrouter"   : "https://openrouter.ai/api/v1",
    "openai"       : "",
    "anthropic"    : "",
    "google-gemini": "",
}


# ---------------------------------------------------------------------------
# Tab handlers
# ---------------------------------------------------------------------------

def update_base_url(provider: str) -> str:
    """Auto-fill base URL when provider dropdown changes."""
    return PROVIDER_URLS.get(provider, "")


def connect(
    provider         : str,
    model            : str,
    api_key          : str,
    base_url         : str,
    kite_api_key     : str,
    kite_access_token: str,
    temperature      : float,
    max_tokens       : int,
) -> str:
    """
    Build LLMConfig → LLMService → OrchestratorAgent.
    Provider dropdown controls display name and routing.
    All sub-agents share the same LLMService instance.
    """
    global _orchestrator
    try:
        # ── 1. init Kite ─────────────────────────────────────────────────
        if kite_api_key.strip() and kite_access_token.strip():
            T.init_kite(
                api_key      = kite_api_key.strip(),
                access_token = kite_access_token.strip(),
            )

        # ── 2. resolve provider ──────────────────────────────────────────
        langchain_provider = PROVIDER_MAP.get(provider.lower(), "openai")
        resolved_url       = (
            base_url.strip()
            or PROVIDER_URLS.get(provider.lower(), "")
            or None
        )

        # ── 3. build LLMConfig via llm_service.py ────────────────────────
        config = LLMConfig(
            model       = model.strip(),
            api_key     = api_key.strip(),
            provider    = langchain_provider,
            temperature = float(temperature),
            max_tokens  = int(max_tokens),
            base_url    = resolved_url,
        )

        # ── 4. build orchestrator (all sub-agents share this service) ────
        _orchestrator = build_orchestrator(
            model       = model.strip(),
            api_key     = api_key.strip(),
            base_url    = resolved_url or "https://openrouter.ai/api/v1",
            temperature = float(temperature),
            max_tokens  = int(max_tokens),
            approval_fn = human_approval_popup,
        )

        return (
            f"✅ Connected\n"
            f"   provider : {provider}\n"
            f"   model    : {model.strip()}\n"
            f"   endpoint : {resolved_url or 'provider default'}\n"
            f"   kite     : {'✅' if kite_api_key.strip() else '⚠️  not set'}"
        )
    except Exception as e:
        logger.exception("connect() failed")
        return f"❌ Error: {e}"


def chat(message: str, history: list) -> tuple:
    """Route free-form message through the orchestrator."""
    try:
        response = get_orchestrator().run(message)
        history.append((message, response))
        return history, history, ""
    except Exception as e:
        history.append((message, f"❌ Error: {e}"))
        return history, history, ""


def analyse(symbol: str) -> tuple:
    """Run full technical analysis + candlestick chart."""
    try:
        results    = get_orchestrator().analyse_symbol(symbol.strip().upper())
        output     = "\n\n".join(
            f"**{k}**\n{v}" for k, v in results.items()
        )
        chart_name = f"{symbol.strip().upper().replace(':', '_')}_60d"
        chart_html = load_html(f"data/visualizations/{chart_name}.html")
        return output, chart_html
    except Exception as e:
        return f"❌ {e}", ""


def run_backtest(
    symbol     : str,
    strategy   : str,
    params_json: str,
    days       : int,
) -> tuple:
    """Backtest a strategy and return stats + equity curve."""
    try:
        result = get_orchestrator().strategy.run(
            f"Backtest {strategy} on {symbol} for {days} days "
            f"with params {params_json}"
        )
        strat_name = f"{symbol}_{strategy}_{days}d"
        chart_resp = get_orchestrator().viz.run(
            f"Create backtest chart for {strat_name}"
        )
        chart_info = json.loads(chart_resp) if chart_resp.startswith("{") else {}
        chart_html = load_html(chart_info.get("chart_path", ""))
        return result, chart_html
    except Exception as e:
        return f"❌ {e}", ""


def show_portfolio() -> tuple:
    """Refresh portfolio dashboard."""
    try:
        result     = get_orchestrator().get_dashboard()
        info       = json.loads(result) if result.startswith("{") else {}
        chart_html = load_html(info.get("dashboard_path", ""))
        return result, chart_html
    except Exception as e:
        return f"❌ {e}", ""


def execute_trade_handler(
    symbol        : str,
    exchange      : str,
    action        : str,
    qty           : int,
    order_type    : str,
    price         : float,
    product       : str,
    user_confirmed: bool,
) -> str:
    """Place trade — only proceeds if confirm checkbox is ticked."""
    if not user_confirmed:
        return "⚠️ Tick 'I confirm this trade' to proceed."
    try:
        order = dict(
            symbol      = symbol.strip().upper(),
            exchange    = exchange,
            transaction = action,
            quantity    = int(qty),
            order_type  = order_type,
            price       = float(price) if order_type == "LIMIT" else 0.0,
            product     = product,
        )
        return get_orchestrator().execute_trade(order)
    except Exception as e:
        return f"❌ {e}"


def internet_search_handler(query: str) -> str:
    """Run internet search through orchestrator."""
    try:
        return get_orchestrator().run(f"Search the internet for: {query}")
    except Exception as e:
        return f"❌ {e}"


def show_strategies() -> str:
    """List all saved strategies."""
    strats = memory.list_strategies()
    return "\n".join(strats) if strats else "No strategies saved yet."


def show_trade_log(date: str) -> str:
    """Show trade log for a given date."""
    logs = memory.get_trade_log(date.strip() or None)
    return json.dumps(logs, indent=2) if logs else "No trades found."


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

def build_dashboard() -> gr.Blocks:
    with gr.Blocks(
        title   = "🤖 AI Trading System",
        theme   = gr.themes.Soft(),
        css     = ".gradio-container { max-width: 1400px; margin: auto; }",
    ) as demo:

        gr.Markdown(
            "# 🤖 AI Trading System\n"
            "Powered by **LangChain** · **Zerodha Kite** · **Tavily Search**"
        )

        # ── ⚙️ Config ─────────────────────────────────────────────────────
        with gr.Tab("⚙️ Config"):
            gr.Markdown(
                "### LLM + Kite Configuration\n"
                "All agents share **one** LLM connection via `llm_service.py`."
            )
            with gr.Row():
                with gr.Column():
                    # ✅ provider dropdown — controls display + routing
                    provider_in = gr.Dropdown(
                        choices = ["openrouter", "openai", "anthropic", "google-gemini"],
                        value   = "openrouter",
                        label   = "Provider",
                    )
                    model_in = gr.Textbox(
                        label = "Model",
                        value = os.getenv("LLM_MODEL", "google/gemini-2.5-flash"),
                    )
                    api_key_in = gr.Textbox(
                        label = "API Key",
                        value = os.getenv("OPENROUTER_API_KEY", ""),
                        type  = "password",
                    )
                    base_url_in = gr.Textbox(
                        label = "Base URL (auto-filled)",
                        value = "https://openrouter.ai/api/v1",
                    )
                    temperature_in = gr.Slider(
                        0, 1, value=0.0, step=0.1, label="Temperature"
                    )
                    max_tokens_in = gr.Slider(
                        512, 8192, value=4096, step=512, label="Max Tokens"
                    )
                with gr.Column():
                    kite_key_in = gr.Textbox(
                        label = "Kite API Key",
                        value = os.getenv("KITE_API_KEY", ""),
                        type  = "password",
                    )
                    kite_token_in = gr.Textbox(
                        label = "Kite Access Token",
                        value = os.getenv("KITE_ACCESS_TOKEN", ""),
                        type  = "password",
                    )

            # auto-fill base_url on provider change
            provider_in.change(
                fn      = update_base_url,
                inputs  = provider_in,
                outputs = base_url_in,
            )

            connect_btn    = gr.Button("🔌 Connect", variant="primary")
            connect_status = gr.Textbox(
                label       = "Status",
                interactive = False,
                lines       = 6,
            )
            connect_btn.click(
                fn      = connect,
                inputs  = [
                    provider_in,       # ← provider display name
                    model_in,
                    api_key_in,
                    base_url_in,
                    kite_key_in,
                    kite_token_in,
                    temperature_in,
                    max_tokens_in,
                ],
                outputs = connect_status,
            )

        # ── 💬 Chat ───────────────────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            gr.Markdown(
                "Ask anything — the orchestrator routes to the right agent."
            )
            chatbot    = gr.Chatbot(height=520)
            chat_in    = gr.Textbox(
                placeholder = "e.g. Analyse NSE:INFY and suggest a trade",
                label       = "Message",
            )
            chat_state = gr.State([])
            with gr.Row():
                send_btn  = gr.Button("Send",  variant="primary")
                clear_btn = gr.Button("Clear")
            send_btn.click(
                fn      = chat,
                inputs  = [chat_in, chat_state],
                outputs = [chatbot, chat_state, chat_in],
            )
            clear_btn.click(
                fn      = lambda: ([], []),
                outputs = [chatbot, chat_state],
            )

        # ── 🔍 Search ─────────────────────────────────────────────────────
        with gr.Tab("🔍 Search"):
            gr.Markdown("### Internet Search via Tavily")
            search_in  = gr.Textbox(
                label       = "Search Query",
                placeholder = "e.g. INFY Q4 2024 earnings results",
            )
            search_btn = gr.Button("🔍 Search", variant="primary")
            search_out = gr.Markdown()
            search_btn.click(
                fn      = internet_search_handler,
                inputs  = search_in,
                outputs = search_out,
            )

        # ── 📊 Analysis ───────────────────────────────────────────────────
        with gr.Tab("📊 Analysis"):
            gr.Markdown("### Technical Analysis + Chart")
            with gr.Row():
                sym_in      = gr.Textbox(
                    label       = "Symbol",
                    placeholder = "e.g. NSE:INFY",
                )
                analyse_btn = gr.Button("🔍 Analyse", variant="primary")
            analysis_out = gr.Markdown()
            chart_out    = gr.HTML(label="Candlestick Chart")
            analyse_btn.click(
                fn      = analyse,
                inputs  = sym_in,
                outputs = [analysis_out, chart_out],
            )

        # ── 🧪 Backtest ───────────────────────────────────────────────────
        with gr.Tab("🧪 Backtest"):
            gr.Markdown("### Strategy Back-testing")
            with gr.Row():
                bt_symbol   = gr.Textbox(label="Symbol",   value="NSE:INFY")
                bt_strategy = gr.Dropdown(
                    choices = [
                        "sma_crossover",
                        "rsi_mean_reversion",
                        "macd_trend",
                    ],
                    label = "Strategy",
                    value = "sma_crossover",
                )
                bt_days = gr.Slider(
                    30, 1000, value=365, step=30, label="Days"
                )
            bt_params = gr.Textbox(
                label = "Strategy Params (JSON)",
                value = '{"fast": 20, "slow": 50}',
            )
            bt_btn    = gr.Button("▶️ Run Backtest", variant="primary")
            bt_result = gr.Markdown()
            bt_chart  = gr.HTML(label="Equity Curve")
            bt_btn.click(
                fn      = run_backtest,
                inputs  = [bt_symbol, bt_strategy, bt_params, bt_days],
                outputs = [bt_result, bt_chart],
            )

        # ── 💼 Portfolio ──────────────────────────────────────────────────
        with gr.Tab("💼 Portfolio"):
            gr.Markdown("### Live Portfolio Dashboard")
            refresh_btn     = gr.Button("🔄 Refresh", variant="primary")
            portfolio_md    = gr.Markdown()
            portfolio_chart = gr.HTML(label="Portfolio Dashboard")
            refresh_btn.click(
                fn      = show_portfolio,
                outputs = [portfolio_md, portfolio_chart],
            )

        # ── ⚡ Execute Trade ──────────────────────────────────────────────
        with gr.Tab("⚡ Execute Trade"):
            gr.Markdown(
                "### Manual Trade Execution\n"
                "> ⚠️ Human approval required — tick the confirm box."
            )
            with gr.Row():
                with gr.Column():
                    trade_symbol   = gr.Textbox(label="Symbol",   value="INFY")
                    trade_exchange = gr.Dropdown(
                        ["NSE", "BSE"], label="Exchange", value="NSE"
                    )
                    trade_action   = gr.Dropdown(
                        ["BUY", "SELL"], label="Action"
                    )
                    trade_qty      = gr.Number(label="Quantity", value=1)
                with gr.Column():
                    trade_type    = gr.Dropdown(
                        ["MARKET", "LIMIT"], label="Order Type", value="MARKET"
                    )
                    trade_price   = gr.Number(
                        label="Limit Price (LIMIT orders only)", value=0
                    )
                    trade_product = gr.Dropdown(
                        ["CNC", "MIS", "NRML"], label="Product", value="CNC"
                    )
                    trade_confirm = gr.Checkbox(
                        label="✅ I confirm this trade"
                    )
            execute_btn    = gr.Button("🚀 Execute Trade", variant="stop")
            execute_result = gr.Textbox(
                label="Result", interactive=False, lines=4
            )
            execute_btn.click(
                fn      = execute_trade_handler,
                inputs  = [
                    trade_symbol, trade_exchange, trade_action,
                    trade_qty,    trade_type,     trade_price,
                    trade_product, trade_confirm,
                ],
                outputs = execute_result,
            )

        # ── 🧠 Memory ─────────────────────────────────────────────────────
        with gr.Tab("🧠 Memory"):
            gr.Markdown("### Saved Strategies & Trade Log")
            with gr.Row():
                strat_btn = gr.Button("📋 List Strategies")
                strat_out = gr.Textbox(
                    label="Saved Strategies", lines=10, interactive=False
                )
            strat_btn.click(fn=show_strategies, outputs=strat_out)

            gr.Markdown("---")
            with gr.Row():
                log_date = gr.Textbox(
                    label="Date (YYYY-MM-DD, blank = today)"
                )
                log_btn  = gr.Button("📜 Trade Log")
            log_out = gr.Textbox(
                label="Trade Log", lines=15, interactive=False
            )
            log_btn.click(
                fn      = show_trade_log,
                inputs  = log_date,
                outputs = log_out,
            )

    return demo


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_dashboard()
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,
        inbrowser   = True,
    )
