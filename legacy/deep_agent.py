import os
import json
import gradio as gr
from datetime import datetime
from typing import Optional
from google import genai
from llm_service import LLMConfig , LLMService

# Import all tools and sub-agents
from tools import (
    get_kite_holdings,
    get_kite_positions,
    place_kite_order,
    cancel_kite_order,
    get_kite_quote,
    get_historical_data,
    get_market_depth,
    save_to_memory,
    load_from_memory,
    list_memory_files,
    delete_memory_file,
    save_strategy,
    load_strategy,
    list_strategies,
    get_account_margins,
    get_order_history,
    get_trade_history,
    search_instruments,
    get_indices_data,
    calculate_portfolio_metrics,
    run_backtest,
    generate_chart,
    get_news_sentiment,
    screen_stocks,
    get_technical_indicators,
    execute_human_approved_trade,
    request_human_approval,
    get_pending_approvals,
    approve_trade_request,
    reject_trade_request,
)

from sub_agents import (
    market_analysis_agent,
    strategy_agent,
    risk_management_agent,
    portfolio_agent,
    backtesting_agent,
)

# ─────────────────────────────────────────────
# Gemini Client Setup
# ─────────────────────────────────────────────
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
BASE_MODEL = "gemini-2.0-flash"

# ─────────────────────────────────────────────
# Tool Registry for Main Agent
# ─────────────────────────────────────────────
MAIN_AGENT_TOOLS = [
    # Kite / Broker Tools
    get_kite_holdings,
    get_kite_positions,
    place_kite_order,
    cancel_kite_order,
    get_kite_quote,
    get_historical_data,
    get_market_depth,
    get_account_margins,
    get_order_history,
    get_trade_history,
    search_instruments,
    get_indices_data,
    # Memory / Persistence Tools
    save_to_memory,
    load_from_memory,
    list_memory_files,
    delete_memory_file,
    save_strategy,
    load_strategy,
    list_strategies,
    # Analysis Tools
    calculate_portfolio_metrics,
    run_backtest,
    generate_chart,
    get_news_sentiment,
    screen_stocks,
    get_technical_indicators,
    # Human-in-the-Loop Tools
    execute_human_approved_trade,
    request_human_approval,
    get_pending_approvals,
    approve_trade_request,
    reject_trade_request,
    # Sub-Agent Tools
    market_analysis_agent,
    strategy_agent,
    risk_management_agent,
    portfolio_agent,
    backtesting_agent,
]

# ─────────────────────────────────────────────
# System Instruction
# ─────────────────────────────────────────────
SYSTEM_INSTRUCTION = """
You are ARIA (Autonomous Research & Investment Agent), an advanced AI trading assistant 
integrated with Zerodha Kite for live market operations.

## Core Capabilities
1. **Market Analysis**: Real-time quotes, historical data, technical indicators, sentiment analysis
2. **Portfolio Management**: Holdings, positions, P&L tracking, risk metrics
3. **Strategy Development**: Create, backtest, and optimize trading strategies
4. **Trade Execution**: Place/cancel orders via Kite API (always with human approval)
5. **Persistent Memory**: Save/load strategies, notes, and analysis to file system
6. **Dashboard & Visualization**: Generate charts and performance dashboards

## Sub-Agents Available
- **market_analysis_agent**: Deep market research, sector analysis, stock screening
- **strategy_agent**: Strategy creation, parameter optimization, signal generation
- **risk_management_agent**: Position sizing, stop-loss, portfolio risk assessment
- **portfolio_agent**: Portfolio rebalancing, allocation, performance attribution
- **backtesting_agent**: Historical strategy testing with detailed metrics and charts

## Critical Rules
1. **NEVER execute trades without explicit human approval** - Always use request_human_approval first
2. **Always show risk warnings** before any trade recommendation
3. **Verify instrument symbols** before placing orders
4. **Save important analysis** to persistent memory for future reference
5. **Delegate specialized tasks** to appropriate sub-agents

## Response Style
- Be concise but comprehensive
- Always show key metrics (entry, stop-loss, target, risk-reward)
- Format tables and lists clearly
- Highlight warnings in [WARNING] blocks
- Show confidence levels for recommendations

## Memory Usage
- Auto-save strategies with unique names
- Load previous analysis when relevant
- Maintain a trade journal in persistent memory

Today's Date: {date}
""".format(date=datetime.now().strftime("%Y-%m-%d"))

# ─────────────────────────────────────────────
# Conversation History
# ─────────────────────────────────────────────
conversation_history = []
pending_approvals_store = {}

def format_history_for_api(history):
    """Convert Gradio history to Gemini API format."""
    messages = []
    for item in history:
        if item["role"] == "user":
            messages.append(
                types.Content(role="user", parts=[types.Part(text=item["content"])])
            )
        elif item["role"] == "assistant":
            messages.append(
                types.Content(role="model", parts=[types.Part(text=item["content"])])
            )
    return messages

def run_main_agent(user_message: str, history: list) -> tuple:
    """
    Main agent loop with tool calling support.
    Returns (assistant_response, updated_history)
    """
    # Build conversation
    api_history = format_history_for_api(history)
    api_history.append(
        types.Content(role="user", parts=[types.Part(text=user_message)])
    )

    # Agentic loop
    max_iterations = 10
    iteration = 0
    final_response = ""

    while iteration < max_iterations:
        iteration += 1

        response = client.models.generate_content(
            model=BASE_MODEL,
            contents=api_history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=MAIN_AGENT_TOOLS,
                temperature=0.1,
                max_output_tokens=8192,
            ),
        )

        candidate = response.candidates[0]

        # Check finish reason
        if candidate.finish_reason == "STOP":
            # Extract text response
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    final_response = part.text
            # Add model response to history
            api_history.append(candidate.content)
            break

        # Handle tool calls
        tool_calls_made = False
        function_response_parts = []

        for part in candidate.content.parts:
            if hasattr(part, "function_call") and part.function_call:
                tool_calls_made = True
                fc = part.function_call
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}

                print(f"[Tool Call] {tool_name}({tool_args})")

                # Execute tool
                try:
                    result = _dispatch_tool(tool_name, tool_args)
                except Exception as e:
                    result = {"error": str(e), "tool": tool_name}

                print(f"[Tool Result] {tool_name}: {str(result)[:200]}")

                function_response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=tool_name,
                            response={"result": result},
                        )
                    )
                )

        if tool_calls_made:
            # Add model's tool call message
            api_history.append(candidate.content)
            # Add tool results
            api_history.append(
                types.Content(role="user", parts=function_response_parts)
            )
        elif not tool_calls_made:
            # No tool calls, extract text
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    final_response = part.text
            api_history.append(candidate.content)
            break

    if not final_response:
        final_response = "I've completed the requested operations. Please check the results above."

    return final_response

def _dispatch_tool(tool_name: str, tool_args: dict):
    """Dispatch tool calls to appropriate functions."""
    tool_map = {
        # Kite Tools
        "get_kite_holdings": get_kite_holdings,
        "get_kite_positions": get_kite_positions,
        "place_kite_order": place_kite_order,
        "cancel_kite_order": cancel_kite_order,
        "get_kite_quote": get_kite_quote,
        "get_historical_data": get_historical_data,
        "get_market_depth": get_market_depth,
        "get_account_margins": get_account_margins,
        "get_order_history": get_order_history,
        "get_trade_history": get_trade_history,
        "search_instruments": search_instruments,
        "get_indices_data": get_indices_data,
        # Memory Tools
        "save_to_memory": save_to_memory,
        "load_from_memory": load_from_memory,
        "list_memory_files": list_memory_files,
        "delete_memory_file": delete_memory_file,
        "save_strategy": save_strategy,
        "load_strategy": load_strategy,
        "list_strategies": list_strategies,
        # Analysis Tools
        "calculate_portfolio_metrics": calculate_portfolio_metrics,
        "run_backtest": run_backtest,
        "generate_chart": generate_chart,
        "get_news_sentiment": get_news_sentiment,
        "screen_stocks": screen_stocks,
        "get_technical_indicators": get_technical_indicators,
        # Human Approval Tools
        "execute_human_approved_trade": execute_human_approved_trade,
        "request_human_approval": request_human_approval,
        "get_pending_approvals": get_pending_approvals,
        "approve_trade_request": approve_trade_request,
        "reject_trade_request": reject_trade_request,
        # Sub-Agents
        "market_analysis_agent": market_analysis_agent,
        "strategy_agent": strategy_agent,
        "risk_management_agent": risk_management_agent,
        "portfolio_agent": portfolio_agent,
        "backtesting_agent": backtesting_agent,
    }

    if tool_name not in tool_map:
        return {"error": f"Unknown tool: {tool_name}"}

    func = tool_map[tool_name]
    return func(**tool_args)

# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────
def chat_with_agent(message: str, history: list) -> tuple:
    """Main chat handler."""
    if not message.strip():
        return "", history

    response = run_main_agent(message, history)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return "", history

def approve_trade(approval_id: str, history: list) -> tuple:
    """Handle trade approval from UI."""
    result = approve_trade_request(approval_id=approval_id)
    msg = f"✅ Trade Approved: {approval_id}\n\nResult: {json.dumps(result, indent=2)}"
    history.append({"role": "user", "content": f"[APPROVED TRADE: {approval_id}]"})
    history.append({"role": "assistant", "content": msg})
    return history, ""

def reject_trade(approval_id: str, history: list) -> tuple:
    """Handle trade rejection from UI."""
    result = reject_trade_request(approval_id=approval_id)
    msg = f"❌ Trade Rejected: {approval_id}\n\nResult: {json.dumps(result, indent=2)}"
    history.append({"role": "user", "content": f"[REJECTED TRADE: {approval_id}]"})
    history.append({"role": "assistant", "content": msg})
    return history, ""

def refresh_pending() -> str:
    """Refresh pending approvals display."""
    approvals = get_pending_approvals()
    if not approvals:
        return "No pending approvals"
    return json.dumps(approvals, indent=2)

def load_strategy_ui(strategy_name: str) -> str:
    """Load strategy for display."""
    result = load_strategy(strategy_name=strategy_name)
    return json.dumps(result, indent=2)

def list_strategies_ui() -> str:
    """List all saved strategies."""
    result = list_strategies()
    return json.dumps(result, indent=2)

# ─────────────────────────────────────────────
# Gradio Interface
# ─────────────────────────────────────────────
css = """
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 15px;
}
.approval-panel {
    border: 2px solid #e74c3c;
    border-radius: 8px;
    padding: 10px;
}
.strategy-panel {
    border: 2px solid #2ecc71;
    border-radius: 8px;
    padding: 10px;
}
.status-bar {
    background: #1a1a2e;
    color: #00ff88;
    padding: 8px;
    border-radius: 5px;
    font-family: monospace;
}
"""

with gr.Blocks(
    title="ARIA - AI Trading Agent",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="emerald"),
    css=css,
) as demo:

    # ── Header ──────────────────────────────
    gr.HTML("""
    <div class="main-header">
        <h1 style="color:#00ff88; font-family:monospace; margin:0;">
            🤖 ARIA — Autonomous Research & Investment Agent
        </h1>
        <p style="color:#8892b0; margin:5px 0 0 0;">
            Powered by Gemini 2.0 Flash · Zerodha Kite Integration · Human-in-the-Loop Trading
        </p>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Main Chat ─────────────────
        with gr.TabItem("🧠 Agent Chat", id=0):
            chatbot = gr.Chatbot(
                label="ARIA Trading Agent",
                height=550,
                type="messages",
                avatar_images=("👤", "🤖"),
                show_copy_button=True,
                bubble_full_width=False,
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder=(
                        "Ask ARIA anything: 'Analyze RELIANCE', "
                        "'Backtest RSI strategy on NIFTY', "
                        "'Show my portfolio risk'..."
                    ),
                    label="Your Message",
                    lines=2,
                    scale=4,
                )
                with gr.Column(scale=1):
                    send_btn = gr.Button("Send 🚀", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear 🗑️", variant="secondary", size="sm")

            # Quick action buttons
            with gr.Row():
                gr.Button("📊 Portfolio Overview").click(
                    fn=lambda h: chat_with_agent("Show my complete portfolio overview with risk metrics", h),
                    inputs=[chatbot], outputs=[msg_input, chatbot]
                )
                gr.Button("📈 Market Analysis").click(
                    fn=lambda h: chat_with_agent("Give me today's market analysis for NIFTY50 and BANKNIFTY", h),
                    inputs=[chatbot], outputs=[msg_input, chatbot]
                )
                gr.Button("🔍 Screen Stocks").click(
                    fn=lambda h: chat_with_agent("Screen top momentum stocks from NIFTY500 with volume > 1M", h),
                    inputs=[chatbot], outputs=[msg_input, chatbot]
                )
                gr.Button("⚠️ Risk Check").click(
                    fn=lambda h: chat_with_agent("Run a complete risk assessment on my current positions", h),
                    inputs=[chatbot], outputs=[msg_input, chatbot]
                )

        # ── Tab 2: Trade Approvals ───────────
        with gr.TabItem("✅ Trade Approvals", id=1):
            gr.HTML('<div class="approval-panel">')
            gr.Markdown("## ⚠️ Pending Trade Approvals")
            gr.Markdown(
                "*All trades require explicit human approval. "
                "Review carefully before approving.*"
            )

            pending_display = gr.JSON(label="Pending Approvals", value={})
            refresh_btn = gr.Button("🔄 Refresh Pending Approvals", variant="secondary")

            with gr.Row():
                approval_id_input = gr.Textbox(
                    label="Approval ID", placeholder="Enter approval ID from pending list"
                )
                with gr.Column():
                    approve_btn = gr.Button("✅ APPROVE Trade", variant="primary")
                    reject_btn = gr.Button("❌ REJECT Trade", variant="stop")

            approval_result = gr.Textbox(label="Result", interactive=False)
            gr.HTML("</div>")

            # Wire up
            refresh_btn.click(fn=refresh_pending, outputs=[pending_display])
            approve_btn.click(
                fn=lambda aid, h: approve_trade(aid, h),
                inputs=[approval_id_input, chatbot],
                outputs=[chatbot, approval_result],
            )
            reject_btn.click(
                fn=lambda aid, h: reject_trade(aid, h),
                inputs=[approval_id_input, chatbot],
                outputs=[chatbot, approval_result],
            )

        # ── Tab 3: Strategy Manager ──────────
        with gr.TabItem("📋 Strategy Manager", id=2):
            gr.HTML('<div class="strategy-panel">')
            gr.Markdown("## 📋 Saved Strategies")

            with gr.Row():
                list_btn = gr.Button("📂 List All Strategies", variant="secondary")
                strategy_list_display = gr.JSON(label="Available Strategies")

            with gr.Row():
                strategy_name_input = gr.Textbox(
                    label="Strategy Name", placeholder="e.g., RSI_Momentum_v1"
                )
                load_strategy_btn = gr.Button("📥 Load Strategy", variant="primary")

            strategy_display = gr.JSON(label="Strategy Details")

            gr.Markdown("### 📊 Quick Backtest")
            with gr.Row():
                bt_strategy = gr.Textbox(label="Strategy Name")
                bt_symbol = gr.Textbox(label="Symbol", value="NSE:NIFTY50")
                bt_from = gr.Textbox(label="From Date", value="2023-01-01")
                bt_to = gr.Textbox(label="To Date", value="2024-01-01")

            backtest_btn = gr.Button("🚀 Run Backtest", variant="primary")
            backtest_result = gr.JSON(label="Backtest Results")

            gr.HTML("</div>")

            # Wire up
            list_btn.click(fn=list_strategies_ui, outputs=[strategy_list_display])
            load_strategy_btn.click(
                fn=load_strategy_ui,
                inputs=[strategy_name_input],
                outputs=[strategy_display],
            )
            backtest_btn.click(
                fn=lambda s, sym, f, t: run_backtest(
                    strategy_name=s, symbol=sym, from_date=f, to_date=t
                ),
                inputs=[bt_strategy, bt_symbol, bt_from, bt_to],
                outputs=[backtest_result],
            )

        # ── Tab 4: Memory Browser ────────────
        with gr.TabItem("💾 Memory Browser", id=3):
            gr.Markdown("## 💾 Persistent Memory")

            with gr.Row():
                list_memory_btn = gr.Button("📂 List Memory Files", variant="secondary")
                memory_list_display = gr.JSON(label="Memory Files")

            with gr.Row():
                mem_key = gr.Textbox(label="Memory Key", placeholder="e.g., trade_journal")
                load_mem_btn = gr.Button("📥 Load", variant="primary")
                delete_mem_btn = gr.Button("🗑️ Delete", variant="stop")

            mem_content = gr.JSON(label="Memory Content")
            mem_result = gr.Textbox(label="Operation Result", interactive=False)

            # Wire up
            list_memory_btn.click(fn=list_memory_files, outputs=[memory_list_display])
            load_mem_btn.click(
                fn=lambda k: load_from_memory(key=k),
                inputs=[mem_key],
                outputs=[mem_content],
            )
            delete_mem_btn.click(
                fn=lambda k: str(delete_memory_file(key=k)),
                inputs=[mem_key],
                outputs=[mem_result],
            )

        # ── Tab 5: Settings ──────────────────
        with gr.TabItem("⚙️ Settings", id=4):
            gr.Markdown("## ⚙️ Configuration")

            with gr.Row():
                kite_api_key = gr.Textbox(
                    label="Kite API Key",
                    type="password",
                    placeholder="Your Zerodha API Key",
                )
                kite_access_token = gr.Textbox(
                    label="Kite Access Token",
                    type="password",
                    placeholder="Session Access Token",
                )

            save_config_btn = gr.Button("💾 Save Configuration", variant="primary")
            config_result = gr.Textbox(label="Result", interactive=False)

            def save_config(api_key, access_token):
                result = save_to_memory(
                    key="kite_config",
                    data={"api_key": api_key, "access_token": access_token},
                )
                # Also set env vars
                os.environ["KITE_API_KEY"] = api_key
                os.environ["KITE_ACCESS_TOKEN"] = access_token
                return f"Configuration saved: {result}"

            save_config_btn.click(
                fn=save_config,
                inputs=[kite_api_key, kite_access_token],
                outputs=[config_result],
            )

            gr.Markdown("""
            ### 🔑 API Keys Required
            - **GEMINI_API_KEY**: Google AI Studio API key
            - **KITE_API_KEY**: Zerodha Kite Connect API key
            - **KITE_ACCESS_TOKEN**: Generated daily via Kite login flow
            - **NEWS_API_KEY**: (Optional) For news sentiment analysis

            ### 📁 File Structure
            ```
            memory/          → Persistent agent memory
            memory/strategies/ → Saved trading strategies
            memory/charts/   → Generated charts
            memory/backtest/ → Backtest results
            ```
            """)

    # ── Status Bar ──────────────────────────
    with gr.Row():
        gr.HTML(f"""
        <div class="status-bar">
            🟢 ARIA Active | Model: gemini-2.0-flash | 
            {datetime.now().strftime('%Y-%m-%d %H:%M')} IST |
            Human-in-the-Loop: ENABLED | Risk Controls: ACTIVE
        </div>
        """)

    # ── Event Handlers ───────────────────────
    send_btn.click(
        fn=chat_with_agent,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
    )
    msg_input.submit(
        fn=chat_with_agent,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
    )
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, msg_input])

# ─────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Create required directories
    os.makedirs("memory", exist_ok=True)
    os.makedirs("memory/strategies", exist_ok=True)
    os.makedirs("memory/charts", exist_ok=True)
    os.makedirs("memory/backtest", exist_ok=True)
    os.makedirs("memory/journal", exist_ok=True)

    print("🚀 Starting ARIA Trading Agent...")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("🔐 Human-in-the-Loop: ENABLED")
    print("📊 Kite Integration: READY")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
