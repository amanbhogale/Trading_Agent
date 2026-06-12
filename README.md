### Deep Trading Agent


> Multi-agent AI trading system powered by **LangChain**, **Zerodha Kite**, and **Tavily Search**. Built around a single reusable `LLMService` and orchestrated through a Gradio dashboard.

---

## 📑 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#installation)
- [Configuration](#-configuration)
- [Getting Started](#-getting-started)
- [Dashboard Walkthrough](#-dashboard-walkthrough)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Using as a Library](#-using-as-a-library)
- [Persistent Memory](#-persistent-memory)
- [Strategies & Backtesting](#-strategies--backtesting)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Disclaimer](#-disclaimer)

---

## ✨ Features

| Capability                | Description                                                  |
|---------------------------|--------------------------------------------------------------|
| 💬 Conversational Agent   | Free-form chat that routes to specialized sub-agents         |
| 📊 Technical Analysis     | RSI, MACD, Bollinger Bands, EMA 9/21/50                      |
| 🧪 Backtesting            | SMA crossover, RSI mean reversion, MACD trend                |
| 📈 Visualizations         | Candlestick, equity curves, portfolio dashboard (Plotly)     |
| 💼 Live Portfolio         | Holdings, positions, P&L via Zerodha Kite                    |
| ⚡ Order Execution        | Place / cancel orders with human-in-the-loop approval        |
| 🔍 Internet Search        | Tavily-powered live web search for news & research           |
| 🧠 Persistent Memory      | File-system memory for conversations, strategies, trades     |
| 🎛️ Multi-Provider         | OpenRouter, OpenAI, Anthropic, Google Gemini                 |

---

## 🏗 Architecture

```text
┌────────────────────────── Gradio Dashboard ──────────────────────────┐
│  ⚙️ Config · 💬 Chat · 📊 Analysis · 🧪 Backtest · 💼 Portfolio        │
│  ⚡ Execute Trade · 🧠 Memory · 🔍 Search                              │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────── OrchestratorAgent (main_agent.py) ──────────────┐
│  • Intent classification                                             │
│  • Routes tasks to sub-agents                                        │
│  • Synthesizes final answer                                          │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
   MarketDataAgent       AnalysisAgent          StrategyAgent
   VisualizationAgent    RiskAgent              ExecutionAgent
                               │
                               ▼
                       ┌───────────────┐
                       │   tools.py    │  Kite · Indicators · Charts
                       │   memory.py   │  File-system persistence
                       └───────────────┘
                               ▲
                               │
                  ┌────────────┴────────────┐
                  │   llm_service.py        │  ← Single shared LLM
                  │   (LLMConfig + Service) │
                  └─────────────────────────┘
```
## Installation

step-1 create virtual enviroment

python -m venv {eniroment_name}

step-2 Install Dependencies

`pip install --upgrade pip`
`pip install -r requirements.txt`

step-4 verify installation

python -c "from llm_service import LLMService;  print('llm_service ✅')"
python -c "import tools;                         print('tools ✅')"
python -c "from sub_agents import MarketDataAgent; print('sub_agents ✅')"
python -c "from main_agent import build_orchestrator; print('main_agent ✅')"


for openrouter keys https://openrouter.ai/keys

for Tavily Keys https://tavily.com/


## Running the System 

Launch the Flask Application:

`cd Trading_Agent`
`python flask_app.py`

*(Note: The old Gradio dashboard can still be run via `python -m dashboard`)*

## Project Structure

```
Trading_Agent/
├── flask_app.py           # Flask web application (main entry point)
├── dashboard.py           # Gradio UI (legacy entry point)
├── kite_auth.py           # Kite authentication script
├── test_kite.py           # Kite connection test
├── test_edge_cases.py     # Edge cases testing script
├── backtested_dat.ipynb   # Backtesting notebook
├── fundemental_analysis.ipynb # Fundamental analysis notebook
├── requirements.txt       # pip dependencies
├── README.md              # this file
├── src/                   # Core application source code
│   └── trading_system/
│       ├── __init__.py    
│       ├── llm_service.py # base LLM wrapper (single source of truth)
│       ├── memory.py      # file-system persistent memory
│       ├── tools.py       # @tool functions (Kite, indicators, charts, search)
│       ├── sub_agents.py  # specialized agents (market, analysis, strategy …)
│       ├── main_agent.py  # OrchestratorAgent — routes to sub-agents
│       ├── planner_agent.py # Planner agent for task orchestration
│       ├── data/          # data management modules
│       └── execution/     # execution logic modules
├── static/                # Flask static files
├── templates/             # Flask templates
├── output/                # Outputs (visualizations, etc.)
├── docs/                  # Documentation
└── data/                  # Auto-created on first run
    ├── memory/            # JSON conversation & context
    ├── strategies/        # saved backtest results
    ├── visualizations/    # HTML Plotly charts
    └── trade_logs/        # daily trade JSON logs
```


## ⚠️ Disclaimer

`This software is for educational purposes only.
Trading in financial markets involves substantial risk of loss.
The authors are not responsible for any financial losses incurred from using this system.
Always paper-trade strategies before going live and consult a SEBI-registered investment advisor.`



