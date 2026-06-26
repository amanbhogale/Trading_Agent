# Deep Trading Agent

> A multi-agent AI trading system powered by **LangChain**, **Vite + React (TypeScript)**, **Flask API**, **Zerodha Kite**, and **Tavily Search**. Built around a centralized reusable `LLMService`, orchestrated through an Orchestrator Agent, and featuring deep learning (LSTM & Reinforcement Learning DQN) backtesting capabilities.

---

## 📑 Table of Contents

- [✨ Features](#-features)
- [🏗 Architecture](#-architecture)
- [📦 Project Structure](#-project-structure)
- [🚀 Installation & Setup](#-installation--setup)
- [🖥 Running the System](#-running-the-system)
- [🧪 Deep Learning Backtesting](#-deep-learning-backtesting)
- [🧠 Persistent Memory](#-persistent-memory)
- [🔒 Git Guidelines](#-git-guidelines)
- [⚠️ Disclaimer](#-disclaimer)

---

## ✨ Features

| Capability | Description |
|---|---|
| 💻 **React Web App** | Modern React + Vite (TypeScript) frontend dashboard containing comprehensive analysis, live log viewing, interactive charting, configuration management, and portfolio monitoring. |
| 💬 **Conversational Agent** | Free-form chat routing to specialized sub-agents via a central orchestrator. |
| 📊 **Technical Analysis** | Built-in technical indicators (RSI, MACD, Bollinger Bands, EMA 9/21/50) and interactive candlestick charting using TradingView lightweight-charts. |
| 🧪 **Algorithmic Backtesting** | Backtesting engine for SMA crossover, RSI mean reversion, and MACD trend strategies. |
| 🤖 **Deep Learning** | Advanced LSTM predictors and Deep Q-Network (DQN) reinforcement learning agents for market trading decision-making. |
| 💼 **Live Portfolio** | Real-time monitoring of holdings, positions, orders, and P&L via Zerodha Kite API. |
| ⚡ **Order Execution** | Automated order placement with human-in-the-loop validation and approvals. |
| 🔍 **Web Search** | Tavily-powered real-time internet search for news aggregation, sentiment analysis, and general research. |
| 🎛️ **Multi-Provider LLM** | Support for OpenRouter, OpenAI, Anthropic, and Google Gemini with quick-select chips. |

---

## 🏗 Architecture

```text
┌─────────────────────── React Frontend (Vite) ────────────────────────┐
│ ⚙️ Config · 💬 Chat · 📊 Analysis · 🧪 Backtest · 💼 Portfolio · Logs │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │ (HTTP / JSON API)
                                   ▼
┌───────────────────────── Flask Backend API ──────────────────────────┐
│   Serving API endpoints, routing to main agent, and managing logs   │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────── OrchestratorAgent (main_agent.py) ──────────────┐
│  • Intent classification & Routing                                   │
│  • Task synthesis & Sub-agent delegation                             │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
   MarketDataAgent           AnalysisAgent             StrategyAgent
   VisualizationAgent        RiskAgent                 ExecutionAgent
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   ▼
                           ┌───────────────┐
                           │   tools.py    │  Kite · Indicators · Charts
                           │   memory.py   │  File persistence
                           └───────────────┘
                                   ▲
                                   │
                     ┌─────────────┴─────────────┐
                     │       llm_service.py      │ (Single shared LLM Config)
                     └───────────────────────────┘
```

---

## 📦 Project Structure

```text
Trading_Agent/
├── flask_app.py                # Flask API application (main backend entry)
├── dashboard.py                # Gradio UI (legacy backend entry)
├── kite_auth.py                # Kite authentication helper script
├── init_db.py                  # Database initialization script
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation file
│
├── frontend/                   # React + TypeScript + Vite Frontend Web App
│   ├── src/
│   │   ├── components/         # Reusable UI components (TickerBar, TickerSearch, etc.)
│   │   ├── pages/              # Routing pages (Analysis, Backtest, Chat, Portfolio, Logs, etc.)
│   │   ├── App.tsx             # Main App layout & routing definition
│   │   ├── api.ts              # Centralized API service using Axios
│   │   └── index.css           # Global custom CSS styles (Aesthetic design system)
│   ├── package.json            # npm package definition
│   └── vite.config.ts          # Vite configuration
│
├── src/trading_system/         # Core Multi-agent Framework
│   ├── llm_service.py          # Unified LLM Service (Google Gemini, OpenAI, Anthropic, etc.)
│   ├── main_agent.py           # Orchestrator Agent logic
│   ├── sub_agents.py           # Specialized sub-agent declarations
│   ├── tools.py                # @tool definitions for data, indicators, execution, and search
│   ├── strategies.py           # Algorithmic strategy logic
│   ├── api_manager.py          # API connection integrations
│   ├── model_predictor.py      # Predictor module integrations
│   ├── news_classifier.py      # Live news sentiment classifier
│   └── memory.py               # File-system persistent memory
│
├── deep_learning_backtest/     # LSTM & Reinforcement Learning (DQN) pipeline
│   ├── configs/                # Configuration files for training and pipelines
│   ├── 01_environment_setup.ipynb
│   ├── 02_data_pipeline.ipynb
│   ├── 03_lstm_model.ipynb
│   ├── 04_dqn_agent.ipynb
│   ├── 05_backtest_engine.ipynb
│   ├── 06_results_dashboard.ipynb
│   └── helper scripts          # build_colab.py, combine_notebooks.py, update_agent.py, etc.
│
├── static/ & templates/        # Legacy Flask HTML static folders
└── data/                       # Local data directory (Auto-created, ignored in Git)
    ├── memory/                 # JSON file-system conversation memory
    ├── strategies/             # Saved backtest output results
    ├── visualizations/         # Exported HTML charts
    └── trade_logs/             # JSON-based daily trade audit logs
```

---

## 🚀 Installation & Setup

### 1. Python Backend Setup
Create a virtual environment and install the required Python packages:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Verify your Python backend components are correctly installed:
```bash
python -c "from src.trading_system.llm_service import LLMService; print('LLMService ✅')"
python -c "import src.trading_system.tools as t; print('Tools ✅')"
```

### 2. React Frontend Setup
Install the npm dependencies in the `frontend` directory:

```bash
cd frontend
npm install
cd ..
```

---

## 🖥 Running the System

You will need two terminals running concurrently to launch the full system locally.

### Terminal 1: Backend API
```bash
source venv/bin/activate
python flask_app.py
```
This launches the Flask server at `http://127.0.0.1:5000/`.

### Terminal 2: React Frontend
```bash
cd frontend
npm run dev
```
This runs the Vite dev server, typically at `http://localhost:5173/`. Open this URL in your web browser to access the complete application dashboard.

*(Note: The old Gradio layout can still be run via `python dashboard.py`)*

---

## 🧪 Deep Learning Backtesting

The `deep_learning_backtest/` folder contains Jupyter notebooks detailing the steps to set up, extract datasets, train predictive models, and evaluate performance:
- **01-02**: Environment Setup & Data Pipeline (data ingestion and engineering features)
- **03**: LSTM Model (time-series sequence training for direction classification)
- **04-05**: DQN Agent & Backtesting Engine (reinforcement learning environment simulation)
- **06**: Results Dashboard (visualizing metrics and trade scorecard outputs)

---

## 🧠 Persistent Memory

The orchestrator and specialized sub-agents retain state using a file-system JSON database stored under `data/memory/`. This persists conversations, strategy logs, risk metrics, and order history across server restarts.

---

## 🔒 Git Guidelines

To maintain repository cleanliness and prevent uploading large binaries or private keys, the following file types are configured to remain **locally on your machine** (and are excluded in [`.gitignore`](.gitignore)):
- All `test_*.py` unit test scripts (e.g. `test_kite.py`, `test_edge_cases.py`, etc.) and `pytest.ini`
- Python bytecode files (`__pycache__/`, `*.pyc`)
- Local datasets and processed caches (`data/`, `deep_learning_backtest/data/`)
- Trained PyTorch models (`deep_learning_backtest/models/`)
- Training performance reports, trade outputs, and EDA charts (`deep_learning_backtest/results/`, `deep_learning_backtest/eda_results/`, `output/`)
- Draft & exploratory notebooks (`backtested_dat.ipynb`, `fundemental_analysis.ipynb`, etc.)

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. Trading in financial markets involves substantial risk of loss. The authors are not responsible for any financial losses incurred from using this system. Always paper-trade your strategies before deploying capital and consult a registered financial advisor.
