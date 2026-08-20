# Distributed Deep Trading Agent

> An enterprise-grade, event-driven, multi-agent AI trading system. Powered by **Microservices**, **Kafka**, **LangChain**, **Vite + React (TypeScript)**, **Flask API**, **Zerodha Kite**, and advanced **Deep Learning (TFT, LSTM, DQN)**. Built around a centralized orchestrator and a robust real-time data engineering pipeline for automated trading and market analysis.

---

## 📑 Table of Contents

- [✨ Key Features](#-key-features)
- [🏗 Architecture](#-architecture)
- [📦 Project Structure](#-project-structure)
- [🚀 Installation & Setup](#-installation--setup)
- [🖥 Running the System](#-running-the-system)
- [🧪 Deep Learning & Backtesting](#-deep-learning--backtesting)
- [🧠 Persistent Memory](#-persistent-memory)
- [🔒 Git Guidelines](#-git-guidelines)
- [⚠️ Disclaimer](#-disclaimer)

---

## ✨ Key Features

| Capability | Description |
|---|---|
| 🏛️ **Microservices Architecture** | Highly scalable, decoupled services (Risk, Execution, Strategy, Analysis) communicating via **Kafka** event streams. |
| 🐳 **Containerized Deployment** | Fully Dockerized stack managed via `docker-compose`, ensuring consistent environments from development to production. |
| 💬 **Conversational AI Framework** | Free-form chat routing to specialized sub-agents via a central orchestrator and a new **Shepherd Layer** for supervision. |
| 🤖 **Advanced Deep Learning** | Cutting-edge **Temporal Fusion Transformers (TFT)**, LSTM sequence predictors, and Double Deep Q-Network (DDQN) reinforcement learning agents. |
| ⚡ **Robust Data Engineering** | High-throughput data pipelines leveraging **MongoDB**, **PySpark**, and live **Orderbook Collectors** for historical replay and real-time streaming. |
| 💻 **React Web Dashboard** | Modern Vite + TypeScript frontend for comprehensive analysis, interactive charting, live portfolio monitoring, and backtest results. |
| 💼 **Live Portfolio & Execution** | Real-time monitoring of holdings, positions, and automated order placement via **Zerodha Kite** and **Binance** APIs. |
| 🔍 **Web Search & Sentiment** | Real-time internet search and live news sentiment classification for fundamental analysis and market context. |
| 🎛️ **Multi-Provider LLM** | Seamlessly swap between Google Gemini, OpenAI, Anthropic, and local models with a unified `LLMService`. |

---

## 🏗 Architecture

The system has evolved from a monolithic Flask app into a distributed, event-driven architecture, enabling massive scalability for high-frequency data and concurrent agent tasks.

```text
┌─────────────────────── React Frontend (Vite) ────────────────────────┐
│ ⚙️ Config · 💬 Chat · 📊 Analysis · 🧪 Backtest · 💼 Portfolio · Logs │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │ (HTTP / WebSockets)
                                   ▼
┌───────────────────────── API Gateway / Flask ────────────────────────┐
│   Serving REST endpoints, managing WebSockets, & Shepherd Routing    │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │ (Kafka Event Stream)
                                   ▼
┌───────────────────────── Microservices Layer ────────────────────────┐
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌─────────┐ │
│  │ Orchestrator  │ │ Market Data   │ │ Strategy/Risk │ │ Exec.   │ │
│  │    Agent      │ │ (Mongo/Kafka) │ │ Agents (LLM)  │ │ Agent   │ │
│  └──────┬────────┘ └──────┬────────┘ └──────┬────────┘ └────┬────┘ │
└─────────┼─────────────────┼─────────────────┼───────────────┼──────┘
          ▼                 ▼                 ▼               ▼
┌─────────────────── External Integrations & Data ───────────────────┐
│ Kite/Binance APIs · Tavily Search · MongoDB · Redis/File Memory    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Project Structure

```text
Trading_Agent/
├── docker-compose.yml          # Container orchestration for the entire stack
├── flask_app.py                # Main backend API gateway & WebSocket server
├── frontend/                   # React + TypeScript + Vite Frontend App
│
├── services/                   # Dedicated Microservices
│   ├── orchestrator/           # Central task delegation
│   ├── market-data/            # Kafka producers for Binance & Kite
│   ├── data-etl/ & mongo-etl/  # High-throughput data pipelines
│   ├── strategy/ & risk/       # Specialized agent execution
│   └── execution/              # Order routing and execution validation
│
├── src/trading_system/         # Core Trading Framework
│   ├── main_agent.py           # Orchestrator logic
│   ├── shepherd_layer.py       # Agent supervision and guardrails
│   ├── api_manager.py          # Exchange integrations (Kite, Binance)
│   ├── model_predictor.py      # TFT & LSTM inference integrations
│   ├── orderbook_collector.py  # Live Level-2 data streaming
│   ├── mongo_schema.py         # MongoDB database schemas
│   └── historical_replayer.py  # Replay engine for agent backtesting
│
├── deep_learning_backtest/     # Advanced ML & Reinforcement Learning
│   ├── 07_transformer_model.ipynb # TFT architecture exploration
│   ├── tft_model.py            # Temporal Fusion Transformer implementation
│   ├── train_tft_ddqn.py       # Training pipeline for TFT-DDQN agents
│   └── ...                     # Standard LSTM & DQN pipelines
│
└── data/                       # Local data directory (Memory, strategies, logs)
```

---

## 🚀 Installation & Setup

### Option 1: Docker (Recommended)
The easiest way to run the entire distributed system, including databases and message brokers, is via Docker Compose.

```bash
# Build and start all services in detached mode
docker-compose up -d --build
```
*This will spin up the Frontend, Flask API, Kafka, Zookeeper, MongoDB, and all Agent Microservices.*

### Option 2: Manual Local Setup (Development)

**1. Python Backend Setup**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r services/base_requirements.txt
```

**2. React Frontend Setup**
```bash
cd frontend
npm install
cd ..
```

---

## 🖥 Running the System (Manual Mode)

If not using Docker, you will need to start infrastructure services (MongoDB, Kafka) manually, followed by the app layers:

### Terminal 1: Backend API
```bash
source venv/bin/activate
python flask_app.py
```

### Terminal 2: React Frontend
```bash
cd frontend
npm run dev
```

### Terminal 3+: Microservices
You can individually run the required microservices from the `services/` directory depending on the pipeline you are testing (e.g., `python services/market-data/main.py`).

---

## 🧪 Deep Learning & Backtesting

The `deep_learning_backtest/` directory contains our state-of-the-art modeling pipeline:
- **Temporal Fusion Transformers (TFT)**: Advanced multi-horizon forecasting implemented in `tft_model.py` and trained via `train_transformer.py`.
- **Double Deep Q-Networks (DDQN)**: Reinforcement learning agents integrating TFT features (`train_tft_ddqn.py`) for superior market navigation.
- **Historical Replayer**: The `historical_replayer.py` engine simulates realistic market environments to validate agent logic over past data.

---

## 🧠 Persistent Memory & Data

State retention has been upgraded to support both local file-system JSON stores (`data/memory/`) and robust document storage via **MongoDB**. This ensures that conversational context, strategy backtests, and high-frequency orderbook snapshots are securely persisted across sessions.

---

## 🔒 Git Guidelines

To maintain repository cleanliness, the following are excluded from version control:
- Environment variable files (`.env`, `atlas-credentials.env`)
- Local datasets, processed caches, and `__pycache__`/`venv` directories.
- Large PyTorch model weights (`*.pt`), output logs (`*.log`), and Jupyter checkpoints.
- Infrastructure data directories (`kafka/`).

*See [`.gitignore`](.gitignore) for the comprehensive list.*

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. Trading in financial markets involves substantial risk of loss. The authors are not responsible for any financial losses incurred from using this system. Always paper-trade your strategies before deploying capital and consult a registered financial advisor.
