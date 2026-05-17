### Deep Trading Agent





~

> Multi-agent AI trading system powered by **LangChain**, **Zerodha Kite**, and **Tavily Search**. Built around a single reusable `LLMService` and orchestrated through a Gradio dashboard.

---

## 📑 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Getting Started](#-getting-started)
- [Dashboard Walkthrough](#-dashboard-walkthrough)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Using as a Library](#-using-as-a-library)
- [Persistent Memory](#-persistent-memory)
- [Strategies & Backtesting](#-strategies--backtesting)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [License](#-license)
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
