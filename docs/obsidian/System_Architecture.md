# System Architecture

The architecture relies on a master-worker pattern where a central Orchestrator Agent classifies intents and routes tasks to specialized sub-agents. 

## 1. Orchestrator
Defined in `src/trading_system/main_agent.py`, the `OrchestratorAgent` acts as the central brain.
- **Intent Classification**: Evaluates the user prompt and categorizes it into one or more domains: `market_data`, `analysis`, `strategy`, `visualization`, `risk`, or `execution`.
- **Synthesis**: Combines the outputs of the sub-agents to formulate a final human-readable response.

## 2. Sub-Agents
Defined in `src/trading_system/sub_agents.py`, these agents have specific scopes and tools:
- **MarketDataAgent**: Fetches quotes and historical data.
- **AnalysisAgent**: Computes technical indicators (RSI, MACD, etc.).
- **StrategyAgent**: Backtests strategies (SMA Crossover, Brownian Motion, etc.).
- **VisualizationAgent**: Creates Plotly interactive charts.
- **RiskAgent**: Checks position sizing and evaluates risk.
- **ExecutionAgent**: Places orders (requires explicit human approval).

## 3. Data Layer
The system uses **Zerodha Kite** for primary data and live execution. If Kite is unavailable, it automatically falls back to **Yahoo Finance** for historical data and quotes. This logic is handled in `src/trading_system/tools.py`.

## 4. UI Layer
The project uses `dashboard.py` (a Gradio interface) and `flask_app.py` as frontend layers to interact with the LLM orchestrator.
