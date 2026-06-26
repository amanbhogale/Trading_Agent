# File Structure

The project is divided into several main components.

## Root Directory
- `dashboard.py` - The Gradio UI for the trading system.
- `flask_app.py` - Flask API interface.
- `requirements.txt` - Core dependencies.
- `test_kite.py` & `test_edge_cases.py` - Standalone testing scripts.

## `src/trading_system/`
This is the core Python package for the Multi-Agent LLM logic.
- `main_agent.py`: Houses the Orchestrator Agent. See [[System_Architecture]].
- `sub_agents.py`: Contains the definitions of specialized workers (Data, Analysis, Risk, etc.). See [[Code_Logic]].
- `tools.py`: A massive file containing all `@tool` implementations. Handles API calls to Kite, Yahoo Finance, and Tavily.
- `memory.py`: Handles persistent storage. See [[Persistent_Memory]].
- `llm_service.py`: Wrapper for the OpenAI-compatible API (e.g., OpenRouter).

## `deep_learning_backtest/`
This directory is an independent pipeline for developing pure Deep Learning and Reinforcement Learning models.
- `01_environment_setup.ipynb` through `06_results_dashboard.ipynb`: Step-by-step Jupyter notebooks for training models.
- `Colab_Master_Pipeline.ipynb`: A unified notebook for Google Colab execution.
- Contains LSTM models for price prediction and DQN (Deep Q-Network) agents for autonomous trading without LLMs.

## `data/`
Generated at runtime by the system.
- `data/memory/` - Agent conversation logs.
- `data/visualizations/` - Saved Plotly HTML charts.
- `data/strategies/` - Saved backtest results.
