# Persistent Memory

The Trading Agent needs to remember past conversations, saved strategies, and portfolio context across sessions. This is handled by the `MemoryManager` in `src/trading_system/memory.py`.

## How it Works
The memory is entirely file-system backed, saving state to a local `data/` directory using JSON serialization. This provides a lightweight database without requiring external services like Redis or Postgres.

### Conversation Memory
`AgentMemory` is a dataclass storing a list of conversation dictionaries. 
When an agent communicates, `MemoryManager.append_conversation(agent_id, role, content)` is called, which appends the log and writes it to `data/memory/{agent_id}.json`.

### Strategy Persistence
When a user backtests a strategy, `tools.backtest_strategy()` calls `memory.save_strategy()`.
This saves the full JSON payload—including daily returns, Sharpe ratio, and drawdown metrics—into `data/strategies/{name}.json`. These can later be retrieved by `load_strategy_tool()` without rerunning the expensive backtest.

### Visualizations & Trade Logs
- **Charts**: When Plotly generates a candlestick chart in `tools.py`, an HTML file is saved to `data/visualizations/`, and the metadata is stored in memory.
- **Trade Logs**: Any executed (or simulated) trades are appended to a daily file in `data/trade_logs/{YYYY-MM-DD}.json`.

By abstracting state into `memory.py`, the LLM agents remain completely stateless and horizontally scalable.
