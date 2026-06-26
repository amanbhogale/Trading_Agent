# Code Logic

The core logic of the trading agent centers around the **LangChain `@tool` paradigm** and prompt engineering.

## Tool Execution (`tools.py`)
Tools are pure Python functions decorated with `@tool` from LangChain. 
For instance, the `backtest_strategy` tool is defined here. When the `StrategyAgent` receives a prompt, the LLM decides to invoke `backtest_strategy`, passing JSON arguments for the symbol, strategy name (e.g., `brownian_motion`), and lookback days.

```python
@tool
def backtest_strategy(symbol: str, strategy: str, params: str = "{}", days: int = 365) -> str:
    # Logic...
```

## Agent Lifecycle
1. **User Input**: The user sends a message via `dashboard.py`.
2. **Orchestrator Routing**: `OrchestratorAgent.run()` is called. It queries the LLM with a classification prompt to determine the `intent`.
3. **Sub-Agent Delegation**: If the intent includes `market_data`, `MarketDataAgent.run()` is executed. The sub-agent is equipped with tools like `get_quote` and `get_historical_data`. The LLM agent uses ReAct (Reasoning and Acting) to loop through tool calls until it finds the answer.
4. **Result Synthesis**: `OrchestratorAgent._synthesize()` combines all sub-agent string outputs into a cohesive response for the user.

## Deep Learning / RL (`deep_learning_backtest`)
Unlike the LLM Agent, the `deep_learning_backtest` pipeline uses standard ML libraries (PyTorch). The `DQN` agent, for example, is trained by running millions of episodes over historical data, learning a policy $\pi(s)$ that maximizes cumulative reward (Sharpe ratio or pure P&L).
