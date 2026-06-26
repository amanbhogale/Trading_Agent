# Trading Agent Documentation

Welcome to the Trading Agent documentation! This vault contains in-depth documentation covering the entire project architecture, file system, code logic, and more.

## Quick Links
- [[System_Architecture]] - Overall design, components, and interactions.
- [[File_Structure]] - The project's directory structure and what each folder/file does.
- [[Code_Logic]] - How the Multi-Agent LLM system operates under the hood.
- [[Persistent_Memory]] - How the agent maintains context, saves strategies, and logs trades.

## Project Overview
The Trading Agent is a multi-agent system powered by LLMs to coordinate market data retrieval, technical analysis, strategy backtesting, visualization, risk evaluation, and trade execution (via Zerodha Kite). It falls back to Yahoo Finance for market data when Kite is unavailable. It also includes an advanced Deep Learning / Reinforcement Learning pipeline (`deep_learning_backtest`) for more complex AI-driven trading models.
