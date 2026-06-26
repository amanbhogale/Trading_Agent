# 🧠 Deep Learning Backtesting Suite — LSTM + DQN Strategy

This folder contains a modular, notebook-driven backtesting framework for deep learning-based trading strategies using an **all-PyTorch** stack:
- **LSTM** (Long Short-Term Memory) for price/feature sequence modelling
- **DQN** (Deep Q-Network) for reinforcement learning-based policy optimization

## 📂 Folder Structure
```
deep_learning_backtest/
├── 01_environment_setup.ipynb         # Package install, version check, GPU detect
├── 02_data_pipeline.ipynb             # Data fetch, cleaning, feature engineering
├── 03_lstm_model.ipynb                # LSTM signal model training & evaluation
├── 04_dqn_agent.ipynb                 # Dueling Double DQN + PER — training
├── 05_backtest_engine.ipynb           # Full backtest: metrics, equity curve, tearsheet
├── 06_results_dashboard.ipynb         # Interactive Plotly strategy dashboard
├── configs/
│   ├── model_config.yaml              # Model hyperparameters (LSTM + DQN + env)
│   └── data_config.yaml              # Dataset & symbol config
├── data/
│   ├── raw/                           # Downloaded raw OHLCV data (CSV)
│   └── processed/                    # Feature-engineered + sequence datasets (PKL)
├── models/
│   ├── lstm_checkpoints/              # Saved LSTM weights (best_lstm.pth)
│   └── dqn_checkpoints/              # Saved DQN weights (best_dqn.pth)
├── results/
│   ├── trades/                        # Trade logs (trade_log.csv)
│   └── metrics/                       # Performance metrics (JSON + HTML dashboards)
└── requirements_dl.txt                # All package requirements (PyTorch-based)
```

## ⚡ Tech Stack — Pure PyTorch (Python 3.14 compatible)

| Component | Implementation |
|---|---|
| LSTM Signal Model | `torch.nn.LSTM` — stacked + LayerNorm + Dropout |
| DQN Agent | Dueling + Double DQN + Prioritized Experience Replay |
| RL Environment | Custom `gymnasium.Env` (`TradingEnv`) |
| Data Pipeline | `yfinance` + `pandas_ta` (32 technical indicators) |
| Normalisation | `sklearn.MinMaxScaler` (fit on train only) |
| Backtesting | NumPy (Sharpe, Sortino, Calmar, VaR, CVaR) |
| Visualisation | `plotly`, `matplotlib`, `seaborn` |

> **No TensorFlow / Keras.** All deep learning is pure PyTorch — works on Python 3.14+.

## 🚀 Quickstart
```bash
pip install -r requirements_dl.txt
```
Then run notebooks **in order**:

| # | Notebook | Output |
|---|---|---|
| 01 | `01_environment_setup.ipynb` | Environment verified, seeds set |
| 02 | `02_data_pipeline.ipynb` | `data/processed/*.pkl` |
| 03 | `03_lstm_model.ipynb` | `models/lstm_checkpoints/best_lstm.pth` + `lstm_test_probs.pkl` |
| 04 | `04_dqn_agent.ipynb` | `models/dqn_checkpoints/best_dqn.pth` + `dqn_eval_portfolio.pkl` |
| 05 | `05_backtest_engine.ipynb` | `results/metrics/performance_report.json` + HTML charts |
| 06 | `06_results_dashboard.ipynb` | `results/metrics/full_dashboard.html` |
