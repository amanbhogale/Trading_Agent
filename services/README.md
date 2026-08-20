# Trading Agent — Microservices

This directory contains all the individual microservice FastAPI apps and their Dockerfiles.

## Architecture

```
trading_net (Docker bridge network)
│
├── db               :5432   PostgreSQL 15
│
├── market-data      :8001   MarketDataAgent (quotes, portfolio, history)
├── analysis         :8002   AnalysisAgent   (technical indicators)
├── strategy         :8003   StrategyAgent   (backtesting)
├── visualization    :8004   VisualizationAgent (charts)
├── risk             :8005   RiskAgent       (position sizing, risk checks)
├── execution        :8006   ExecutionAgent  (order placement)
│
├── orchestrator     :8000   Routes messages to the agents above via HTTP
├── data-etl         :8007   PySpark ETL pipeline + indicator API
│
├── backend          :5000   Flask API Gateway (proxies to orchestrator)
└── frontend         :5173   React + Vite UI
```

## Quickstart

```bash
# 1. Copy and fill in your environment variables
cp .env.example .env

# 2. Initialize the database (only needed once)
docker compose run --rm backend python init_db.py

# 3. Start everything
docker compose up --build

# 4. (Optional) Trigger the PySpark ETL to precompute indicators
curl -X POST http://localhost:8007/run-etl
curl http://localhost:8007/etl-status
```

## Service Endpoints

| Service | URL | Key Routes |
|---------|-----|------------|
| Frontend | http://localhost:5173 | Full React dashboard |
| Flask API Gateway | http://localhost:5000 | `/api/chat`, `/api/analysis` |
| Orchestrator | http://localhost:8000 | `POST /chat`, `GET /health/agents` |
| Market Data | http://localhost:8001 | `POST /run`, `GET /health` |
| Analysis | http://localhost:8002 | `POST /run` |
| Strategy | http://localhost:8003 | `POST /run` |
| Visualization | http://localhost:8004 | `POST /run` |
| Risk | http://localhost:8005 | `POST /run` |
| Execution | http://localhost:8006 | `POST /run`, `POST /execute-with-approval` |
| Data ETL | http://localhost:8007 | `POST /run-etl`, `GET /etl-status`, `GET /tickers`, `GET /indicators/{symbol}` |

## Adding a New Agent Service

1. Create `services/<name>/main.py` following the pattern of any existing service.
2. Add it to `docker-compose.yml` using the `x-agent-build` and `x-agent-env` anchors.
3. Register its URL in `services/orchestrator/main.py` `SERVICES` dict.
