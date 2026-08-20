# Trading Agent Runbook

This guide covers everything you need to know to set up, run, and manage the Trading Agent application.

## 📋 Prerequisites
- **Docker** and **Docker Compose** installed on your machine.
- Required API keys for LLMs (OpenRouter/Gemini) and market data providers.

## ⚙️ Setup Instructions

1. **Configure Environment Variables**
   Before running the application, you need to set up your `.env` file.
   ```bash
   # Copy the example env file
   cp .env.example .env
   ```
   Open `.env` in your preferred text editor and fill in the required keys, primarily:
   - `OPENROUTER_API_KEY` (or `GEMINI_API_KEY`)
   - `KITE_API_KEY` & `KITE_ACCESS_TOKEN` (for Indian Equities)
   - `TAVILY_API_KEY` (for web search capabilities)

## 🚀 Running the Application

The entire stack is containerized and managed via Docker Compose.

**Start the complete system:**
```bash
# Build and start all services (runs in the foreground)
docker compose up --build

# OR run in detached mode (background)
docker compose up --build -d
```

## 🌐 Accessing the Application

Once the containers are successfully running, the applications are exposed on the following ports:

- **React Frontend**: [http://localhost:3000](http://localhost:3000)
- **Flask API Gateway**: [http://localhost:5000](http://localhost:5000)
- **Data ETL Dashboard**: [http://localhost:8007](http://localhost:8007)

*Note: Individual microservices (like strategy, risk, market data, etc.) run internally on port 8000 within the Docker network.*

## 🛠️ Useful Commands

**View logs for all services:**
```bash
docker compose logs -f
```

**View logs for a specific service (e.g., backend, orchestrator):**
```bash
docker compose logs -f backend
docker compose logs -f orchestrator
```

**Rebuild a single service after making code changes:**
```bash
docker compose up --build backend
```

**Stop all running services:**
```bash
docker compose down
```

**Stop and completely wipe the database volume (reset):**
```bash
docker compose down -v
```

## 🏗️ Architecture Overview

The `docker-compose up` command spins up the following services:
- **trading_db**: PostgreSQL Database
- **trading_frontend**: React UI
- **trading_backend**: Main Flask API Gateway
- **trading_orchestrator**: Agent Orchestrator
- **trading_market_data**: Market Data Agent
- **trading_analysis**: Analysis Agent
- **trading_strategy**: Strategy Agent
- **trading_visualization**: Visualization Agent
- **trading_risk**: Risk Agent
- **trading_execution**: Execution Agent
- **trading_data_etl**: PySpark Data Pipeline
