# ============================================================
# Dockerfile — mongo-etl service
# ============================================================
FROM python:3.10-slim

# ---- OS-level dependencies (includes JDK for PySpark) ------
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        curl \
        default-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

# ---- Java environment for PySpark --------------------------
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# ---- Python environment ------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- Application root --------------------------------------
WORKDIR /app

# Install base requirements first (better Docker layer caching)
COPY services/base_requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional ETL/service dependencies
RUN pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        pymongo \
        "pymongo[srv]" \
        motor \
        python-dotenv \
        pyspark \
        pyarrow \
        yfinance \
        numpy \
        pandas \
        confluent-kafka

# ---- Copy application source --------------------------------
# Project source tree (contains src/trading_system/mongo_etl.py etc.)
COPY src/ ./src/

# Top-level kite auth helper
COPY kite_auth.py .

# Service-specific code
COPY services/mongo-etl/ ./service/

# ---- Working directory for the service ---------------------
WORKDIR /app/service

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]
