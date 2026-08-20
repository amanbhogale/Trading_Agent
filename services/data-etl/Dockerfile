FROM python:3.10-slim

# Java is required for PySpark
RUN apt-get update && apt-get install -y \
    gcc libpq-dev curl \
    default-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app

# Python deps: shared agent deps + PySpark specific
COPY services/base_requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir psycopg2-binary pyspark pyarrow fastparquet

# Copy the core trading system
COPY src/ ./src/
COPY kite_auth.py .

# Copy this service's code
COPY services/data-etl/ ./service/

WORKDIR /app/service

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
