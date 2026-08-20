# ============================================================
# Dockerfile — kafka-consumer service
# ============================================================
FROM python:3.10-slim

# ---- OS-level dependencies ---------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Python environment ------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python dependencies directly (no requirements.txt needed)
RUN pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        confluent-kafka \
        pymongo \
        "pymongo[srv]" \
        python-dotenv

# ---- Application -------------------------------------------
WORKDIR /app/service

# Copy only the service source (build context is repo root)
COPY services/kafka-consumer/ ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
