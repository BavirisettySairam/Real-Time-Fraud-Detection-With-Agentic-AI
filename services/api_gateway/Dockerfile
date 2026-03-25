FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# libgomp1 = OpenMP for LightGBM/XGBoost, curl = healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.api-docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.api-docker.txt

COPY services/ ./services/
COPY ml/ ./ml/
COPY models/ ./models/
COPY .env* ./

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "services.api_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]