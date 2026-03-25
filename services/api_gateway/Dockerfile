FROM python:3.11-slim

# Prevent Python from writing pyc files & buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime libs required by LightGBM and container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (cache layer)
COPY requirements.api-docker.txt .

# Install dependencies (keep cache for speed)
RUN pip install --upgrade pip && \
    pip install -r requirements.api-docker.txt

# Copy only necessary code
COPY services/ ./services/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run API
CMD ["python", "-m", "uvicorn", "services.api_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]