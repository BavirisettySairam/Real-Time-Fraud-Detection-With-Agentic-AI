FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.frontend-docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.frontend-docker.txt

COPY frontend/ ./frontend/
COPY models/ ./models/
COPY .env* ./

ENV API_URL=http://api:8000
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "frontend/Home.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
