FROM python:3.11-slim

WORKDIR /app

COPY requirements.frontend-docker.txt /app/requirements.frontend-docker.txt
RUN pip install --no-cache-dir --default-timeout=600 --retries 10 -r /app/requirements.frontend-docker.txt

COPY . /app

ENV API_URL=http://api:8000
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "frontend/Home.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
