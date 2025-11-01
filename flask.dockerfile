FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/ /app/
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

COPY src/predictions/dataset /app/predictions/dataset

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app.main:app"]
