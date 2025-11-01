FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir prefect==3.4.25

COPY src/ /app/

EXPOSE 4200

ENV PREFECT_LOGGING_LEVEL=INFO

CMD ["prefect", "server", "start", "--host", "0.0.0.0", "--port", "4200"]
