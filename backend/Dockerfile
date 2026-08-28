FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY models/ ../models/

EXPOSE 8000
# Shell form (not exec-array) so $PORT expands -- Render injects PORT
# dynamically; local docker-compose leaves it unset, falling back to 8000.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
