# Deploy the vector store API (e.g. Fly.io, Railway, Render)
FROM python:3.11-slim

# Up-to-date CA certs and OpenSSL for MongoDB Atlas TLS
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Cloud: bind to 0.0.0.0 and use PORT from platform (Fly, Railway, etc.)
ENV HOST=0.0.0.0
ENV PORT=8000
EXPOSE 8000

CMD ["python", "api.py"]
