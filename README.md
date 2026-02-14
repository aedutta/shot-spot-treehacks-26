# treehacks-2026

**Vector store** on MongoDB Atlas: store embeddings (e.g. from video frames) and search by similarity using Atlas Vector Search.

## Features

- **insert_frame** — Store an embedding vector and optional metadata.
- **search** — Vector search (nearest neighbors) via Atlas Vector Search index.

Requires a **vector index** on your collection. See **[docs/ATLAS_VECTOR_INDEX.md](docs/ATLAS_VECTOR_INDEX.md)**.

## API (upload & search over HTTP)

Run the API server:

```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

Then open **http://localhost:8000/docs** for interactive docs.

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check DB connection |
| POST | `/frames` | Upload one frame (embedding + optional `frame_id`, `source`, `metadata`) |
| POST | `/frames/bulk` | Upload multiple frames |
| POST | `/search` | Search by vector (body: `{"vector": [0.1, 0.2, ...], "top_k": 10}`) |

**Example — upload one frame:**

```bash
curl -X POST http://localhost:8000/frames \
  -H "Content-Type: application/json" \
  -d '{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5], "frame_id": "my-frame-1"}'
```

**Example — search:**

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3, 0.4, 0.5], "top_k": 5}'
```