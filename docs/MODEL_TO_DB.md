# Connect your model (e.g. Modal) to the DB

Your model runs elsewhere (e.g. Modal); it sends embeddings to this API, which writes to MongoDB.

## 1. Run the API

Deploy or run the API so it’s reachable (e.g. `https://your-api.com` or `http://localhost:8000` for local).

## 2. Optional: set an API key

In the API’s `.env`:

```env
API_KEY=your-secret-key
```

Then every request to **POST /frames**, **POST /frames/bulk**, and **POST /search** must include:

```http
X-API-Key: your-secret-key
```

Your model should read this key from its own secrets (e.g. Modal secrets) and send it in the header. If you don’t set `API_KEY`, no key is required.

## 3. From your model: send data to the DB

**Upload one frame (one embedding):**

```python
import os
import requests

API_URL = os.environ.get("VECTOR_API_URL", "http://localhost:8000")  # or your deployed URL
API_KEY = os.environ.get("VECTOR_API_KEY", "")  # set if you use API_KEY

def send_frame(embedding: list, frame_id: str = None, source: str = None, metadata: dict = None):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    body = {"embedding": embedding}
    if frame_id is not None:
        body["frame_id"] = frame_id
    if source is not None:
        body["source"] = source
    if metadata is not None:
        body["metadata"] = metadata
    r = requests.post(f"{API_URL}/frames", json=body, headers=headers)
    r.raise_for_status()
    return r.json()  # {"ok": true, "id": "..."}
```

**Upload many frames (bulk):**

```python
def send_frames(frames: list):
    # frames = [{"embedding": [...], "frame_id": "f1"}, ...]
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    r = requests.post(f"{API_URL}/frames/bulk", json={"frames": frames}, headers=headers)
    r.raise_for_status()
    return r.json()  # {"ok": true, "count": N, "ids": [...]}
```

**Modal VideoIngestor (CLIP): use POST /frames/modal**

Your Modal app returns `[{"time": t, "vector": v.tolist(), "title": title}, ...]`. Send that **directly** to the API:

```python
import os
import requests

API_URL = os.environ.get("VECTOR_API_URL", "http://localhost:8000")  # or your deployed URL
API_KEY = os.environ.get("VECTOR_API_KEY", "")  # Modal secret

def send_modal_results(results: list):
    """results = list of {time, vector, title} from VideoIngestor._embed_batch / process_stream."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    r = requests.post(f"{API_URL}/frames/modal", json={"results": results}, headers=headers)
    r.raise_for_status()
    return r.json()  # {"ok": true, "count": N, "ids": [...]}
```

**From your Modal `process_stream`:** after you have the list of `{time, vector, title}` per video, flatten and send in chunks (e.g. 100 at a time) to avoid huge requests:

```python
# Inside your Modal app, after processing:
all_results = list(ingestor.process_stream.map(urls))  # list of lists
for video_results in all_results:
    for i in range(0, len(video_results), 100):
        chunk = video_results[i : i + 100]
        send_modal_results(chunk)
```

**CLIP ViT-B/32** produces **512-dimensional** vectors. Create an Atlas vector index with **numDimensions: 512** (see docs/ATLAS_VECTOR_INDEX.md).

**Other models: use /frames or /frames/bulk**

```python
# You can send "vector" instead of "embedding" (same thing)
send_frame(embedding, frame_id=f"modal-{i}", source="modal")
# Or bulk:
send_frames([{"vector": emb, "frame_id": f"modal-{i}"} for i, emb in enumerate(embeddings)])
```

## 4. Search from your model (optional)

```python
def search_similar(vector: list, top_k: int = 10):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    r = requests.post(f"{API_URL}/search", json={"vector": vector, "top_k": top_k}, headers=headers)
    r.raise_for_status()
    return r.json()["results"]  # list of {"score": ..., "frame_id": ..., ...}
```

## Summary

| Step | What to do |
|------|------------|
| Run API | Deploy or run `uvicorn api:app` so the model can reach it. |
| API key (optional) | Set `API_KEY` in the API’s `.env`; model sends `X-API-Key` header. |
| Model → DB | Model calls **POST /frames** or **POST /frames/bulk** with embeddings (+ frame_id, source, metadata). |
| Embedding size | Vectors must match your Atlas index `numDimensions`. **CLIP ViT-B/32 → 512.** |

Your model never needs MongoDB credentials; it only needs the API URL and, if you set it, the API key.
