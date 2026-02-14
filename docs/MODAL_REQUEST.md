# Request to implement in Modal

**POST /frames/modal** accepts the exact shape your VideoIngestor returns: `{"results": [{"time", "vector", "title"}, ...]}`.

---

## 1. cURL (for manual testing)

Replace `YOUR_API_URL` and `YOUR_API_KEY` with your deployed API URL and the key from your `.env`.

```bash
curl -X POST "YOUR_API_URL/frames/modal" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"results": [{"time": 0.0, "vector": [0.1, 0.2, 0.3, 0.4, 0.5], "title": "My Video"}, {"time": 1.0, "vector": [0.2, 0.3, 0.4, 0.5, 0.6], "title": "My Video"}]}'
```

**Success response:** `{"ok":true,"count":2,"ids":["...","..."]}`

---

## 2. Python for Modal (copy into your app)

Add **`requests`** to your Modal image:

```python
# In your image definition, add to .uv_pip_install():
.uv_pip_install(
    ...
    "requests",
)
```

**Secrets in Modal:** set `VECTOR_API_URL` and `VECTOR_API_KEY` (same value as `API_KEY` in your API `.env`).

**Helper + usage:**

```python
import os
import requests

def send_to_vector_db(results: list) -> dict:
    """Send list of {time, vector, title} to the vector API. Call from Modal."""
    api_url = os.environ.get("VECTOR_API_URL", "http://localhost:8000").rstrip("/")
    api_key = os.environ.get("VECTOR_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    r = requests.post(f"{api_url}/frames/modal", json={"results": results}, headers=headers)
    r.raise_for_status()
    return r.json()
```

**From `process_stream`** — after you have `results` (list of `{time, vector, title}`), send in chunks:

```python
CHUNK_SIZE = 100
for i in range(0, len(results), CHUNK_SIZE):
    chunk = results[i : i + CHUNK_SIZE]
    send_to_vector_db(chunk)
```

**From `main()`** — after `process_stream.map`, flatten and send:

```python
all_results = list(ingestor.process_stream.map(urls))
for video_results in all_results:
    for i in range(0, len(video_results), 100):
        send_to_vector_db(video_results[i : i + 100])
```

---

## 3. Request shape (reference)

| Field    | Type   | Required | Description                    |
|----------|--------|----------|--------------------------------|
| `results`| array  | yes      | List of frame objects         |
| `results[].time`  | number | yes | Timestamp (e.g. seconds)       |
| `results[].vector`| array  | yes | Embedding (CLIP = 512 floats)  |
| `results[].title` | string | yes | Video title                    |

**Response:** `{"ok": true, "count": N, "ids": ["_id1", "_id2", ...]}`

---

## 4. CLIP dimension

Your CLIP model outputs **512-dimensional** vectors. Create an Atlas vector index with **numDimensions: 512** (see ATLAS_VECTOR_INDEX.md). The test above used 5 dimensions only to match an existing test index.
