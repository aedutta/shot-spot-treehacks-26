# Header and cURL for Modal → API → MongoDB

## Header required

So your API accepts the call and forwards to MongoDB, send this header on every POST to `/frames`, `/frames/bulk`, `/frames/modal`, and `/search`:

| Header        | Value |
|---------------|--------|
| **X-API-Key** | Same value as `API_KEY` in your API's `.env` (and in Fly secrets) |

If you don't set `API_KEY` in `.env` / Fly secrets, you can omit the header (all requests are allowed).

---

## Ready-to-run cURL (upload to MongoDB via API)

Replace `YOUR_API_URL` (e.g. `https://treehacks-vector-api.fly.dev`) and `YOUR_API_KEY` (the value of `API_KEY` in your `.env`).

**POST /frames/modal** (Modal-style: time, vector, title):

```bash
curl -X POST "YOUR_API_URL/frames/modal" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"results": [{"time": 0.0, "vector": [0.1, 0.2, 0.3, 0.4, 0.5], "title": "My Video"}, {"time": 1.0, "vector": [0.2, 0.3, 0.4, 0.5, 0.6], "title": "My Video"}]}'
```

**Example with real URL and placeholder key:**

```bash
curl -X POST "https://treehacks-vector-api.fly.dev/frames/modal" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"results": [{"time": 0.0, "vector": [0.1, 0.2, 0.3, 0.4, 0.5], "title": "My Video"}]}'
```

Success: `{"ok":true,"count":1,"ids":["..."]}` — those documents are in MongoDB.

---

## Another call: POST /search (vector search)

Same headers. Body: a vector (same length as your index, e.g. 5) and optional `top_k`. Returns nearest documents from MongoDB.

```bash
curl -X POST "YOUR_API_URL/search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"vector": [0.1, 0.2, 0.3, 0.4, 0.5], "top_k": 5}'
```

**Example:**

```bash
curl -X POST "https://treehacks-vector-api.fly.dev/search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"vector": [0.1, 0.2, 0.3, 0.4, 0.5], "top_k": 5}'
```

Success: `{"ok":true,"results":[...]}` — each result has `_id`, `embedding` (truncated), and any stored fields (e.g. `time`, `title`).

---

## In Modal

In your Modal app, set **secrets**:

- `VECTOR_API_URL` = your API base URL (e.g. `https://treehacks-vector-api.fly.dev`)
- `VECTOR_API_KEY` = same as `API_KEY` in your `.env`

Then in code, send the header on every request:

```python
headers = {"Content-Type": "application/json", "X-API-Key": os.environ["VECTOR_API_KEY"]}
requests.post(f"{os.environ['VECTOR_API_URL']}/frames/modal", json={"results": results}, headers=headers)
```
