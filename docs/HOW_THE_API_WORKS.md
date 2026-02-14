# How This API Works (POST, Search, etc.)

Plain-language explanation of the API and each endpoint.

---

## What is an API?

An **API** (Application Programming Interface) is a way for one program to talk to another over the **internet** using **HTTP**. Your API is a small server that:

- **Listens** at a URL (e.g. `http://localhost:8000`).
- **Exposes routes** (paths like `/health`, `/frames`, `/search`).
- For each route you send a **method** (GET or POST) and sometimes a **body** (JSON). The API does something (e.g. save to the DB, run a search) and **returns** a JSON response.

So: **client** (browser, your model, curl) sends a **request** → **API** runs code → **API** sends back a **response**.

---

## GET vs POST (quick)

| Method | Typical use | Body? |
|--------|-------------|--------|
| **GET** | “Give me something” (no change to data) | Usually no |
| **POST** | “Here is data, do something with it” (create or run an action) | Yes (e.g. JSON) |

This API uses **GET** for one simple check and **POST** for uploading and searching.

---

## The Endpoints

### 1. GET /health

**What it does:** Checks that the server is up and can talk to MongoDB.

**Request:** No body. Optional: header `X-API-Key` (not required for health).

**Example:**
```http
GET http://localhost:8000/health
```

**Response (success):**
```json
{"status": "ok", "db": "connected"}
```

**Response (DB down):** Status 503 and an error message.

**Use it for:** Monitoring, or to confirm the API and DB are working before calling other endpoints.

---

### 2. POST /frames (upload one frame)

**What it does:** Saves **one** frame (one embedding + optional metadata) into MongoDB.

**Request:**
- **Method:** POST
- **URL:** `http://localhost:8000/frames`
- **Headers:** `Content-Type: application/json` and, if you set `API_KEY` in `.env`, `X-API-Key: <your-key>`
- **Body (JSON):**
  - **embedding** (required): list of numbers, e.g. `[0.1, 0.2, 0.3, 0.4, 0.5]`
  - **frame_id** (optional): string
  - **source** (optional): string
  - **metadata** (optional): object
  - Any other fields you want stored with the document

**Example:**
```json
POST http://localhost:8000/frames
Content-Type: application/json
X-API-Key: your-secret-key

{
  "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
  "frame_id": "frame-001",
  "source": "modal"
}
```

**Response (success):**
```json
{"ok": true, "id": "65a1b2c3d4e5f6789012345"}
```

`id` is the MongoDB `_id` of the new document. If the key is wrong or missing (when `API_KEY` is set), you get **401**.

---

### 3. POST /frames/bulk (upload many frames)

**What it does:** Saves **multiple** frames in one request. Same as calling POST /frames many times, but in a single HTTP call.

**Request:**
- **Method:** POST
- **URL:** `http://localhost:8000/frames/bulk`
- **Headers:** Same as POST /frames (`Content-Type`, `X-API-Key` if needed)
- **Body (JSON):**
  - **frames**: array of objects; each object has the same shape as for POST /frames (e.g. `embedding`, optional `frame_id`, `source`, `metadata`)

**Example:**
```json
POST http://localhost:8000/frames/bulk
Content-Type: application/json
X-API-Key: your-secret-key

{
  "frames": [
    { "embedding": [0.1, 0.2, 0.3, 0.4, 0.5], "frame_id": "a" },
    { "embedding": [0.2, 0.3, 0.4, 0.5, 0.6], "frame_id": "b" }
  ]
}
```

**Response (success):**
```json
{"ok": true, "count": 2, "ids": ["65a1...", "65a2..."]}
```

`ids` are the MongoDB `_id`s of the inserted documents, in the same order as `frames`.

---

### 4. POST /search (vector search)

**What it does:** Takes a **query vector**, runs **vector search** in MongoDB (Atlas Vector Search), and **returns** the most similar stored documents, each with a **score**.

**Request:**
- **Method:** POST
- **URL:** `http://localhost:8000/search`
- **Headers:** `Content-Type: application/json` and, if you set `API_KEY`, `X-API-Key: <your-key>`
- **Body (JSON):**
  - **vector** (required): list of numbers (same length as your stored embeddings)
  - **top_k** (optional): how many results to return (default 10)

**Example:**
```json
POST http://localhost:8000/search
Content-Type: application/json
X-API-Key: your-secret-key

{
  "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
  "top_k": 5
}
```

**Response (success):**
```json
{
  "ok": true,
  "results": [
    {"_id": "...", "score": 0.99, "frame_id": "frame-001", ...},
    {"_id": "...", "score": 0.85, "frame_id": "frame-002", ...}
  ]
}
```

Results are sorted by **score** (highest first). **score** is the similarity between the query vector and that document’s embedding. Each item in `results` is a stored document (with `_id`, `frame_id`, etc.); long embeddings may be truncated in the response.

---

## Summary Table

| Method | Endpoint       | Purpose                    | Body |
|--------|----------------|----------------------------|------|
| GET    | /health        | Check API + DB connection  | No   |
| POST   | /frames        | Upload one embedding       | `{ "embedding": [...], "frame_id": "..." }` |
| POST   | /frames/bulk   | Upload many embeddings     | `{ "frames": [ {...}, {...} ] }` |
| POST   | /search        | Search by vector, get similar items + scores | `{ "vector": [...], "top_k": 10 }` |

---

## Flow in One Sentence

- **POST /frames** and **POST /frames/bulk**: you **send** embeddings (and optional metadata) in the **body** → the API **saves** them to MongoDB → the API **returns** `ok` and the new document id(s).
- **POST /search**: you **send** a query **vector** in the **body** → the API **runs** vector search on the DB → the API **returns** the best matches and their **scores** in the **response**.

All of that is implemented in **api.py** (and **db.py** for the actual database operations).

---

## Search not working? Checklist

1. **Use POST, not GET** — Search is `POST /search`, with a JSON body.
2. **Body shape** — Send `{"vector": [0.1, 0.2, ...], "top_k": 10}`. The key must be **"vector"** (not "embedding").
3. **Vector length** — Must match your Atlas index `numDimensions` (e.g. 5). Wrong length → 400 error.
4. **API key** — If `API_KEY` is set in `.env`, send header: `X-API-Key: <your-api-key>`. Missing/wrong key → **401 Unauthorized**.
5. **Content-Type** — Send `Content-Type: application/json`.
6. **Working example (curl):**
   ```bash
   curl -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -H "X-API-Key: YOUR_KEY" \
     -d '{"vector": [0.1, 0.2, 0.3, 0.4, 0.5], "top_k": 5}'
   ```
