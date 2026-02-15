# Option A: CLIP in Modal, search on Fly

**Fly** only stores vectors and runs **POST /search** (vector in → top results out). **Modal** runs the CLIP text encoder: user text → 512-dim vector → call Fly **POST /search** with that vector.

---

## Your end (Fly API)

### 1. Deploy the API

You already have the slim API (no CLIP, no sentence-transformers). Just deploy:

```bash
fly deploy
```

Builds stay fast. The API exposes:

- **POST /frames/modal** – Modal sends frame vectors (time, title, vector) → stored in MongoDB.
- **POST /search** – Accepts a **vector** + `top_k`, returns nearest frames (cosine similarity).

### 2. Fly secrets (already set)

Keep these on Fly:

- **MONGODB_URI** – Atlas connection string.
- **MONGODB_DB** – Database name (e.g. `TreeHacks2026`).
- **API_KEY** – Same value you use in Modal as `VECTOR_API_KEY`.

### 3. You don’t do anything else on Fly

No CLIP, no text embedding. Fly only stores and searches vectors.

---

## Modal end (CLIP + search)

### 1. Modal secrets

In the Modal dashboard, set:

| Secret            | Value |
|-------------------|--------|
| **VECTOR_API_URL** | `https://treehacks-vector-api.fly.dev` (no trailing slash) |
| **VECTOR_API_KEY** | Same as **API_KEY** on Fly |

### 2. Add CLIP to your Modal app

- Use the **same CLIP model** you use for frame embeddings (e.g. image encoder for frames, **text** encoder for search).
- In your Modal image, include CLIP (e.g. `sentence-transformers` or the same library you use for frame encoding).

Example image (if you use sentence-transformers for CLIP):

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests", "sentence-transformers")
)
```

### 3. Search-by-text function on Modal

Run CLIP **text** encoder on the query string, then call your Fly API **POST /search** with the resulting vector.

Example (paste into your Modal app; use your existing `app`/`stub` and image that has CLIP/sentence_transformers):

```python
import os
import requests
import modal

# CRITICAL: Use the SAME CLIP model (and thus same dim) as your frame encoder.
# clip-ViT-B-32 → 512 dimensions. If your frames use a different model, use its text side here.
CLIP_TEXT_MODEL = "sentence-transformers/clip-ViT-B-32"
EXPECTED_DIM = 512  # must match Atlas index numDimensions and your frame embeddings

# Use your existing Modal app and an image that includes sentence_transformers (or your CLIP lib)
app = modal.App("your-app-name")
image = modal.Image.debian_slim(python_version="3.11").pip_install("requests", "sentence-transformers")

@app.cls(image=image)
class SearchByText:
    @modal.enter()
    def load_model(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(CLIP_TEXT_MODEL)

    @modal.method()
    def run(self, query: str, top_k: int = 5) -> list:
        """Encode query with CLIP text encoder, call Fly POST /search, return top_k results."""
        vector = self.model.encode(query.strip() or " ", convert_to_numpy=True).tolist()
        if len(vector) != EXPECTED_DIM:
            raise ValueError(f"Query vector is {len(vector)} dims; index expects {EXPECTED_DIM}. Use the same CLIP model as your frame encoder.")
        url = os.environ["VECTOR_API_URL"].rstrip("/")
        key = os.environ.get("VECTOR_API_KEY", "")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["X-API-Key"] = key
        r = requests.post(f"{url}/search", json={"vector": vector, "top_k": top_k}, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json().get("results", [])
```

- **query** – User search text (e.g. `"dog running"`).
- **top_k** – Number of results (default 5).
- **Returns** – List of docs with `_id`, `time`, `title`, `score` (cosine similarity), etc.

### 4. Call it from your Modal app

From a Modal function, web endpoint, or local entrypoint:

```python
results = SearchByText().run.remote("sunset on the beach", top_k=5)
for r in results:
    print(r.get("time"), r.get("title"), r.get("score"))
```

---

## Flow summary

1. **User** types a search query (e.g. in your frontend or CLI).
2. **Modal** receives the query → runs CLIP **text** encoder → gets 512-dim vector.
3. **Modal** calls **POST https://treehacks-vector-api.fly.dev/search** with `{"vector": [...], "top_k": 5}` and **X-API-Key** header.
4. **Fly** runs vector search (cosine similarity) in MongoDB and returns the top 5 frames.
5. **Modal** (or your app) returns those results (time, title, score) to the user.

Fly stays small and fast; CLIP runs only on Modal (and can use GPU there if you want).

---

## If your text vector is not 512 dimensions

- **You must use the same model for text as for your frame embeddings.** If your Modal pipeline uses a different CLIP (or another encoder) for **images/frames**, that model may output a different size (e.g. 768). In that case:
  1. Find the **exact model name** you use for frame embeddings in Modal (e.g. `openai/clip-vit-base-patch32` or another CLIP).
  2. Use that model’s **text** side for search. With `sentence-transformers`, the same model name usually gives the same dimension for text and image.
  3. Set **EXPECTED_DIM** in the snippet above to that size (e.g. 512 for `clip-ViT-B-32`), and set your **Atlas index numDimensions** to the same value.
- **Check what you’re actually sending:** In Modal, log `len(vector)` before calling the API. If it’s not 512, switch to the CLIP text model that matches your frame encoder, or change your Atlas index and frame pipeline to 512 and use `sentence-transformers/clip-ViT-B-32` everywhere.
