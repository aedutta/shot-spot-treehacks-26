# Making search better for general queries

When the search query is very general (e.g. "dog", "people", "outdoor"), results can feel weak. Here are practical ways to improve the model, embeddings, and search.

---

## 1. Improve the query before embedding (Modal)

**Idea:** Turn a short, general query into a richer description that matches how CLIP was trained (often on image captions like "a photo of a dog running").

- **Simple:** Prefix the query, e.g. `"a photo of " + query` or `"a video frame of " + query`, then embed that.
- **Stronger:** Use an LLM (e.g. in Modal) to expand the query: e.g. "dog" → "a dog, dog running, dog playing, puppy, canine". Embed each phrase and average the vectors, or embed the single best expansion.

Example (in Modal, before encoding):

```python
def expand_query(query: str) -> str:
    """Make general query more descriptive for CLIP."""
    query = query.strip().lower()
    if not query:
        return "a photo"
    # Simple prefix often helps CLIP
    if not query.startswith(("a ", "an ", "the ")):
        query = f"a photo of {query}"
    return query

# Then: vector = model.encode(expand_query(user_query), ...)
```

---

## 2. Use more candidates in vector search (API / db.py)

Atlas vector search has **numCandidates**: how many neighbors to consider before returning top_k. Larger = better recall for general queries, slightly slower.

**Current default** in `db.search()`: `max(limit * 20, 100)` (e.g. top_k=5 → 100 candidates).

**Change:** Pass a larger `num_candidates` when calling search. In [db.py](db.py), `search()` already accepts `num_candidates`. You can expose it via the API (e.g. optional `num_candidates` in the POST /search body) and set it to 200 or 500 for general queries.

In **api.py** you could add:

```python
# In SearchRequest model:
num_candidates: Optional[int] = None  # default None → db uses its default

# In search_vectors():
results = search(req.vector, top_k=req.top_k, num_candidates=req.num_candidates)
```

Then from Modal send `{"vector": [...], "top_k": 5, "num_candidates": 300}` when the query is general.

---

## 3. Rerank with a cross-encoder (Modal)

**Idea:** Get more candidates (e.g. top_k=20), then rerank with a model that scores (query, candidate) pairs. Cross-encoders are better at relevance than the single-vector similarity.

- In Modal: call Fly **POST /search** with `top_k=20` (or 30).
- In Modal: run a **cross-encoder** (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` or a CLIP-based reranker) on each (query, candidate_title) or (query, candidate_snippet).
- Return the top 5 after reranking.

This adds latency and compute in Modal but improves quality for vague queries.

---

## 4. Stronger embedding model (same 512 dims)

If you stay with 512 dimensions, you can still try a different CLIP-style model that might align better with your content:

- **Same size, different training:** e.g. `sentence-transformers/clip-ViT-B-32` vs `openai/clip-vit-base-patch32` (both 512). Try the other if one gives weak results.
- **Larger model (requires re-indexing):** e.g. **ViT-L/14** (768 dims). Better quality but you must:
  - Re-create the Atlas index with **numDimensions: 768**.
  - Re-embed all frames with the 768-dim model and re-upload.
  - Use the same 768-dim model for text in Modal.

---

## 5. Hybrid: filter then vector search

If your frames have **title**, **time**, or other metadata:

- **Narrow first:** e.g. filter by time range or title containing a keyword (if the user added filters).
- **Then vector search** within that subset (Atlas supports `filter` in `$vectorSearch`).

That doesn’t fix “general query” directly but can make results more relevant when the user has some context (e.g. “in the first 5 minutes”).

---

## Quick wins (no re-index)

1. **Query phrasing** in Modal: e.g. `"a photo of " + query` before encoding.
2. **Larger numCandidates**: expose and use 200–500 for general queries.
3. **Rerank** in Modal: get top 20 from Fly, rerank with a small cross-encoder, return top 5.

These keep your current 512-dim index and frame embeddings and still improve results for general search.
