"""
API to connect to the vector DB: upload frames (embeddings + metadata) and search by vector.
Your model (e.g. on Modal) can POST to /frames or /frames/bulk to send embeddings to the DB.

Run: uvicorn api:app --reload
Optional: set API_KEY in .env and send header "X-API-Key: <API_KEY>" so only your model can upload.
"""

import os
from typing import Any, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from db import get_client, get_collection, insert_frame, search


app = FastAPI(
    title="Vector Store API",
    description="Upload embeddings and search by similarity.",
)

# Allow Modal (and other servers) to call this API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    """Connect to MongoDB on startup and log status."""
    import os
    try:
        client = get_client()
        client.admin.command("ping")
        db_name = os.environ.get("MONGODB_DB", "treehacks")
        print(f"Connected to MongoDB (db: {db_name})")
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        print("Check .env: MONGODB_URI and MONGODB_DB must be set.")


# --- Request/Response models ---

class FrameUpload(BaseModel):
    """One frame: use 'embedding' or 'vector' (Modal uses 'vector') + optional metadata."""
    model_config = ConfigDict(extra="allow")
    embedding: Optional[List[float]] = None
    vector: Optional[List[float]] = None
    frame_id: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[dict] = None

    def get_embedding(self) -> List[float]:
        if self.vector is not None:
            return self.vector
        if self.embedding is not None:
            return self.embedding
        raise ValueError("Provide either 'embedding' or 'vector'")


class FrameUploadBulk(BaseModel):
    """Multiple frames to upload."""
    frames: List[FrameUpload]


class ModalFrameResult(BaseModel):
    """One frame from Modal VideoIngestor: time, vector, title."""
    time: float
    vector: List[float]
    title: str


class ModalUploadBody(BaseModel):
    """Bulk upload in Modal's format: list of {time, vector, title} from process_stream."""
    results: List[ModalFrameResult]


class SearchRequest(BaseModel):
    """Search by vector."""
    vector: List[float]
    top_k: int = 10


# --- Optional API key (if API_KEY is set in .env, upload/search require X-API-Key header) ---

def _require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    api_key = os.environ.get("API_KEY")
    if not api_key:
        return  # no key configured, allow all
    if x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# --- Endpoints ---

@app.get("/")
def root():
    """Always returns 200 if the app is running. No DB check."""
    return {"status": "ok", "message": "Vector store API"}


@app.get("/frames/count")
def frames_count():
    """Return how many documents are in the frames collection. Use to verify uploads."""
    try:
        coll = get_collection()
        n = coll.count_documents({})
        db_name = os.environ.get("MONGODB_DB", "treehacks")
        return {"ok": True, "count": n, "database": db_name, "collection": "frames"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/health")
def health():
    """Returns 200 within ~5s; check body for db status. Uses a short timeout so it doesn't hang."""
    try:
        client = get_client(server_selection_timeout_ms=5000)
        client.admin.command("ping")
        return {"status": "ok", "db": "connected"}
    except Exception as e:
        return {"status": "ok", "db": "disconnected", "error": str(e)}


def _frame_extra(frame: FrameUpload) -> dict:
    """Build extra kwargs for insert_frame (exclude embedding/vector, drop None)."""
    d = frame.model_dump(exclude={"embedding", "vector"}, exclude_none=True)
    return d


@app.post("/frames")
def upload_frame(frame: FrameUpload, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Upload one frame. Send 'embedding' or 'vector' (Modal uses 'vector') + optional metadata."""
    _require_api_key(x_api_key)
    try:
        vec = frame.get_embedding()
        doc_id = insert_frame(vec, **_frame_extra(frame))
        return {"ok": True, "id": str(doc_id)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/frames/bulk")
def upload_frames_bulk(body: FrameUploadBulk, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Upload multiple frames. Each item can use 'embedding' or 'vector'."""
    _require_api_key(x_api_key)
    ids = []
    for frame in body.frames:
        vec = frame.get_embedding()
        doc_id = insert_frame(vec, **_frame_extra(frame))
        ids.append(str(doc_id))
    return {"ok": True, "count": len(ids), "ids": ids}


@app.post("/frames/modal")
def upload_modal_results(body: ModalUploadBody, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Accept Modal VideoIngestor output directly: list of {time, vector, title}. Stores only those fields (vector as embedding)."""
    _require_api_key(x_api_key)
    n = len(body.results)
    print(f"[frames/modal] received {n} frame(s)")
    ids = []
    for r in body.results:
        doc_id = insert_frame(r.vector, time=r.time, title=r.title)
        ids.append(str(doc_id))
    print(f"[frames/modal] inserted {len(ids)} doc(s) into MongoDB")
    return {"ok": True, "count": len(ids), "ids": ids}


@app.post("/search")
def search_vectors(req: SearchRequest, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Search for nearest vectors. Body must be: {"vector": [0.1, 0.2, ...], "top_k": 10}. Send X-API-Key header if API_KEY is set."""
    _require_api_key(x_api_key)
    try:
        results = search(req.vector, top_k=req.top_k)
        # Convert ObjectId and other non-JSON types for response
        out = []
        for doc in results:
            d = dict(doc)
            d["_id"] = str(d.get("_id", ""))
            if "embedding" in d and len(d["embedding"]) > 20:
                d["embedding"] = d["embedding"][:10] + ["..."]  # truncate long vectors in response
            out.append(d)
        return {"ok": True, "results": out}
    except Exception as e:
        err = str(e)
        if "vector" in err.lower() or "index" in err.lower() or "dimension" in err.lower():
            raise HTTPException(status_code=400, detail=f"{err} (Check: vector length must match your index numDimensions, e.g. 5)")
        raise HTTPException(status_code=400, detail=err)


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
