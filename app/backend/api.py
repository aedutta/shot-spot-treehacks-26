# app/backend/api.py
import os
from typing import Any, List, Optional
import random
import time

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

# Simple mock DB for demo purposes
class MockDB:
    def __init__(self):
        self.frames = []
        self.id_counter = 0

    def insert_frame(self, frame_data):
        self.id_counter += 1
        frame_data["_id"] = str(self.id_counter)
        self.frames.append(frame_data)
        return str(self.id_counter)

    def search(self, vector, top_k=10):
        # In a real app, this would use vector similarity
        # For demo, just return random frames
        return random.sample(self.frames, min(len(self.frames), top_k))

db = MockDB()

app = FastAPI(
    title="Ingest.ai API",
    description="Just-in-Time Dataset Factory API",
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/Response models ---

class FrameUpload(BaseModel):
    model_config = ConfigDict(extra="allow")
    embedding: Optional[List[float]] = None
    vector: Optional[List[float]] = None
    source: Optional[str] = None
    timestamp: Optional[float] = None
    title: Optional[str] = None

class FrameUploadBulk(BaseModel):
    frames: List[FrameUpload]

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

# --- Endpoints ---

@app.get("/")
def root():
    return {"status": "ok", "message": "Ingest.ai API Operational"}

@app.get("/stats")
def stats():
    """Mock stats for the dashboard"""
    return {
        "active_workers": random.randint(5, 100),
        "fps_processed": random.randint(120, 500),
        "bandwidth_mbps": random.randint(50, 200),
        "total_frames": len(db.frames) + 1240, # Fake baseline
    }

@app.post("/frames/bulk")
def upload_frames_bulk(body: FrameUploadBulk):
    ids = []
    print(f"Received {len(body.frames)} frames")
    for frame in body.frames:
        # Simulate processing
        data = frame.model_dump()
        doc_id = db.insert_frame(data)
        ids.append(doc_id)
    return {"ok": True, "count": len(ids), "ids": ids}

@app.post("/search")
def search_frames(req: SearchRequest):
    # Mock search response
    results = [
        {
            "id": f"frame-{i}",
            "url": f"https://source.unsplash.com/random/300x200?sig={i}", # Placeholder images
            "timestamp": random.uniform(0, 600),
            "score": random.uniform(0.7, 0.99),
            "source_url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
        }
        for i in range(req.top_k)
    ]
    return {"ok": True, "results": results}

@app.post("/ingest/start")
def start_ingest(source_url: str, prompt: str, scale: int, stealth: bool):
    """Trigger the ingestion process (Mock)"""
    return {
        "ok": True, 
        "job_id": f"job-{random.randint(1000, 9999)}",
        "message": f"Started ingestion for '{prompt}' from {source_url} (Scale: {scale}, Stealth: {stealth})"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
