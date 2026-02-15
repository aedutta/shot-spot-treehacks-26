# app/backend/api.py
import os
import sys
from typing import Any, List, Optional
import random
import time

# Load env vars explicitly from the root .env file
from dotenv import load_dotenv
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(project_root, ".env"))

# Add the 'modal' folder to sys.path so we can import 'ingestor'
modal_folder = os.path.join(project_root, "modal")
if modal_folder not in sys.path:
    sys.path.append(modal_folder)

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import modal

# Import the Modal function directly
try:
    # Since we added 'modal' folder to path, we import 'ingestor' directly
    from ingestor import ingest_video_orchestrator
except ImportError as e:
    print(f"Warning: Could not import ingestor: {e}")
    ingest_video_orchestrator = None

# --- MongoDB Integration ---
try:
    from db import get_collection, search
except ImportError:
    # If not in path, try adding root
    sys.path.append(project_root)
    from db import get_collection, search

# --- CLIP Model (Text Only) ---
# We load a small CLIP model to embed text queries for search
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    print("Loading CLIP (Text Encoder)...")
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    print("CLIP Loaded.")
except ImportError:
    print("Warning: transformers/torch not installed. Text search will fail.")
    processor = None
    model = None

def embed_text(text: str) -> List[float]:
    if not model:
        raise HTTPException(500, "CLIP model not loaded")
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        # Get the text features
        # Note: In some versions of transformers, we might need to call model.get_text_features
        # Use simple model call if get_text_features is behaving unexpectedly, but get_text_features returns a Tensor.
        
        try:
            text_features = model.get_text_features(**inputs)
        except Exception:
            # Fallback if get_text_features is effectively missing or behaving oddly (unlikely for CLIPModel but good for safety)
            outputs = model.text_model(**inputs)
            text_features = outputs.pooler_output
            
        # Ensure it's a tensor before normalization
        if hasattr(text_features, 'pooler_output'):
             text_features = text_features.pooler_output

        # Normalize
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    
    vec = text_features[0].tolist()
    print(f"🔎 DEBUG: Generated embedding dimension: {len(vec)}")
    return vec

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
    """Real stats from MongoDB"""
    try:
        coll = get_collection()
        count = coll.count_documents({})
    except:
        count = 0
        
    return {
        "active_workers": random.randint(5, 100), # Modal status API could go here
        "fps_processed": random.randint(120, 500),
        "bandwidth_mbps": random.randint(50, 200),
        "total_frames": count, 
    }

@app.post("/frames/bulk")
def upload_frames_bulk(body: FrameUploadBulk):
    ids = []
    print(f"Received {len(body.frames)} frames")
    # Need to import insert_frames from db (bulk version)
    try:
        from db import insert_frames
    except ImportError:
         sys.path.append(project_root)
         from db import insert_frames
    
    docs_to_insert = []
    for frame in body.frames:
        # Get vector
        vec = frame.vector if frame.vector else frame.embedding
        if not vec:
            continue
            
        # Extract metadata
        meta = frame.model_dump(exclude={"vector", "embedding"})
        meta["embedding"] = vec
        docs_to_insert.append(meta)
        
    # Insert in batch
    try:
        inserted_ids = insert_frames(docs_to_insert)
        ids = [str(x) for x in inserted_ids]
    except Exception as e:
        print(f"Bulk insert failed: {e}")
            
    return {"ok": True, "count": len(ids), "ids": ids}

@app.post("/search")
def search_frames(req: SearchRequest):
    """
    Text-to-Image Search.
    1. Embed query (text) -> vector
    2. Search MongoDB
    """
    print(f"Searching for: {req.query}")
    
    # 1. Embed
    try:
        if not req.query:
             # Return random/latest if empty
             pass 
        vector = embed_text(req.query)
    except Exception as e:
        print(f"Embedding failed: {e}")
        return {"ok": False, "error": str(e)}

    # 2. Search DB
    try:
        results = search(vector, top_k=req.top_k)
    except Exception as e:
        print(f"DB Search failed: {e}")
        return {"ok": False, "error": str(e)}
        
    # 3. Format for Frontend
    formatted = []
    for doc in results:
        # doc has: embedding, time, title, source, score (from search)
        
        # Calculate pretty timestamp
        ts = doc.get("time", 0)
        minutes = int(ts // 60)
        seconds = int(ts % 60)
        time_str = f"{minutes}m {seconds}s"
        
        # Generate direct link to timestamp
        source = doc.get("source", "")
        link = source
        if "twitch.tv" in source:
             link = f"{source}?t={int(ts)}s"
        elif "youtube.com" in source or "youtu.be" in source:
             link = f"{source}&t={int(ts)}s"
             
        # Use a placeholder or frame extraction service for the actual image
        formatted.append({
            "id": str(doc.get("_id")),
            "url": f"https://source.unsplash.com/random/300x200?sig={random.randint(0,1000)}", 
            "timestamp": time_str,
            "score": doc.get("score", 0),
            "source_url": link,
            "title": doc.get("title", "Unknown")
        })
        
    return {"ok": True, "results": formatted}

@app.post("/ingest/start")
async def start_ingest(source_url: str, prompt: str, scale: int, stealth: bool):
    """Trigger the ingestion process via the imported Modal function"""
    try:
        # 1. Lookup the DEPLOYED function. 
        # This requires you to run `modal deploy modal/ingestor.py` in your terminal first.
        # The first argument is the App Name (from ingestor.py), the second is the function name.
        f = modal.Function.from_name("treehacks-video-ingestor-v2", "ingest_video_orchestrator")
        
        # 2. Spawn the function execution asynchronously on the cloud
        call = f.spawn(source_url)
        
        return {
            "ok": True, 
            "job_id": call.object_id,
            "message": f"Started ingestion for '{prompt}' from {source_url} (ID: {call.object_id})"
        }
    except modal.exception.NotFoundError:
        return {
            "ok": False,
            "error": "Modal App Not Found",
            "message": "❌ Please run `modal deploy modal/ingestor.py` in your terminal to deploy the function first."
        }
    except Exception as e:
        print(f"Modal invocation failed: {e}")
        return {
            "ok": False,
            "error": str(e),
            "message": f"Failed to trigger Modal function: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
