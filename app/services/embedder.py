import modal
import os
import sys

# Add project root to path to allow imports
# We need to go up 3 levels to reach the workspace root (where 'modal_infra' might be, or 'app')
# app/services/embedder.py -> services -> app -> treehacks-2026
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Try importing from local services first, then fallback to modal_infra
try:
    from app.services.ingestor import image, MODEL_REPO
except ImportError:
    from modal_infra.ingestor import image, MODEL_REPO

app = modal.App("clip-embedder-service")

@app.cls(
    image=image,
    gpu="A10G",
    keep_warm=1,
    container_idle_timeout=300
)
class EmbedderService:
    def __enter__(self):
        from transformers import CLIPProcessor, CLIPModel
        import torch
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model... Device: {self.device}")
        
        # Load model (cached in image)
        # Allow env override
        model_repo = os.getenv("CLIP_MODEL_REPO", MODEL_REPO)
        
        self.model = CLIPModel.from_pretrained(model_repo).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_repo)
    
    @modal.method()
    def embed_text(self, text: str):
        import torch
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            return emb.cpu().numpy()[0].tolist()

@app.local_entrypoint()
def main(text: str):
    service = EmbedderService()
    print(f"Embedding '{text}'...")
    vector = service.embed_text.remote(text)
    print(f"Vector (first 5): {vector[:5]}...")
