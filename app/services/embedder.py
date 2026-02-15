import modal
import os
import sys

# Add project root to path to allow imports from ingestor
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modal_infra.ingestor import image, MODEL_REPO

app = modal.App("clip-embedder")

@app.function(image=image, keep_warm=1)
def embed_text(text: str):
    from transformers import CLIPProcessor, CLIPModel
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model... Device: {device}")
    
    # Load model (cached in image)
    model = CLIPModel.from_pretrained(MODEL_REPO).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_REPO)
    
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.cpu().numpy()[0].tolist()

@app.local_entrypoint()
def main():
    print(embed_text.remote("hello world"))
