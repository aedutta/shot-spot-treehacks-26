import modal
import os
import sys

# Add project root to path to allow imports from ingestor
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import shared configuration.
#
# Modal flattens this file to /root/embedder.py inside the container, so the
# `modal_infra` package does not exist there and the package-qualified import
# fails at container start (crash-looping every call). `ingestor` is mounted
# alongside it, so fall back to the sibling import when the package form is
# unavailable. Local runs still take the first branch.
try:
    from modal_infra.ingestor import image, MODEL_REPO
except ModuleNotFoundError:
    from ingestor import image, MODEL_REPO

app = modal.App("clip-embedder")

# Modal no longer installs FastAPI implicitly for @modal.fastapi_endpoint, so it
# has to be declared. Extending the shared image here rather than adding it to
# ingestor.py keeps the ingestor's image unchanged.
embedder_image = image.pip_install("fastapi[standard]")

@app.cls(
    image=embedder_image,
    gpu="A10G", # Use GPU for fast embedding
    min_containers=1, # Keep at least one instance warm
    scaledown_window=300 # Keep container alive for 5 minutes of inactivity
)
class CLIPEmbedder:
    # Modal no longer invokes a bare __enter__ as the container lifecycle hook;
    # it must be marked with @modal.enter(). Without this the model and
    # processor are never loaded and every call fails with AttributeError.
    @modal.enter()
    def load(self):
        from transformers import CLIPProcessor, CLIPModel
        import torch
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚡ Loading CLIP model on {self.device}...")
        
        # Load model (cached in image)
        self.model = CLIPModel.from_pretrained(MODEL_REPO).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL_REPO)
        print("✅ CLIP model ready.")

    @modal.method()
    def embed_text(self, text: str):
        import torch
        import numpy as np

        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            return emb.cpu().numpy()[0].tolist()

    @modal.fastapi_endpoint(method="POST", label="embed")
    def web_embed(self, item: dict):
        """
        Web endpoint for embedding text.
        Usage: POST /embed {"text": "your text here"}
        """
        text = item.get("text")
        if not text:
            return {"error": "No text provided"}
        
        return {"embedding": self.embed_text.local(text)}

@app.local_entrypoint()
def main(text: str = "test query"):
    embedder = CLIPEmbedder()
    print(f"Embedding '{text}'...")
    vector = embedder.embed_text.remote(text)
    print(f"Vector (first 5 dim): {vector[:5]}... Total dims: {len(vector)}")
