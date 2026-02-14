import modal
import os
import subprocess
import time
from pathlib import Path

CACHE_DIR = "/cache"
MODEL_REPO = "openai/clip-vit-base-patch32"
MODEL_REVISION = "e9734e622b7c6225a6e872d825c342f7734892c9"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "locales")
    .run_commands(
        "sed -i '/^#\\s*en_US.UTF-8 UTF-8/ s/^#//' /etc/locale.gen",
        "locale-gen en_US.UTF-8",
        "update-locale LANG=en_US.UTF-8",
    )
    .env({"LANG": "en_US.UTF-8", "HF_HUB_CACHE": CACHE_DIR})
    .uv_pip_install(
        "torch==2.5.1",
        "transformers==4.47.1",
        "huggingface-hub==0.36.0",
        "yt-dlp==2024.12.23",
        "pillow==11.0.0",
        "numpy<2.0.0",
    )
)

model_cache = modal.Volume.from_name("treehacks-model-cache", create_if_missing=True)

app = modal.App("treehacks-video-ingestor", image=image)

@app.function(volumes={CACHE_DIR: model_cache})
def download_model():
    from huggingface_hub import snapshot_download
    print(f"Downloading {MODEL_REPO}...")
    snapshot_download(MODEL_REPO, revision=MODEL_REVISION, cache_dir=CACHE_DIR)
    print("✅ Model downloaded and cached.")

@app.cls(
    gpu="H100", 
    volumes={CACHE_DIR: model_cache}, 
    max_containers=10,       
    scaledown_window=300
)
@modal.concurrent(max_inputs=4)
class VideoIngestor:
    @modal.enter()
    def load_model(self):
        """Loads model into VRAM once per container."""
        import torch
        from transformers import CLIPProcessor, CLIPModel

        self.device = "cuda"
        print("⚡ Loading CLIP model...")
        self.processor = CLIPProcessor.from_pretrained(MODEL_REPO, cache_dir=CACHE_DIR)
        self.model = CLIPModel.from_pretrained(MODEL_REPO, cache_dir=CACHE_DIR).to(self.device)
        print("✅ Model loaded.")

    def _stream_frames(self, url, fps):
        import yt_dlp
        from PIL import Image
        
        with yt_dlp.YoutubeDL({"format": "best[ext=mp4]", "quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            stream_url = info['url']
            self.current_title = info.get('title', 'Unknown')

        cmd = [
            "ffmpeg", "-i", stream_url,
            "-vf", f"fps={fps},scale=224:224",
            "-f", "image2pipe",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo", "-"
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frame_bytes = 224 * 224 * 3
        
        while True:
            raw = process.stdout.read(frame_bytes)
            if len(raw) != frame_bytes: break
            yield Image.frombytes("RGB", (224, 224), raw)
        
        process.terminate()
    
    @modal.method()
    def process_stream(self, url: str, fps: int = 1, batch_size: int = 32):
        import torch
        
        print(f"▶️ Processing: {url}")
        results = []
        batch_frames = []
        batch_timestamps = []
        frame_idx = 0

        for frame in self._stream_frames(url, fps):
            batch_frames.append(frame)
            batch_timestamps.append(frame_idx / fps)
            frame_idx += 1
            
            if len(batch_frames) >= batch_size:
                results.extend(self._embed_batch(batch_frames, batch_timestamps))
                batch_frames, batch_timestamps = [], []

        if batch_frames:
            results.extend(self._embed_batch(batch_frames, batch_timestamps))
            
        print(f"✅ Finished {self.current_title}: {len(results)} vectors.")
        return results
    
    def _embed_batch(self, frames, timestamps):
        inputs = self.processor(images=frames, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            
        return [
            {"time": t, "vector": v.tolist(), "title": self.current_title}
            for t, v in zip(timestamps, emb.cpu())
        ]

@app.local_entrypoint()
def main():
    urls = [
        "https://youtu.be/dQw4w9WgXcQ?si=CqbeYpdLFPf9mZ4T",
        "https://youtu.be/Aq5WXmQQooo?si=XYiwlQ5kjf0MfbZW",
    ]
    
    print("🚀 Launching concurrent ingestion...")
    ingestor = VideoIngestor()
    results = list(ingestor.process_stream.map(urls))
    print(f"Processed {len(results)} videos.")

if __name__ == "__main__":
    with app.run():
        main()