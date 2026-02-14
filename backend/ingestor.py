import modal
import os
import asyncio

# Configuration
CACHE_DIR = "/cache"
MODEL_REPO = "openai/clip-vit-base-patch32"
SEGMENT_DURATION = 60  

# 1. Define Image with Baked-in Model + Node.js + Cookies
def download_model_build_step():
    from transformers import CLIPProcessor, CLIPModel
    import os
    os.environ["HF_HUB_CACHE"] = CACHE_DIR
    print(f"🏗️ Baking {MODEL_REPO} into image...")
    CLIPProcessor.from_pretrained(MODEL_REPO)
    CLIPModel.from_pretrained(MODEL_REPO)

# Base Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "nodejs", "npm") 
    .pip_install(
        "torch==2.5.1",
        "transformers==4.47.1",
        "yt-dlp",
        "pillow",
        "requests",
        "numpy"
    )
    .env({"HF_HUB_CACHE": CACHE_DIR})
    .run_function(download_model_build_step)
)

# 2. Define the Cookie Mount (Runtime injection instead of build-time)
cookie_mount = modal.Mount.from_local_file("cookies.txt", remote_path="/root/cookies.txt") if os.path.exists("cookies.txt") else None
params = {"mounts": [cookie_mount]} if cookie_mount else {}

app = modal.App("treehacks-video-ingestor-v2", image=image)

# 2. Worker Class
@app.cls(
    gpu="A10G", 
    scaledown_window=120,
    secrets=[modal.Secret.from_name("mongo")],
    **params  # Mount cookies at runtime
)
class VideoWorker:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import CLIPProcessor, CLIPModel
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚡ Loading CLIP on {self.device}...", flush=True)
        
        self.processor = CLIPProcessor.from_pretrained(MODEL_REPO, local_files_only=True)
        self.model = CLIPModel.from_pretrained(MODEL_REPO, local_files_only=True).to(self.device)
        self.model.eval() 
        print("✅ Model ready.")

    def _get_stream_url(self, url):
        import yt_dlp
        # Use cookies.txt if available
        ydl_opts = {
            "format": "best", 
            "quiet": True,
            "cookiefile": "/root/cookies.txt" if os.path.exists("/root/cookies.txt") else None
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']

    @modal.method()
    def process_segment(self, video_url: str, start: float, end: float, title: str, fps: int = 1):
        import subprocess
        from PIL import Image
        import torch
        
        print(f"▶️ Segment {start}-{end}s: {title}", flush=True)
        
        try:
            stream_url = self._get_stream_url(video_url)
        except Exception as e:
            print(f"❌ Failed to get stream for {video_url}. (Check cookies.txt): {e}")
            return 0

        # FFMPEG: Seek to specific time (-ss) and duration (-t)
        cmd = [
            "ffmpeg", 
            "-ss", str(start),
            "-t", str(end - start),
            "-i", stream_url,
            "-vf", f"fps={fps},scale=224:224",
            "-f", "image2pipe",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo", 
            "-loglevel", "error",
            "-"
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frame_bytes = 224 * 224 * 3
        
        frames = []
        timestamps = []
        frame_idx = 0

        while True:
            raw = process.stdout.read(frame_bytes)
            if len(raw) != frame_bytes: break
            
            image = Image.frombytes("RGB", (224, 224), raw)
            frames.append(image)
            timestamps.append(start + (frame_idx / fps))
            frame_idx += 1

        process.terminate()

        if not frames:
            return 0

        # Batch Inference
        print(f"🧠 Embedding {len(frames)} frames...", flush=True)
        inputs = self.processor(images=frames, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)

        # STRICT API MATCH: Only sending time, vector, title
        results = [
            {"time": t, "vector": v.tolist(), "title": title, "source": video_url}
            for t, v in zip(timestamps, emb.cpu())
        ]

        self._push_results(results)
        return len(results)

    def _push_results(self, vectors):
        import requests
        if not vectors: return
        
        api_url = os.environ.get("VECTOR_API_URL")
        api_key = os.environ.get("VECTOR_API_KEY")
        
        if not api_url:
            print("⚠️ VECTOR_API_URL missing. Skipping upload.")
            return

        try:
            # Modified to use /frames/bulk for flexibility
            url = api_url.rstrip("/") + "/frames/bulk"
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["X-API-Key"] = api_key

            resp = requests.post(
                url, 
                json={"frames": vectors}, 
                headers=headers,
                timeout=15
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"❌ Upload failed: {e}")

# 3. Orchestrator
@app.function(image=image, **params)
def ingest_video_orchestrator(url: str):
    import yt_dlp
    
    print(f"🔎 Analyzing {url}...")
    
    ydl_opts = {
        "quiet": True, 
        "cookiefile": "/root/cookies.txt" if os.path.exists("/root/cookies.txt") else None
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            duration = info.get('duration')
            title = info.get('title', 'Unknown')
            if not duration:
                print(f"❌ Could not determine duration for {url}")
                return
        except Exception as e:
            print(f"❌ Metadata fetch failed: {e}")
            print("💡 TIP: Ensure 'cookies.txt' is in your local folder to bypass YouTube login checks.")
            return

    print(f"🎞️ Title: {title} | Duration: {duration}s")
    
    segments = []
    for t in range(0, duration, SEGMENT_DURATION):
        end = min(t + SEGMENT_DURATION, duration)
        segments.append((url, t, end, title))

    print(f"🚀 Launching {len(segments)} parallel workers...")
    
    worker = VideoWorker()
    results = list(worker.process_segment.starmap(segments))
    
    total_frames = sum(results)
    print(f"✅ Completed {title}: Processed {total_frames} frames.")

@app.local_entrypoint()
def main():
    if not os.path.exists("cookies.txt"):
        print("⚠️ WARNING: cookies.txt not found. YouTube may block the request.")
    
    urls = [
        "https://www.twitch.tv/videos/2689445480"
    ]
    
    for url in urls:
        ingest_video_orchestrator.remote(url)