import modal
import os
import uuid
from pathlib import Path

# Configuration
CACHE_DIR = os.getenv("CACHE_DIR", "/data")  # Shared Volume
MODEL_REPO = os.getenv("MODEL_REPO", "laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
# Reduce segment duration to increase parallelism. 
# 15s segments = 4x more workers per video minute compared to 60s.
SEGMENT_DURATION = int(os.getenv("SEGMENT_DURATION", "15"))

# Transcription
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
# Increase max containers to allow massive parallelism
MAX_CONTAINERS = int(os.getenv("MAX_CONTAINERS", "100"))

# 1. Define Image
def download_model_build_step():
    from transformers import CLIPProcessor, CLIPModel
    import os
    os.environ["HF_HUB_CACHE"] = "/root/models"
    print(f"🏗️ Baking {MODEL_REPO} into image...")
    CLIPProcessor.from_pretrained(MODEL_REPO)
    CLIPModel.from_pretrained(MODEL_REPO)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "nodejs", "npm", "locales")
    .run_commands(
        "sed -i '/^#\\s*en_US.UTF-8 UTF-8/ s/^#//' /etc/locale.gen",
        "locale-gen en_US.UTF-8",
        "update-locale LANG=en_US.UTF-8",
    )
    .env({"LANG": "en_US.UTF-8", "HF_HUB_CACHE": "/root/models"})
    .pip_install(
        "torch==2.5.1",
        "transformers==4.47.1",
        "yt-dlp[default,curl-cffi]",
        "pillow",
        "requests",
        "numpy",
        "openai-whisper",
        "accelerate",
    )
    .run_function(download_model_build_step)
)

model_cache = modal.Volume.from_name("treehacks-video-ingestor-cache", create_if_missing=True)

# --- Helper Functions ---

def get_yt_dlp_opts(filename=None):
    """Standardized yt-dlp options"""
    opts = {
        "quiet": True, 
        "noplaylist": True, 
        "geo_bypass": True,
        # Client spoofing + curl-cffi is crucial for TikTok/YT
        "extractor_args": {"youtube": {"player_client": ["android", "ios"]}},
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    }
    if filename:
        opts.update({
             "format": "bestaudio/best",
             "outtmpl": filename,
             "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        })
    return opts

app = modal.App("treehacks-video-ingestor-v2", image=image)

# 2. Worker Class (CLIP + Whisper)
@app.cls(
    gpu="A10G",
    scaledown_window=300, # Keep containers warm longer
    secrets=[modal.Secret.from_name("mongo")],
    volumes={CACHE_DIR: model_cache},
    timeout=900, # 15 min timeout per segment
    max_containers=MAX_CONTAINERS,
)
class VideoWorker:
    @modal.enter()
    def setup(self):
        import torch
        from transformers import CLIPProcessor, CLIPModel, pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, logging as hf_logging
        import warnings
        
        # Suppress noise
        warnings.filterwarnings("ignore")
        hf_logging.set_verbosity_error()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚡ Loading Models on {self.device}... (v3.4 Silence)", flush=True)
        
        # Load CLIP
        self.processor = CLIPProcessor.from_pretrained(MODEL_REPO, local_files_only=True)
        self.model = CLIPModel.from_pretrained(MODEL_REPO, local_files_only=True).to(self.device)
        self.model.eval()
        
        # Load Whisper
        model_id = "openai/whisper-tiny"
        
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        ).to(self.device)
        
        # Fix conflict warning
        whisper_model.config.forced_decoder_ids = None
        whisper_model.generation_config.forced_decoder_ids = None
        
        whisper_processor = AutoProcessor.from_pretrained(model_id)

        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            tokenizer=whisper_processor.tokenizer,
            feature_extractor=whisper_processor.feature_extractor,
            chunk_length_s=30,
            batch_size=8,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device=self.device,
        )
        print("✅ Models ready.")

    def _get_stream_info(self, url):
        """Returns (url, http_headers)"""
        import yt_dlp
        opts = get_yt_dlp_opts()
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get('url'), info.get('http_headers', {})

    @modal.method()
    def process_segment_pipeline(self, video_url: str, start: float, end: float, title: str):
        """
        Runs Transcription AND Frame Embedding in parallel for this specific segment.
        This provides near-realtime results as each segment finishes.
        """
        import subprocess
        from PIL import Image
        import torch
        import numpy as np
        import tempfile

        print(f"▶️ Processing Segment {start}-{end}s: {title} (v3.3)", flush=True)

        frames = []
        timestamps = []
        transcription_text = ""
        
        # Combined download & processing to save bandwidth and startup time
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                media_path = os.path.join(temp_dir, "segment.mp4")
                
                # 1. Single Download (Video + Audio)
                cmd_dl = [
                    "yt-dlp",
                    "--quiet", "--no-warnings",
                    "--download-sections", f"*{start}-{end}",
                    "--force-keyframes-at-cuts",
                    # Best mp4 video + best m4a audio, merged
                    "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "--output", media_path,
                    video_url
                ]
                # Spoofing args for TikTok/YT
                cmd_dl.extend(["--extractor-args", "youtube:player_client=android,ios"])
                
                try:
                    subprocess.run(cmd_dl, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Download failed for {start}-{end}: {e.stderr.decode()}")
                    return 0
                
                # Verify file
                if not os.path.exists(media_path):
                    # Sometimes yt-dlp appends .mp4 or .mkv dependent on merge
                    # Just pick the largest file in temp_dir
                    files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)]
                    if not files: 
                         print(f"⚠️ No file downloaded for {start}-{end}")
                         return 0
                    media_path = max(files, key=os.path.getsize)

                # 2. Transcription (Robust In-Memory FFMPEG)
                try:
                    # Direct PCM decode to memory avoids "Soundfile" format errors entirely
                    cmd_audio = [
                        "ffmpeg", "-i", media_path,
                        "-f", "s16le", "-ac", "1", "-ar", "16000", # 16k mono PCM
                        "-vn", "-loglevel", "error", "-"
                    ]
                    
                    proc_audio = subprocess.run(cmd_audio, capture_output=True, check=False)
                    if proc_audio.returncode == 0 and len(proc_audio.stdout) > 0:
                        # Convert bytes -> float32 numpy array
                        audio_np = np.frombuffer(proc_audio.stdout, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        res = self.transcriber(
                            {"raw": audio_np, "sampling_rate": 16000},
                            batch_size=8,
                            return_timestamps=True,
                            generate_kwargs={"language": "en"}
                        )
                        transcription_text = res.get("text", "").strip() # type: ignore
                    else:
                        print(f"⚠️ Audio extraction warning (ffmpeg output empty or failed)")
                        
                except Exception as e:
                    print(f"⚠️ Transcription warning: {e}")

                # 3. Frame Extraction (Reduced FPS for speed: 0.2 FPS = 1 frame every 5s)
                # This reduces CLIP load by 5-25x compared to 1fps or 5fps
                INTERVAL = 5
                
                cmd_ffmpeg = [
                    "ffmpeg", 
                    "-i", media_path,
                    "-vf", f"fps=1/{INTERVAL},scale=224:224", 
                    "-f", "image2pipe",
                    "-pix_fmt", "rgb24",
                    "-vcodec", "rawvideo", 
                    "-loglevel", "error",
                    "-"
                ]
                
                process = subprocess.Popen(cmd_ffmpeg, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                frame_bytes = 224 * 224 * 3
                
                frame_idx = 0
                while True:
                    raw = process.stdout.read(frame_bytes)
                    if len(raw) != frame_bytes: break
                    frames.append(Image.frombytes("RGB", (224, 224), raw))
                    # Calculate timestamp properly based on interval
                    timestamps.append(start + (frame_idx * INTERVAL)) 
                    frame_idx += 1
                
                process.terminate()

        except Exception as e:
            print(f"❌ processing_pipeline critical error: {e}")
            return 0
            
        if not frames:
            print(f"⚠️ No frames extracted for {start}-{end}s")
            return 0

        # Batch Embed Images (optimized)
        inputs = self.processor(images=frames, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            img_emb = self.model.get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        img_emb = img_emb.cpu().numpy()

        # Embed Text Context (Once)
        text_embedding = None
        if transcription_text:
            inputs = self.processor(text=[transcription_text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                emb = self.model.get_text_features(**inputs)
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
                text_embedding = emb.cpu().numpy()[0]

        # Combine Image + Text (Late Fusion)
        results = []
        for i, t in enumerate(timestamps):
            vec = img_emb[i]
            # Add text context to every frame in segment
            if text_embedding is not None:
                vec = vec + text_embedding
                vec = vec / (np.linalg.norm(vec) + 1e-9) 
            
            results.append({
                "time": t, 
                "vector": vec.tolist(), 
                "title": title, 
                "source": video_url,
                "text_context": transcription_text
            })

        # Upload
        self._push_results(results)
        return len(results)

    def _push_results(self, vectors):
        import requests
        api_url = os.environ.get("VECTOR_API_URL")
        # Retry logic for robustness
        for _ in range(3):
            try:
                if not api_url: return
                resp = requests.post(
                    api_url.rstrip("/") + "/frames/bulk", 
                    json={"frames": vectors}, 
                    headers={"X-API-Key": os.environ.get("VECTOR_API_KEY", "")},
                    timeout=30
                )
                if resp.status_code == 200: break
            except:
                continue

# 3. Orchestrator
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("mongo")]
)
def ingest_video_orchestrator(url: str):
    import yt_dlp
    import math
    import requests
    import os

    print(f"🔎 Analyzing {url}...")
    try:
        with yt_dlp.YoutubeDL(get_yt_dlp_opts()) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get("duration")
            title = info.get("title", "Unknown")
            if not duration:
                print("⚠️ No duration found. Defaulting to 60s check.")
                duration = 60
    except Exception as e:
        print(f"❌ Metadata failed: {e}")
        return

    print(f"🎞️ Title: {title} | Duration: {duration}s")

    # Generate parallel tasks immediately
    segments = []
    # Overlap segments slightly (by 1s) to ensure continuity
    for t in range(0, int(duration), SEGMENT_DURATION):
        end = min(t + SEGMENT_DURATION + 1, duration) 
        segments.append((url, float(t), end, title))

    print(f"🚀 Launching {len(segments)} parallel workers...")
    
    # Notify backend of job start
    try:
        api_url = os.environ.get("VECTOR_API_URL")
        # Strip trailing / if present
        if api_url:
            print(f"📣 Notifying backend: {len(segments)} segments to {api_url}")
            requests.post(
                api_url.rstrip("/") + "/ingestion/init",
                json={"url": url, "total_segments": len(segments)},
                headers={"X-API-Key": os.environ.get("VECTOR_API_KEY", "")},
                timeout=10
            )
    except Exception as e:
        print(f"⚠️ Failed to init job stats: {e}")

    # Run in parallel
    worker = VideoWorker()
    list(worker.process_segment_pipeline.starmap(segments))
    
    print("✅ Ingestion Complete.")

@app.local_entrypoint()
def main(url: str):
    """
    Run the ingestion pipeline on a video URL.
    Usage: modal run modal_infra/ingestor.py --url <video_url>
    """
    ingest_video_orchestrator.remote(url)