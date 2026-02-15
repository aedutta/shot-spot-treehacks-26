import modal
import os
import uuid
from pathlib import Path

# Configuration
CACHE_DIR = "/data"  # Shared Volume
MODEL_REPO = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
SEGMENT_DURATION = 60 # Seconds per parallel worker

# Transcription
WHISPER_MODEL = "base"
MAX_CONTAINERS = 20

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
        from transformers import CLIPProcessor, CLIPModel, pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
        import warnings
        
        # Suppress benign warnings from Transformers/Whisper
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚡ Loading Models on {self.device}...", flush=True)
        
        # Load CLIP
        self.processor = CLIPProcessor.from_pretrained(MODEL_REPO, local_files_only=True)
        self.model = CLIPModel.from_pretrained(MODEL_REPO, local_files_only=True).to(self.device)
        self.model.eval()
        
        # Load Whisper safely using HF Pipeline (More robust & faster on A10G)
        model_id = "openai/whisper-base"
        
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        ).to(self.device)
        
        whisper_processor = AutoProcessor.from_pretrained(model_id)

        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            tokenizer=whisper_processor.tokenizer,
            feature_extractor=whisper_processor.feature_extractor,
            chunk_length_s=30,
            batch_size=8,  # Better throughput
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

        print(f"▶️ Processing Segment {start}-{end}s: {title}", flush=True)

        try:
            stream_url, headers = self._get_stream_info(video_url)
        except Exception as e:
            print(f"❌ Stream extraction failed: {e}")
            return 0

        # Build headers for FFMPEG (Critical for TikTok/Twitch)
        headers_list = []
        for k, v in headers.items():
            headers_list.append(f"{k}: {v}")
        headers_str = "\r\n".join(headers_list)
        if headers_str:
            headers_str += "\r\n"

        # --- 1. Audio Transcription (On the fly) ---
        transcription_text = ""
        try:
            # We use a directory to avoid file locking/existence issues with yt-dlp
            with tempfile.TemporaryDirectory() as temp_dir:
                audio_path = os.path.join(temp_dir, "audio.wav")
                
                # Use yt-dlp to download the audio segment directly
                # This is more robust against 403s than ffmpeg -i URL because yt-dlp handles the session/cookies
                cmd_audio = [
                    "yt-dlp",
                    "--quiet", "--no-warnings",
                    "--download-sections", f"*{start}-{end}",
                    "--force-keyframes-at-cuts",
                    "--extract-audio",
                    "--audio-format", "wav",
                    "--audio-quality", "0",
                    "--output", audio_path,
                    video_url
                ]
                
                # Add extractor args for spoofing if needed
                cmd_audio.extend(["--extractor-args", "youtube:player_client=android,ios"])

                try:
                    subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
                except subprocess.CalledProcessError as e:
                    # If yt-dlp fails (e.g. no audio stream?), we catch it
                    print(f"FAILED yt-dlp Audio Cmd: {' '.join(cmd_audio)}")
                    print(f"Error output: {e.stderr.decode()}")
                    raise e
                
                # Verify file exists (yt-dlp might add extension but we forced wav)
                # simpler: just listdir and pick the first file
                files = os.listdir(temp_dir)
                if not files:
                    raise FileNotFoundError("yt-dlp did not produce an audio file")
                
                final_audio_path = os.path.join(temp_dir, files[0])
                
                # Transcribe using pipeline
                # We can load into memory or pass path
                res = self.transcriber(
                    final_audio_path, 
                    batch_size=8,
                    return_timestamps=True,
                    generate_kwargs={"language": "en"}
                )
                transcription_text = res.get("text", "").strip() # type: ignore
        except Exception as e:
            print(f"⚠️ Audio failed for segment {start}s (skipping text): {e}", flush=True)

        # Embed Transcription if exists
        text_embedding = None
        if transcription_text:
            inputs = self.processor(text=[transcription_text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                emb = self.model.get_text_features(**inputs)
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
                text_embedding = emb.cpu().numpy()[0]

        # --- 2. Frame Extraction & Embedding ---
        # Use yt-dlp to download the video segment to a temp file, then process with ffmpeg
        # This is more robust than streaming directly with ffmpeg
        
        frames = []
        timestamps = []
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                video_path = os.path.join(temp_dir, "video.mp4")
                
                cmd_video_dl = [
                    "yt-dlp",
                    "--quiet", "--no-warnings",
                    "--download-sections", f"*{start}-{end}",
                    "--force-keyframes-at-cuts",
                    "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "--output", video_path,
                    video_url
                ]
                # Spoofing args
                cmd_video_dl.extend(["--extractor-args", "youtube:player_client=android,ios"])
                
                # Download segment
                subprocess.run(cmd_video_dl, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
                
                # Verify file
                files = os.listdir(temp_dir)
                if not files:
                    print(f"⚠️ yt-dlp failed to download video segment {start}-{end}")
                    return 0
                
                final_video_path = os.path.join(temp_dir, files[0])

                # Extract frames from the LOCAL file
                cmd_ffmpeg = [
                    "ffmpeg", 
                    "-i", final_video_path,
                    "-vf", "fps=1,scale=224:224", 
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
                    timestamps.append(start + frame_idx) 
                    frame_idx += 1
                
                process.terminate()

        except subprocess.CalledProcessError as e:
            print(f"❌ Video download/processing failed for {start}-{end}: {e.stderr.decode()}")
            return 0
        except Exception as e:
            print(f"❌ Error in video pipeline: {e}")
            return 0
            
        if not frames:
            print(f"⚠️ No frames extracted for {start}-{end}s")
            return 0

        # Batch Embed Images (rest of the code...)
        inputs = self.processor(images=frames, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            img_emb = self.model.get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        img_emb = img_emb.cpu().numpy()

        # Combine Image + Text (Late Fusion)
        results = []
        for i, t in enumerate(timestamps):
            vec = img_emb[i]
            # If we have text for this segment, add it to every frame in the segment
            if text_embedding is not None:
                vec = vec + text_embedding
                vec = vec / (np.linalg.norm(vec) + 1e-9) # Renormalize
            
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
@app.function(image=image)
def ingest_video_orchestrator(url: str):
    import yt_dlp
    import math

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
    
    # Run in parallel
    worker = VideoWorker()
    list(worker.process_segment_pipeline.starmap(segments))
    
    print("✅ Ingestion Complete.")

@app.local_entrypoint()
def main():
    ingest_video_orchestrator.remote("https://www.youtube.com/watch?v=dQw4w9WgXcQ")