import modal
import os
import uuid
from pathlib import Path

# Configuration
CACHE_DIR = "/data"  # Path for the Shared Volume (Audio, Whisper)
MODEL_CACHE_DIR = "/root/models"  # Path for the Baked CLIP Model (Image)
# Upgrade to LAION-2B (512-dim) for better relevance (used for both image and text embeddings)
MODEL_REPO = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
SEGMENT_DURATION = 60

# Transcription (overlapping windows)
WHISPER_MODEL = "base"
WINDOW_DURATION = 10
WINDOW_OVERLAP = 2
MAX_TRANSCRIPTION_CONTAINERS = 10

# 1. Define Image with Baked-in Model + Node.js + Cookies + Whisper
def download_model_build_step():
    from transformers import CLIPProcessor, CLIPModel
    import os
    # SAVE TO BAKED PATH
    os.environ["HF_HUB_CACHE"] = "/root/models"
    print(f"🏗️ Baking {MODEL_REPO} into image...")
    CLIPProcessor.from_pretrained(MODEL_REPO)
    CLIPModel.from_pretrained(MODEL_REPO)

# Base Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "nodejs", "npm", "locales")
    .run_commands(
        "sed -i '/^#\\s*en_US.UTF-8 UTF-8/ s/^#//' /etc/locale.gen",
        "locale-gen en_US.UTF-8",
        "update-locale LANG=en_US.UTF-8",
    )
    # USE BAKED PATH FOR RUNTIME LOOKUP
    .env({"LANG": "en_US.UTF-8", "HF_HUB_CACHE": "/root/models"})
    .pip_install(
        "torch==2.5.1",
        "transformers==4.47.1",
        "yt-dlp",
        "pillow",
        "requests",
        "numpy",
        "openai-whisper",
    )
    .run_function(download_model_build_step)
)

# Volume for audio + Whisper model cache (shared across workers)
model_cache = modal.Volume.from_name("treehacks-video-ingestor-cache", create_if_missing=True)


# --- Helper Functions ---

def get_yt_dlp_opts(filename=None):
    """Standardized yt-dlp options for automated ingestion"""
    opts = {
        "quiet": True,
        "noplaylist": True,
        "geo_bypass": True,
        # ✨ Spoofing: Pretend to be iOS/Android to bypass bot checks (No cookies needed)
        "extractor_args": {"youtube": {"player_client": ["android", "ios"]}},
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    }
    if filename:
        opts.update({
             "format": "bestaudio/best",
             "outtmpl": filename,
             "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        })
    else:
        opts["format"] = "best"
    return opts

def _time_windows(total_duration_sec: float, duration: float, overlap: float) -> list[tuple[float, float]]:
    """Generate (start, end) windows in seconds. End = start + duration; step = duration - overlap."""
    if duration <= 0 or overlap < 0 or overlap >= duration:
        raise ValueError("duration > 0, 0 <= overlap < duration")
    step = duration - overlap
    windows = []
    start = 0.0
    while start < total_duration_sec:
        end = min(start + duration, total_duration_sec)
        windows.append((start, end))
        start += step
        if start >= total_duration_sec:
            break
    return windows


def _extract_segment(input_path: Path, output_path: Path, start_sec: float, duration_sec: float) -> None:
    """Extract a segment of audio using ffmpeg."""
    import subprocess
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ss", str(start_sec), "-t", str(duration_sec),
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(output_path),
        ],
        capture_output=True,
        check=True,
    )



app = modal.App("treehacks-video-ingestor-v2", image=image)


@app.function(
    image=image,
    volumes={CACHE_DIR: model_cache},
    timeout=300,
)
def download_audio_to_volume(url: str, job_id: str) -> tuple[str, float]:
    """Download audio from URL to the shared volume; return (audio_path, duration_sec)."""
    import yt_dlp
    audio_dir = Path(CACHE_DIR) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_path = audio_dir / f"{job_id}" # yt-dlp suffixes extension
    
    try:
        # Use centralized options (includes client spoofing)
        opts = get_yt_dlp_opts(filename=str(out_path))
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
            
        final_path = out_path.with_suffix(".wav")
        if not final_path.exists():
            raise RuntimeError("Download did not produce WAV")
    except Exception as e:
        raise RuntimeError(f"Failed to download audio from {url}: {e}")
        
    result = __import__("subprocess").run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(final_path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    duration = float(result.stdout.strip())
    model_cache.commit()
    return (str(final_path), duration)


# 2. Worker Class (CLIP for frames + text; Whisper for transcription)
@app.cls(
    gpu="A10G",
    scaledown_window=120,
    secrets=[modal.Secret.from_name("mongo")],
    volumes={CACHE_DIR: model_cache},
    timeout=600,
    max_containers=MAX_TRANSCRIPTION_CONTAINERS,
)
class VideoWorker:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import CLIPProcessor, CLIPModel
        import whisper

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚡ Loading CLIP ({MODEL_REPO}) on {self.device}...", flush=True)
        self.processor = CLIPProcessor.from_pretrained(MODEL_REPO, local_files_only=True)
        self.model = CLIPModel.from_pretrained(MODEL_REPO, local_files_only=True).to(self.device)
        self.model.eval()
        print("⚡ Loading Whisper...", flush=True)
        os.environ["XDG_CACHE_HOME"] = CACHE_DIR
        model_cache.reload()
        self.whisper_model = whisper.load_model(WHISPER_MODEL, download_root=CACHE_DIR)
        print("✅ CLIP + Whisper ready.")

    def _get_stream_url(self, url):
        import yt_dlp
        # Use centralized options (includes client spoofing)
        opts = get_yt_dlp_opts()
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']

    @modal.method()
    def transcribe_segment(
        self, audio_path: str, start_sec: float, end_sec: float
    ) -> tuple[tuple[float, float], str]:
        """Transcribe a single segment; returns ((start, end), text)."""
        import tempfile
        model_cache.reload()
        with tempfile.TemporaryDirectory() as tmpdir:
            segment_path = Path(tmpdir) / "segment.wav"
            _extract_segment(Path(audio_path), segment_path, start_sec, end_sec - start_sec)
            result = self.whisper_model.transcribe(str(segment_path), fp16=True, language="en")
            text = result.get("text", "").strip()
        return ((start_sec, end_sec), text)

    @modal.method()
    def embed_transcriptions(
        self, transcription_dict: dict[tuple[float, float], str]
    ) -> dict[tuple[float, float], dict]:
        """Embed each transcribed segment with same CLIP text encoder as frame embeddings."""
        import torch
        if not transcription_dict:
            return {}
        intervals = list(transcription_dict.keys())
        texts = [transcription_dict[iv] or " " for iv in intervals]
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return {
            iv: {"text": transcription_dict[iv], "embedding": v.tolist()}
            for iv, v in zip(intervals, emb.cpu())
        }

    @modal.method()
    def process_segment(
        self,
        video_url: str,
        start: float,
        end: float,
        title: str,
        transcription_embeddings: dict | None = None,
        fps: int = 1,
    ):
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

        # Batch Inference (image embeddings)
        print(f"🧠 Embedding {len(frames)} frames...", flush=True)
        inputs = self.processor(images=frames, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            img_emb = self.model.get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        img_emb = img_emb.cpu()

        # Merge transcription embeddings for each frame: for time t, average all window embeddings that contain t, then add to image embedding and renormalize
        import numpy as np
        transcription_embeddings = transcription_embeddings or {}
        results = []
        for i, t in enumerate(timestamps):
            vec = img_emb[i].numpy()
            if transcription_embeddings:
                covering = [
                    transcription_embeddings[iv]["embedding"]
                    for iv in transcription_embeddings
                    if iv[0] <= t < iv[1]
                ]
                if covering:
                    avg_text = np.array(covering).mean(axis=0).astype(np.float32)
                    avg_text = avg_text / (np.linalg.norm(avg_text) + 1e-9)
                    vec = vec + avg_text
                    vec = vec / (np.linalg.norm(vec) + 1e-9)
            results.append({"time": t, "vector": vec.tolist(), "title": title, "source": video_url})

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
                timeout=60
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"❌ Upload failed: {e}")

# 3. Orchestrator (single Modal call: download audio → transcribe → embed → frame+merge)
@app.function(image=image)
def ingest_video_orchestrator(url: str):
    import yt_dlp

    print(f"🔎 Analyzing {url}...")
    try:
        # Use centralized options (includes client spoofing)
        with yt_dlp.YoutubeDL(get_yt_dlp_opts()) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get("duration")
            title = info.get("title", "Unknown")
            if not duration:
                print(f"❌ Could not determine duration for {url}")
                return
    except Exception as e:
        print(f"❌ Metadata fetch failed: {e}")
        return

    print(f"🎞️ Title: {title} | Duration: {duration}s")

    # Download audio once for transcription
    job_id = str(uuid.uuid4())
    print("⬇️ Downloading audio for transcription...")
    audio_path, _ = download_audio_to_volume.remote(url, job_id)
    windows = _time_windows(duration, WINDOW_DURATION, WINDOW_OVERLAP)
    print(f"🎤 Transcribing {len(windows)} windows...")
    worker = VideoWorker()
    segment_args = [(audio_path, s, e) for s, e in windows]
    transcription_results = list(worker.transcribe_segment.starmap(segment_args))
    transcription_dict = dict(transcription_results)
    print("📐 Embedding transcriptions with CLIP (same as frames)...")
    with_embeddings = worker.embed_transcriptions.remote(transcription_dict)

    # Video segments: each gets (url, start, end, title, with_embeddings)
    segments = []
    for t in range(0, int(duration), SEGMENT_DURATION):
        end = min(t + SEGMENT_DURATION, duration)
        segments.append((url, float(t), end, title, with_embeddings))

    print(f"🚀 Launching {len(segments)} parallel workers (frames + merged transcription)...")
    results = list(worker.process_segment.starmap(segments))
    total_frames = sum(results)
    print(f"✅ Completed {title}: Processed {total_frames} frames.")

@app.local_entrypoint()
def main():
    urls = [
        "https://www.twitch.tv/videos/2689445480"
    ]
    
    for url in urls:
        ingest_video_orchestrator.remote(url)