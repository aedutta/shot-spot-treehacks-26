import modal
import subprocess
import uuid
from pathlib import Path

CACHE_DIR = "/cache"
CLIP_CACHE = "/clip_cache"
WHISPER_MODEL = "base"  # one of: tiny, base, small, medium, large

# CLIP text encoder (same as backend/ingestor.py) — no revision so we use same cache as ingestor
CLIP_MODEL_REPO = "openai/clip-vit-base-patch32"

# Overlapping-window transcription parameters (seconds)
WINDOW_DURATION = 10
WINDOW_OVERLAP = 2

# Max GPU containers to spin up for parallel segment transcription
MAX_CONTAINERS = 10

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "locales")
    .run_commands(
        "sed -i '/^#\\s*en_US.UTF-8 UTF-8/ s/^#//' /etc/locale.gen",
        "locale-gen en_US.UTF-8",
        "update-locale LANG=en_US.UTF-8",
    )
    .env({"LANG": "en_US.UTF-8", "HF_HUB_CACHE": CLIP_CACHE})
    .pip_install(
        "openai-whisper",
        "yt-dlp",
        "torch==2.5.1",
        "transformers==4.47.1",
        "huggingface-hub==0.36.0",
    )
)

model_cache = modal.Volume.from_name("treehacks-whisper-cache", create_if_missing=True)
clip_model_cache = modal.Volume.from_name("treehacks-model-cache", create_if_missing=True)

app = modal.App("treehacks-chat-transcription", image=image)


def _download_audio(url: str, out_path: Path) -> bool:
    """Extract audio from URL (video or audio) to a WAV file. Returns True on success."""
    import yt_dlp
    try:
        # outtmpl is the path without extension; FFmpegExtractAudio adds .wav
        stem = out_path.with_suffix("")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(stem),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }],
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return stem.with_suffix(".wav").exists()
    except Exception as e:
        print(f"❌ Error downloading audio: {e}")
        return False


def _get_audio_duration_sec(audio_path: Path) -> float:
    """Return duration of audio file in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


def _extract_segment(input_path: Path, output_path: Path, start_sec: float, duration_sec: float) -> None:
    """Extract a segment of audio using ffmpeg."""
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


@app.function(
    image=image,
    volumes={CACHE_DIR: model_cache},
    timeout=300,
)
def download_audio_to_volume(url: str, job_id: str) -> tuple[str, float]:
    """Download audio from URL to the shared volume; return (audio_path, duration_sec)."""
    audio_dir = Path(CACHE_DIR) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_path = audio_dir / f"{job_id}.wav"
    if not _download_audio(url, out_path):
        raise RuntimeError(f"Failed to download audio from {url}")
    duration = _get_audio_duration_sec(out_path)
    model_cache.commit()
    return (str(out_path), duration)


@app.function(
    image=image,
    volumes={CLIP_CACHE: clip_model_cache},
    gpu="T4",
    timeout=300,
)
def embed_transcriptions(
    transcription_dict: dict[tuple[float, float], str],
) -> dict[tuple[float, float], dict]:
    """Convert each transcribed segment to a CLIP text embedding (same text encoder as ingestor's CLIP). Returns interval -> {text, embedding}."""
    import os
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    if not transcription_dict:
        return {}

    intervals = list(transcription_dict.keys())
    texts = [transcription_dict[iv] or " " for iv in intervals]

    os.environ["HF_HUB_CACHE"] = CLIP_CACHE
    clip_model_cache.reload()
    print("⚡ Loading CLIP text encoder (no PIL/image deps)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_REPO, cache_dir=CLIP_CACHE)
    model = CLIPModel.from_pretrained(CLIP_MODEL_REPO, cache_dir=CLIP_CACHE).to(device)
    clip_model_cache.commit()  # persist downloaded model/tokenizer to volume

    print("📐 Computing CLIP text embeddings...")
    inputs = tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)

    return {
        iv: {"text": transcription_dict[iv], "embedding": v.tolist()}
        for iv, v in zip(intervals, emb.cpu())
    }


@app.cls(
    gpu="T4",
    volumes={CACHE_DIR: model_cache},
    timeout=600,
    max_containers=MAX_CONTAINERS,
)
class ChatTranscriber:
    @modal.enter()
    def load_model(self):
        """Load Whisper model once per container."""
        import whisper
        import os
        os.environ["XDG_CACHE_HOME"] = CACHE_DIR
        print("⚡ Loading Whisper model...")
        self.model = whisper.load_model(WHISPER_MODEL, download_root=CACHE_DIR)
        print("✅ Whisper model loaded.")

    @modal.method()
    def transcribe_segment(
        self, audio_path: str, start_sec: float, end_sec: float
    ) -> tuple[tuple[float, float], str]:
        """Transcribe a single segment; returns ((start, end), text). Used in parallel via starmap."""
        import tempfile

        model_cache.reload()
        with tempfile.TemporaryDirectory() as tmpdir:
            segment_path = Path(tmpdir) / "segment.wav"
            _extract_segment(Path(audio_path), segment_path, start_sec, end_sec - start_sec)
            result = self.model.transcribe(str(segment_path), fp16=True, language="en")
            text = result.get("text", "").strip()
        return ((start_sec, end_sec), text)

    @modal.method()
    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an audio file (path inside the container)."""
        import whisper
        print("🎤 Transcribing...")
        result = self.model.transcribe(audio_path, fp16=True, language="en")
        return result.get("text", "").strip()


@app.function(timeout=900)
def transcribe_url(
    url: str,
    window_duration: float = WINDOW_DURATION,
    window_overlap: float = WINDOW_OVERLAP,
) -> dict[tuple[float, float], dict]:
    """Download audio, transcribe in parallel, then embed each segment with CLIP text encoder."""
    job_id = str(uuid.uuid4())
    print("⬇️ Downloading audio to shared volume...")
    audio_path, total_duration = download_audio_to_volume.remote(url, job_id)
    windows = _time_windows(total_duration, window_duration, window_overlap)
    print(f"🎤 Transcribing {len(windows)} windows in parallel (up to {MAX_CONTAINERS} containers)...")

    segment_args = [(audio_path, start_sec, end_sec) for start_sec, end_sec in windows]
    transcriber = ChatTranscriber()
    results = list(transcriber.transcribe_segment.starmap(segment_args))
    transcription_dict = dict(results)

    print("📐 Embedding transcriptions with CLIP text encoder...")
    with_embeddings = embed_transcriptions.remote(transcription_dict)
    return with_embeddings


@app.local_entrypoint()
def main():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    print("🚀 Running chat transcription (parallel overlapping windows)...")
    result = transcribe_url.remote(
        url,
        window_duration=WINDOW_DURATION,
        window_overlap=WINDOW_OVERLAP,
    )
    print("--- Transcription + embeddings by time interval ---")
    for (start, end), data in result.items():
        text = data["text"]
        embedding = data["embedding"]
        print(f"  {start, end}:")
        print(f"    text: {text!r}")
        print(f"    embedding: {embedding}")
    print("--- End ---")


if __name__ == "__main__":
    with app.run():
        main()
