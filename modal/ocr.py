import modal
import subprocess
import uuid
from pathlib import Path

CACHE_DIR = "/cache"
CLIP_CACHE = "/clip_cache"

# CLIP text encoder (same as transcription.py / ingestor)
CLIP_MODEL_REPO = "openai/clip-vit-base-patch32"

# Default: sample one frame every 5 seconds
DEFAULT_FRAME_INTERVAL_SEC = 5.0

# Max GPU containers for parallel frame OCR
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
        "yt-dlp",
        "torch==2.5.1",
        "torchvision",
        "transformers==4.47.1",
        "huggingface-hub==0.36.0",
        "easyocr",
        "pillow",
        "numpy",
    )
)

model_cache = modal.Volume.from_name("treehacks-whisper-cache", create_if_missing=True)
clip_model_cache = modal.Volume.from_name("treehacks-model-cache", create_if_missing=True)

app = modal.App("treehacks-ocr", image=image)


def _download_video(url: str, out_path: Path) -> bool:
    """Download video from URL to a local file. Returns True on success."""
    import yt_dlp
    try:
        stem = out_path.with_suffix("")
        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": str(stem) + ".%(ext)s",
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        # yt-dlp may write stem.mp4, stem.webm, etc.; normalize to out_path
        for p in stem.parent.glob(stem.name + ".*"):
            if p != out_path:
                p.rename(out_path)
            return True
        return out_path.exists()
    except Exception as e:
        print(f"❌ Error downloading video: {e}")
        return False


def _get_video_duration_sec(video_path: Path) -> float:
    """Return duration of video in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


def _frame_intervals(
    total_duration_sec: float,
    interval_sec: float,
) -> list[tuple[float, float]]:
    """Generate (start, end) intervals for frame sampling. One frame per interval at start_sec."""
    if interval_sec <= 0:
        raise ValueError("interval_sec must be positive")
    intervals = []
    start = 0.0
    while start < total_duration_sec:
        end = min(start + interval_sec, total_duration_sec)
        intervals.append((start, end))
        start += interval_sec
    return intervals


def _extract_frame(video_path: Path, timestamp_sec: float, output_path: Path) -> None:
    """Extract a single frame at timestamp_sec using ffmpeg. Fallback to first frame if seek yields no output."""
    video_path = video_path.resolve()
    output_path = output_path.resolve()
    # -map 0:v:0 = require first video stream (fail clearly if audio-only)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-map", "0:v:0", "-ss", str(timestamp_sec), "-vframes", "1", "-q:v", "2",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr or result.stdout}")
    if output_path.exists():
        return
    # Some codecs/containers exit 0 but write nothing at this timestamp; try first frame
    result2 = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-map", "0:v:0", "-vframes", "1", "-q:v", "2", str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result2.returncode != 0:
        raise RuntimeError(f"ffmpeg fallback (first frame) failed: {result2.stderr or result2.stdout}")
    if not output_path.exists():
        raise RuntimeError(
            f"ffmpeg did not write {output_path} (no video stream or codec issue). "
            "Ensure the source has a video stream."
        )


def _bbox_height_and_area(bbox: list, img_height: int, img_width: int) -> tuple[float, float]:
    """Return (height_px, area_ratio) for a 4-point bbox. area_ratio = bbox_area / image_area."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    area_ratio = (w * h) / (img_height * img_width) if (img_height * img_width) > 0 else 0.0
    return (h, area_ratio)


@app.function(
    image=image,
    volumes={CACHE_DIR: model_cache},
    timeout=300,
)
def download_video_to_volume(url: str, job_id: str) -> tuple[str, float]:
    """Download video from URL to the shared volume; return (video_path, duration_sec)."""
    video_dir = Path(CACHE_DIR) / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    out_path = video_dir / f"{job_id}.mp4"
    if not _download_video(url, out_path):
        raise RuntimeError(f"Failed to download video from {url}")
    duration = _get_video_duration_sec(out_path)
    model_cache.commit()
    return (str(out_path), duration)


@app.function(
    image=image,
    volumes={CLIP_CACHE: clip_model_cache},
    gpu="T4",
    timeout=300,
)
def embed_ocr_texts(
    ocr_dict: dict[tuple[float, float], str],
) -> dict[tuple[float, float], dict]:
    """Convert each OCR segment text to a CLIP text embedding. Returns interval -> {text, embedding}."""
    import os
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    if not ocr_dict:
        return {}

    intervals = list(ocr_dict.keys())
    texts = [ocr_dict[iv] or " " for iv in intervals]

    os.environ["HF_HUB_CACHE"] = CLIP_CACHE
    clip_model_cache.reload()
    print("⚡ Loading CLIP text encoder...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_REPO, cache_dir=CLIP_CACHE)
    model = CLIPModel.from_pretrained(CLIP_MODEL_REPO, cache_dir=CLIP_CACHE).to(device)
    clip_model_cache.commit()

    print("📐 Computing CLIP text embeddings...")
    inputs = tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)

    return {
        iv: {"text": ocr_dict[iv], "embedding": v.tolist()}
        for iv, v in zip(intervals, emb.cpu())
    }


@app.cls(
    gpu="T4",
    volumes={CACHE_DIR: model_cache},
    timeout=600,
    max_containers=MAX_CONTAINERS,
)
class VideoOCReader:
    @modal.enter()
    def load_model(self):
        """Load EasyOCR once per container."""
        import easyocr
        print("⚡ Loading EasyOCR...")
        self.reader = easyocr.Reader(["en"], gpu=True)
        print("✅ EasyOCR loaded.")

    @modal.method()
    def ocr_frame(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        min_bbox_height_px: float = 0.0,
        min_bbox_area_ratio: float = 0.0,
    ) -> tuple[tuple[float, float], str]:
        """Extract one frame at start_sec, run EasyOCR, filter by bbox size, return ((start, end), concatenated text)."""
        import tempfile
        import numpy as np
        from PIL import Image

        model_cache.reload()
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_path = Path(tmpdir) / "frame.jpg"
            _extract_frame(Path(video_path), start_sec, frame_path)
            img = np.array(Image.open(frame_path).convert("RGB"))
            h, w = img.shape[0], img.shape[1]

            results = self.reader.readtext(img)
            parts = []
            for (bbox, text, _conf) in results:
                if not text or not text.strip():
                    continue
                bbox_h, area_ratio = _bbox_height_and_area(bbox, h, w)
                # Skip small text when thresholds are set (0 = no filter)
                if min_bbox_height_px > 0 and bbox_h < min_bbox_height_px:
                    continue
                if min_bbox_area_ratio > 0 and area_ratio < min_bbox_area_ratio:
                    continue
                parts.append(text.strip())
            concatenated = " ".join(parts) if parts else ""
        return ((start_sec, end_sec), concatenated)


@app.function(timeout=900)
def ocr_url(
    url: str,
    frame_interval_sec: float = DEFAULT_FRAME_INTERVAL_SEC,
    min_bbox_height_px: float = 0.0,
    min_bbox_area_ratio: float = 0.0,
) -> dict[tuple[float, float], dict]:
    """Download video, sample frames every frame_interval_sec, OCR each frame with size filtering, embed with CLIP."""
    job_id = str(uuid.uuid4())
    print("⬇️ Downloading video to shared volume...")
    video_path, total_duration = download_video_to_volume.remote(url, job_id)
    intervals = _frame_intervals(total_duration, frame_interval_sec)
    print(f"📷 OCRing {len(intervals)} frames (every {frame_interval_sec}s, up to {MAX_CONTAINERS} containers)...")

    segment_args = [
        (video_path, start_sec, end_sec, min_bbox_height_px, min_bbox_area_ratio)
        for start_sec, end_sec in intervals
    ]
    reader = VideoOCReader()
    results = list(reader.ocr_frame.starmap(segment_args))
    ocr_dict = dict(results)

    '''
    print("📐 Embedding OCR text with CLIP text encoder...")
    with_embeddings = embed_ocr_texts.remote(ocr_dict)
    return with_embeddings
    '''
    # For now: print extracted text instead of embedding
    print("--- Extracted OCR text by time interval ---")
    for (start, end), text in sorted(ocr_dict.items()):
        print(f"  [{start:.1f}s - {end:.1f}s]: {text!r}")
    print("--- End ---")

    return ocr_dict


@app.local_entrypoint()
def main():
    url = "https://www.twitch.tv/nascar/clip/InquisitivePowerfulPancakeWTRuck-iatVSdOjrusxEisc"
    # print("🚀 Running OCR (frames every 5s, printing extracted text)...")
    print("🚀 Running OCR (frames every 5s, printing extracted text)...")
    # Use min_bbox_height_px / min_bbox_area_ratio > 0 to filter out tiny text (0 = no filter)
    result = ocr_url.remote(
        url,
        frame_interval_sec=5.0,
        min_bbox_height_px=100.0,       # drop text shorter than 20px
        min_bbox_area_ratio=0.0001,    # drop text smaller than 0.01% of frame area
    )
    '''
    print("--- OCR + embeddings by time interval ---")
    for (start, end), data in result.items():
        text = data["text"]
        embedding = data["embedding"]
        print(f"  ({start}, {end}):")
        print(f"    text: {text!r}")
        print(f"    embedding len: {len(embedding)}")
    '''
    print("--- Result (interval -> text) ---")
    for (start, end), text in sorted(result.items()):
        print(f"  ({start}, {end}): {text!r}")
    print("--- End ---")


if __name__ == "__main__":
    with app.run():
        main()
