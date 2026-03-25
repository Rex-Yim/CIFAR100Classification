"""
Local web demo: upload an image → CIFAR-100 class probabilities (SE-ResNet checkpoint).

Run from repo root:
  CHECKPOINT=path/to/best.pt python -m uvicorn demo.app:app --reload --host 127.0.0.1 --port 8765

Open http://127.0.0.1:8765
"""

from __future__ import annotations

import io
import os
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

from demo.cifar100_labels import FINE_LABEL_NAMES
from model import build_model_from_checkpoint
from utils import CIFAR100_MEAN, CIFAR100_STD

ROOT = Path(__file__).resolve().parent.parent


def _load_fine_labels() -> list[str]:
    meta = ROOT / "data" / "cifar-100-python" / "meta"
    if meta.is_file():
        with meta.open("rb") as f:
            data = pickle.load(f, encoding="latin1")
        return list(data["fine_label_names"])
    return list(FINE_LABEL_NAMES)


def _resolve_checkpoint() -> Path:
    env = os.environ.get("CHECKPOINT", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f"CHECKPOINT not found: {p}")
    candidates = [ROOT / "checkpoints" / "best.pt"]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "No checkpoint found. Set CHECKPOINT=/path/to/best.pt or place checkpoints/best.pt in the repo."
    )


def _resolve_device() -> torch.device:
    d = os.environ.get("DEMO_DEVICE", "").strip().lower()
    if d in ("cpu", "cuda", "mps"):
        if d == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        if d == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(d)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_checkpoint: Path | None = None
_device: torch.device | None = None
_model: torch.nn.Module | None = None
_labels: list[str] | None = None
_preprocess = transforms.Compose(
    [
        transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ]
)


def get_model() -> tuple[torch.nn.Module, torch.device, list[str]]:
    global _checkpoint, _device, _model, _labels
    if _model is None:
        _checkpoint = _resolve_checkpoint()
        _device = _resolve_device()
        ck = torch.load(_checkpoint, map_location=_device, weights_only=False)
        _model = build_model_from_checkpoint(ck).to(_device)
        state = ck.get("ema_model_state_dict") or ck["model_state_dict"]
        _model.load_state_dict(state)
        _model.eval()
        _labels = _load_fine_labels()
        if len(_labels) != 100:
            _labels = list(FINE_LABEL_NAMES)
    assert _model is not None and _device is not None and _labels is not None
    return _model, _device, _labels


app = FastAPI(title="CIFAR-100 classifier demo")
STATIC = Path(__file__).resolve().parent / "static"
if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_path = Path(__file__).resolve().parent / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/api/info")
def api_info() -> dict:
    """Lightweight: resolve checkpoint path without loading weights (first page load)."""
    try:
        ckpt = _resolve_checkpoint()
    except FileNotFoundError as e:
        return {
            "checkpoint": None,
            "device": str(_resolve_device()),
            "num_classes": 100,
            "error": str(e),
        }
    return {"checkpoint": str(ckpt), "device": str(_resolve_device()), "num_classes": 100}


SAMPLES_DIR = STATIC / "samples"


@app.get("/api/samples")
def list_samples() -> dict:
    """PNG thumbnails shipped under static/samples (see scripts/export_demo_sample_images.py)."""
    if not SAMPLES_DIR.is_dir():
        return {"samples": []}
    names = sorted(p.name for p in SAMPLES_DIR.glob("*.png"))
    return {
        "samples": [{"name": n, "url": f"/static/samples/{n}"} for n in names],
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    ct = (file.content_type or "").lower()
    if ct and not ct.startswith("image/"):
        raise HTTPException(400, "Please upload an image (PNG or JPEG).")
    raw = await file.read()
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(400, "Image too large (max 8 MB).")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except OSError as e:
        raise HTTPException(400, f"Could not read image: {e}") from e

    model, device, labels = get_model()
    x = _preprocess(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    topk = min(5, len(labels))
    values, indices = torch.topk(probs, k=topk)
    predictions = [
        {"rank": i + 1, "label": labels[j.item()], "id": int(j.item()), "probability": float(values[i].item())}
        for i, j in enumerate(indices)
    ]

    return {
        "predictions": predictions,
        "note": "Model expects CIFAR-style 32×32 content; photos are resized — results are illustrative.",
    }
