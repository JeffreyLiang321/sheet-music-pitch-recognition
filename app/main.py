"""FastAPI server: upload a score, get back note events plus MIDI/MusicXML.

    uvicorn app.main:app --reload

Serves the built frontend from web/dist when it exists.
"""
from __future__ import annotations

import base64
import os
import tempfile
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from omr.pipeline import Pipeline
from omr.score import assign_timing, events_json, write_midi, write_musicxml, DEFAULT_BPM

MAX_PAGES = int(os.environ.get("MAX_PAGES", 6))
MAX_UPLOAD_MB = 20
RENDER_DPI = 300
DISPLAY_WIDTH = 1400

pipeline: Pipeline | None = None
results: "OrderedDict[str, dict]" = OrderedDict()  # small in-memory cache of recent runs


@asynccontextmanager
async def lifespan(_app):
    global pipeline
    pipeline = Pipeline()
    yield


app = FastAPI(title="sheet-to-audio", lifespan=lifespan)


def _pages_from_upload(data: bytes, filename: str) -> list[np.ndarray]:
    name = filename.lower()
    if name.endswith(".pdf"):
        import pymupdf
        doc = pymupdf.open(stream=data, filetype="pdf")
        pages = []
        for page in list(doc)[:MAX_PAGES]:
            pix = page.get_pixmap(dpi=RENDER_DPI, colorspace=pymupdf.csGRAY)
            pages.append(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width))
        return pages
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(400, "could not read the image")
    return [img]


def _display(gray: np.ndarray) -> tuple[str, float]:
    scale = min(1.0, DISPLAY_WIDTH / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode(), scale


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), bpm: int = Form(DEFAULT_BPM)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1e6:
        raise HTTPException(413, f"upload larger than {MAX_UPLOAD_MB} MB")
    t0 = time.time()
    grays = _pages_from_upload(data, file.filename or "")

    page_results, pages_out = [], []
    for gray in grays:
        res = pipeline.run(gray)
        page_results.append(res)
        image, dscale = _display(gray)
        pages_out.append({
            "image": image, "display_scale": dscale, "width": res.width, "height": res.height,
            "staffs": [{
                "top": s.top / res.scale, "bottom": s.bottom / res.scale,
                "left": s.left / res.scale, "right": s.right / res.scale,
                "clef": s.clef, "key_fifths": s.key_fifths, "system": s.system,
            } for s in res.staffs],
        })
    events = assign_timing(page_results)
    elapsed = time.time() - t0

    run_id = uuid.uuid4().hex[:12]
    results[run_id] = {"events": events, "bpm": bpm}
    while len(results) > 20:
        results.popitem(last=False)

    return {
        "id": run_id, "bpm": bpm, "pages": pages_out, "events": events_json(events),
        "stats": {"pages": len(grays), "notes": len(events),
                  "staffs": sum(len(p.staffs) for p in page_results), "seconds": round(elapsed, 2)},
    }


def _export(run_id: str, kind: str):
    run = results.get(run_id)
    if run is None:
        raise HTTPException(404, "result expired, transcribe again")
    suffix = ".mid" if kind == "midi" else ".musicxml"
    path = Path(tempfile.gettempdir()) / f"sheet-{run_id}{suffix}"
    (write_midi if kind == "midi" else write_musicxml)(run["events"], path, run["bpm"])
    media = "audio/midi" if kind == "midi" else "application/vnd.recordare.musicxml+xml"
    return FileResponse(path, media_type=media, filename=f"sheet-{run_id}{suffix}")


@app.get("/api/result/{run_id}/midi")
def midi(run_id: str):
    return _export(run_id, "midi")


@app.get("/api/result/{run_id}/musicxml")
def musicxml(run_id: str):
    return _export(run_id, "musicxml")


@app.get("/api/health")
def health():
    return {"ok": True, "models": "onnx" if (Path("models") / "detector.onnx").exists() else "keras"}


dist = Path(__file__).resolve().parents[1] / "web" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="web")
