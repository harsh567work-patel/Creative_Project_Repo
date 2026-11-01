"""FastAPI backend for Playlist Explainer.

Run from the project root:
    pip install -r requirements.txt
    python -m backend.rag ingest         # one-time: embed transcripts → Qdrant
    uvicorn backend.main:app --reload    # start the server

Open http://localhost:8000 in your browser.
"""
from contextlib import asynccontextmanager
from pathlib import Path

import aiofiles
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.rag import RagPipeline, settings, ingest_folder, ingest_one

_ROOT = Path(__file__).parent.parent          # project root
_AUDIO_DIR = _ROOT / "data" / "audio" if (_ROOT / "data" / "audio").exists() else _ROOT / "audio"
_STATIC_DIR = Path(__file__).parent / "static"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["rag"] = RagPipeline()   # loads the embedding model once at startup
    yield


app = FastAPI(title="Playlist Explainer", lifespan=lifespan)

# Serve static assets (index.html + any assets in backend/static/)
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ── API ──────────────────────────────────────────────────────────────────────

class AskBody(BaseModel):
    question: str


@app.post("/api/ask")
def ask(body: AskBody):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question is too long (max 1000 chars).")
    try:
        return state["rag"].ask(question).to_dict()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.get("/api/ingest-status")
def ingest_status():
    """Quick health-check: returns the number of chunks currently in Qdrant."""
    try:
        count = state["rag"].store.count()
        return {"chunks": count, "ready": count > 0}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class IngestOneBody(BaseModel):
    target: str


@app.post("/api/ingest-new")
def api_ingest_new():
    """Embed newly added transcripts that are not yet in Qdrant."""
    try:
        rag = state.get("rag")
        store = rag.store if rag else None
        res = ingest_folder(new_only=True, store=store)
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ingest-one")
def api_ingest_one(body: IngestOneBody):
    """Embed or re-embed a single transcript by filename or video_id."""
    target = body.target.strip()
    if not target:
        raise HTTPException(status_code=400, detail="Target is empty.")
    try:
        rag = state.get("rag")
        store = rag.store if rag else None
        res = ingest_one(target, store=store)
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/audio/{filename:path}")
async def serve_audio(filename: str):
    """Stream a local audio/video file from the data/audio/ or audio/ directory."""
    safe = Path(filename).name
    path = _AUDIO_DIR / safe
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {safe}")

    suffix = path.suffix.lower()
    media_type = {
        ".webm": "video/webm",
        ".mp4": "video/mp4",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".opus": "audio/ogg",
    }.get(suffix, "application/octet-stream")

    async def stream():
        async with aiofiles.open(path, "rb") as f:
            while chunk := await f.read(1 << 20):  # 1 MB chunks
                yield chunk

    return StreamingResponse(stream(), media_type=media_type)


# ── Frontend ─────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    index_file = _STATIC_DIR / "index.html"
    if not index_file.exists():
        fallback_index = _ROOT / "app" / "static" / "index.html"
        if fallback_index.exists():
            return FileResponse(fallback_index)
    return FileResponse(index_file)


# Allow SPA-style routing (any unmatched path → index.html)
@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    index_file = _STATIC_DIR / "index.html"
    if not index_file.exists():
        fallback_index = _ROOT / "app" / "static" / "index.html"
        if fallback_index.exists():
            return FileResponse(fallback_index)
    return FileResponse(index_file)


def main():
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
