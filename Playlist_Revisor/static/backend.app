"""FastAPI backend backward-compatibility shim.

This module proxies requests to the consolidated `backend.main` application.
"""
import sys
from pathlib import Path

# Ensure root directory is in sys.path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.main import (
    app,
    lifespan,
    state,
    ask,
    ingest_status,
    serve_audio,
    index,
    spa_fallback,
    main,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.backend:app", host="127.0.0.1", port=8000, reload=True)
