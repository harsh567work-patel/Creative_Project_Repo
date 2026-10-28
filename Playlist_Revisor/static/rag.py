"""RAG pipeline backward-compatibility shim.

This module proxies requests to the consolidated `backend.rag` pipeline.
"""
import sys
from pathlib import Path

# Ensure root directory is in sys.path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.rag import *
from backend.rag import _ROOT, Settings, settings, RagPipeline, main

if __name__ == "__main__":
    main()
