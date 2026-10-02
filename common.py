"""Where things live, and the SurfSense checkout every script runs against.

Import this first: SurfSense reads its data and models directories from the
environment the first time its settings load, so they are set here, before any
SurfSense module is imported.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
WORK = Path(os.environ.get("CJK_EVAL_WORK", Path.home() / ".surfsense-cjk-eval"))

CHECKOUT = Path(os.environ.get("SURFSENSE", ROOT.parent / "SurfSense")).resolve()
BACKEND = CHECKOUT / "surfsense_local" / "backend"
if not (BACKEND / "shared" / "search.py").is_file():
    sys.exit(f"no SurfSense checkout at {CHECKOUT}; set SURFSENSE to one")

MODELS = Path(os.environ.get("SURFSENSE_LOCAL_MODELS_DIR", Path.home() / ".surfsense" / "models"))
os.environ["SURFSENSE_LOCAL_MODELS_DIR"] = str(MODELS)

# The embedders measured, by the folder SurfSense loads each from.
EMBEDDERS = {
    "bge": "bge-small-en-v1.5",
    "granite": "granite-embedding-97m-multilingual-r2",
}
# Each embedder's semantic weight: bge's is SurfSense's own; granite's is the
# one docs/proposals/retrieval.md measured as its plateau.
SEMANTIC_WEIGHTS = {"bge": 0.65, "granite": 0.85}


def use_index(name: str) -> Path:
    """Point SurfSense at one index's own data directory, and return it."""
    directory = WORK / name
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["SURFSENSE_LOCAL_DATA_DIR"] = str(directory)
    for path in (BACKEND, BACKEND / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return directory
