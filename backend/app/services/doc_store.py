import os
import threading

_LOCK = threading.Lock()
_DOCS = {
    # default demo doc content for first run
    "demo_doc": (
        "ACME Corp reported revenue of $4.2 billion in Q2. Net profit margin improved year-over-year. "
        "Management noted increased demand and supply-chain pressures."
    )
}

def _read_file_if_exists(path: str) -> str | None:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        return None
    return None

def get_doc(doc_id: str) -> str:
    with _LOCK:
        if doc_id in _DOCS:
            return _DOCS[doc_id]
    # Try volume path first (/data is mounted), then app-relative path
    candidates = [
        f"/data/examples/{doc_id}.txt",
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "examples", f"{doc_id}.txt"),
    ]
    for p in candidates:
        txt = _read_file_if_exists(os.path.abspath(p))
        if txt:
            with _LOCK:
                _DOCS[doc_id] = txt
            return txt
    # fallback empty
    return ""

def set_doc(doc_id: str, text: str) -> None:
    with _LOCK:
        _DOCS[doc_id] = text or ""
