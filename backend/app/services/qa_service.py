import os
from transformers import pipeline
from .doc_store import get_doc
qa_pipeline = None

def _ensure_qa():
    global qa_pipeline
    if qa_pipeline is not None:
        return qa_pipeline
    try:
        # Skip model init entirely unless explicitly allowed
        if os.environ.get("ALLOW_MODEL_DOWNLOAD", "0") not in ("1", "true", "True"):
            qa_pipeline = False
            return None
        # Respect offline environments; if no cache, this will fail fast and we'll fallback
        if os.environ.get("TRANSFORMERS_OFFLINE") is None:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=-1)
        return qa_pipeline
    except Exception:
        qa_pipeline = False  # sentinel to avoid retrying repeatedly
        return None

DOCS = {
    "demo_doc": "ACME Corp reported revenue of $4.2 billion in Q2. Net profit margin improved year-over-year. Management noted increased demand and supply-chain pressures."
}

def answer_question(question: str, doc_id: str):
    context = get_doc(doc_id)
    if not context:
        return ("Document not found", 0.0)
    # Try to lazily load the model; if unavailable, use fallback
    pipe = _ensure_qa()
    if not pipe:
        # trivial fallback: return the first sentence containing a monetary value or the first sentence
        import re
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
        monetary = next((s for s in sentences if re.search(r"\$\d", s)), None)
        answer = monetary or (sentences[0] if sentences else "No context available")
        return (answer, 0.15)
    res = pipe({"question": question, "context": context})
    return res.get("answer"), float(res.get("score", 0.0))
