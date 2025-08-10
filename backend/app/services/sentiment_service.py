from transformers import pipeline
try:
    sentiment = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except Exception:
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

DOCS = {
    "demo_doc": "ACME's management sounded optimistic about revenue growth; some concern about rising costs."
}

def analyze_sentiment(doc_id: str):
    text = DOCS.get(doc_id, "")
    if not text:
        return 0.0
    out = sentiment(text[:512])
    label = out[0]["label"].lower()
    score = out[0]["score"]
    if label == "positive" or label == "pos":
        return score
    elif label == "negative" or label == "neg":
        return -score
    else:
        return 0.0
