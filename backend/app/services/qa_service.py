from transformers import pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=-1)

DOCS = {
    "demo_doc": "ACME Corp reported revenue of $4.2 billion in Q2. Net profit margin improved year-over-year. Management noted increased demand and supply-chain pressures."
}

def answer_question(question: str, doc_id: str):
    context = DOCS.get(doc_id, "")
    if not context:
        return ("Document not found", 0.0)
    res = qa_pipeline({"question": question, "context": context})
    return res.get("answer"), float(res.get("score", 0.0))
