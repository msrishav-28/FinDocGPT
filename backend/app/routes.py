import os
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from .services.qa_service import answer_question
from .services.sentiment_service import analyze_sentiment
from .services.doc_store import set_doc, get_doc
from .services.forecast_service import forecast_prices
from .services.recommender import recommend

# Import new API structure
from .api.v1 import api_router

router = APIRouter()

# Include versioned API routes
router.include_router(api_router)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    doc_id = "demo_doc"
    try:
        # Save into mounted volume
        os.makedirs('/data/examples', exist_ok=True)
        with open('/data/examples/demo_doc.txt','wb') as f:
            f.write(contents)
    except Exception:
        # Fallback to app-relative path (non-docker local dev)
        os.makedirs('data/examples', exist_ok=True)
        with open('data/examples/demo_doc.txt','wb') as f:
            f.write(contents)
    # Update in-memory store for immediate use
    try:
        set_doc(doc_id, contents.decode('utf-8', errors='ignore'))
    except Exception:
        pass
    return {"doc_id": doc_id}

@router.post("/qa")
async def qa_endpoint(question: str = Form(...), doc_id: str = Form(...)):
    answer, confidence = answer_question(question, doc_id)
    return {"answer": answer, "confidence": confidence}

@router.get("/sentiment")
async def sentiment_endpoint(doc_id: str):
    score = analyze_sentiment(doc_id)
    return {"sentiment_score": score}

@router.get("/forecast")
async def forecast_endpoint(ticker: str):
    df_forecast = forecast_prices(ticker)
    return JSONResponse(content=df_forecast.to_dict(orient='list'))

@router.post("/recommend")
async def recommend_endpoint(ticker: str = Form(...), doc_id: str = Form(...)):
    rec = recommend(ticker, doc_id)
    return rec
