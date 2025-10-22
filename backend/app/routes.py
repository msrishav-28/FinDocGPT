import os
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from .services.qa_service import answer_question
from .services.sentiment_service import analyze_sentiment
from .services.doc_store import set_doc, get_doc
from .services.forecast_service import forecast_prices
from .services.recommender import recommend

# Import new document routes
from .routes.document_routes import router as document_router
from .routes.sentiment_routes import router as sentiment_router
from .routes.anomaly_routes import router as anomaly_router
from .routes.ensemble_forecast_routes import router as ensemble_forecast_router
from .routes.investment_advisory import router as investment_advisory_router
from .routes.websocket_routes import router as websocket_router
from .routes.market_data_routes import router as market_data_router
from .routes.alert_routes import router as alert_router
from .routes.auth import router as auth_router
from .routes.monitoring import router as monitoring_router
from .routes.audit import router as audit_router
from .routes.compliance import router as compliance_router
from .routes.cache_routes import router as cache_router
from .routes.database_optimization_routes import router as db_optimization_router

router = APIRouter()

# Include document processing routes
router.include_router(document_router)

# Include sentiment analysis routes
router.include_router(sentiment_router)

# Include anomaly detection routes
router.include_router(anomaly_router)

# Include ensemble forecasting routes
router.include_router(ensemble_forecast_router)

# Include investment advisory routes
router.include_router(investment_advisory_router)

# Include WebSocket routes
router.include_router(websocket_router)

# Include market data routes
router.include_router(market_data_router)

# Include alert routes
router.include_router(alert_router)

# Include authentication routes
router.include_router(auth_router)

# Include monitoring routes
router.include_router(monitoring_router)

# Include audit routes
router.include_router(audit_router)

# Include compliance routes
router.include_router(compliance_router)

# Include cache management routes
router.include_router(cache_router)

# Include database optimization routes
router.include_router(db_optimization_router)

# Include background tasks routes
from .routes.background_tasks import router as background_tasks_router
router.include_router(background_tasks_router)

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
