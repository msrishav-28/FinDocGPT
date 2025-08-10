from fastapi import FastAPI
from .routes import router

app = FastAPI(title="FinDocGPT Hackathon")
app.include_router(router, prefix="/api")

@app.get("/")
def root():
    return {"message": "FinDocGPT backend running"}
