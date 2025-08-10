# FinDocGPT

FinDocGPT is a full-stack AI-powered financial document analysis platform. This project demonstrates advanced capabilities in financial Q&A, sentiment analysis, time-series forecasting, and actionable stock recommendations, all integrated into a modern web application.

## Features
- **Document Q&A:** Ask questions about uploaded financial documents using state-of-the-art NLP models.
- **Sentiment Analysis:** Analyze the sentiment of financial reports and management commentary.
- **Forecasting:** Predict future stock prices using Prophet and yfinance data.
- **Recommendations:** Get transparent BUY/HOLD/SELL recommendations with explainability.
- **Modern UI:** Built with React, Chart.js, and Tailwind CSS for a clean, responsive experience.

## Getting Started

### Backend (Python/FastAPI)
1. Create a Python virtual environment in the `backend` folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the backend server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

### Frontend (React/Vite)
1. Navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the frontend development server:
   ```bash
   npm run dev
   ```

### Docker Compose (Recommended)
To run both backend and frontend together:
```bash
docker compose up --build
```
Backend: [http://localhost:8000](http://localhost:8000)
Frontend: [http://localhost:5173](http://localhost:5173)

## Development Notes
- The backend is built with FastAPI and leverages HuggingFace Transformers, FinBERT, Prophet, and other ML libraries.
- The frontend is built with React and Vite, using Chart.js for data visualization and Tailwind CSS for styling.
- All code, architecture, and integrations were designed and implemented by me, with a focus on extensibility and clarity.

## Troubleshooting
| Problem                      | Cause                                 | Fix                                                      |
|------------------------------|---------------------------------------|----------------------------------------------------------|
| Frontend CORS error          | Backend not allowing frontend origin   | Add CORS middleware in `backend/app/main.py`             |
| yfinance errors/no forecast  | Invalid ticker or API rate limit       | Try a different ticker (e.g., MSFT, TSLA)                |
| Model download slow          | HuggingFace model cache on first run   | Wait for download, or pre-download models                |
| Prophet install fails        | Missing cmdstanpy build                | `pip install pystan==2.19.1.1` before Prophet            |

## About
For questions, feedback, or collaboration, feel free to reach out.
