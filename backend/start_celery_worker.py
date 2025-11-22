#!/usr/bin/env python3
"""
Celery worker startup script
"""

import os
import sys
from app.celery_app import celery_app

if __name__ == "__main__":
    # Set up environment
    os.environ.setdefault("PYTHONPATH", os.path.dirname(os.path.abspath(__file__)))
    
    # Start Celery worker
    celery_app.start(argv=[
        "worker",
        "--loglevel=info",
        "--concurrency=4",
        "--queues=document_processing,sentiment_analysis,anomaly_detection,forecasting,model_training,data_processing,monitoring",
        "--hostname=worker@%h"
    ])