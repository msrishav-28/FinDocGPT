#!/usr/bin/env python3
"""
Celery beat (scheduler) startup script
"""

import os
import sys
from app.celery_app import celery_app

if __name__ == "__main__":
    # Set up environment
    os.environ.setdefault("PYTHONPATH", os.path.dirname(os.path.abspath(__file__)))
    
    # Start Celery beat scheduler
    celery_app.start(argv=[
        "beat",
        "--loglevel=info",
        "--schedule=/tmp/celerybeat-schedule",
        "--pidfile=/tmp/celerybeat.pid"
    ])