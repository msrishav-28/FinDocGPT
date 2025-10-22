#!/usr/bin/env python3
"""
Flower monitoring startup script
"""

import os
import sys
from app.celery_app import celery_app

if __name__ == "__main__":
    # Set up environment
    os.environ.setdefault("PYTHONPATH", os.path.dirname(os.path.abspath(__file__)))
    
    # Start Flower monitoring
    celery_app.start(argv=[
        "flower",
        "--port=5555",
        "--broker=redis://localhost:6379/1"
    ])