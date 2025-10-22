#!/bin/bash

# Start Celery worker, beat scheduler, and Flower monitoring

echo "Starting Celery services..."

# Start Celery worker in background
echo "Starting Celery worker..."
python start_celery_worker.py &
WORKER_PID=$!

# Start Celery beat scheduler in background
echo "Starting Celery beat scheduler..."
python start_celery_beat.py &
BEAT_PID=$!

# Start Flower monitoring in background
echo "Starting Flower monitoring..."
python start_flower.py &
FLOWER_PID=$!

echo "Celery services started:"
echo "  Worker PID: $WORKER_PID"
echo "  Beat PID: $BEAT_PID"
echo "  Flower PID: $FLOWER_PID"
echo ""
echo "Flower monitoring available at: http://localhost:5555"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo "Stopping Celery services..."
    kill $WORKER_PID $BEAT_PID $FLOWER_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for all background processes
wait