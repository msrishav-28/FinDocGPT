@echo off
REM Start Celery services on Windows

echo Starting Celery services...

REM Start Celery worker
echo Starting Celery worker...
start "Celery Worker" python start_celery_worker.py

REM Start Celery beat scheduler
echo Starting Celery beat scheduler...
start "Celery Beat" python start_celery_beat.py

REM Start Flower monitoring
echo Starting Flower monitoring...
start "Flower" python start_flower.py

echo.
echo Celery services started in separate windows
echo Flower monitoring available at: http://localhost:5555
echo.
echo Close the individual windows to stop services
pause