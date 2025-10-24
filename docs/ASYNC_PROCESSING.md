# Asynchronous Processing with Celery

This document describes the asynchronous processing system implemented using Celery for the Advanced Financial Intelligence System.

## Overview

The system uses Celery with Redis as the message broker to handle computationally intensive tasks asynchronously. This allows the main API to remain responsive while background tasks process documents, train models, and perform complex analyses.

## Architecture

### Components

1. **Celery Worker**: Executes background tasks
2. **Celery Beat**: Schedules periodic tasks
3. **Redis**: Message broker and result backend
4. **Flower**: Web-based monitoring tool

### Task Queues

- `document_processing`: Document ingestion and analysis
- `sentiment_analysis`: Sentiment analysis tasks
- `anomaly_detection`: Anomaly detection tasks
- `forecasting`: Stock price forecasting
- `model_training`: ML model training and retraining
- `data_processing`: Data synchronization and processing
- `monitoring`: System monitoring and maintenance

## Task Categories

### Document Processing Tasks
- `process_document`: Process individual documents
- `batch_process_documents`: Process multiple documents
- `update_document_index`: Update document search index
- `generate_document_summary`: Generate document summaries

### Sentiment Analysis Tasks
- `analyze_document_sentiment`: Analyze document sentiment
- `batch_sentiment_analysis`: Batch sentiment analysis
- `update_sentiment_trends`: Update sentiment trends
- `compare_company_sentiment`: Compare sentiment across companies

### Anomaly Detection Tasks
- `detect_metric_anomalies`: Detect statistical anomalies
- `pattern_anomaly_detection`: Detect pattern-based anomalies
- `systemic_risk_analysis`: Analyze systemic risk
- `update_anomaly_baselines`: Update detection baselines

### Forecasting Tasks
- `generate_stock_forecast`: Generate stock price forecasts
- `batch_forecast_generation`: Generate multiple forecasts
- `update_forecast_models`: Update forecasting models
- `evaluate_forecast_accuracy`: Evaluate model accuracy

### Model Training Tasks
- `retrain_sentiment_model`: Retrain sentiment models
- `retrain_forecasting_models`: Retrain forecasting models
- `retrain_anomaly_models`: Retrain anomaly detection models
- `optimize_model_hyperparameters`: Optimize model parameters

### Data Processing Tasks
- `update_market_data`: Update market data
- `sync_external_data`: Synchronize external data sources
- `process_financial_reports`: Process financial reports
- `calculate_financial_metrics`: Calculate financial metrics
- `data_quality_check`: Perform data quality checks
- `cleanup_old_data`: Clean up old data

### Monitoring Tasks
- `system_health_check`: Comprehensive health check
- `cleanup_old_results`: Clean up old task results
- `performance_monitoring`: Monitor system performance
- `cache_warming`: Warm up cache
- `generate_system_report`: Generate system reports
- `alert_processing`: Process system alerts

## Periodic Tasks

The system includes several periodic tasks scheduled via Celery Beat:

- **Market Data Update**: Every 5 minutes during market hours
- **Model Retraining**: Daily
- **System Health Check**: Every 10 minutes
- **Cleanup Old Results**: Hourly

## Usage

### Starting Services

#### Development (Local)

```bash
# Start Redis
redis-server

# Start Celery worker
python start_celery_worker.py

# Start Celery beat scheduler
python start_celery_beat.py

# Start Flower monitoring
python start_flower.py
```

#### Using Scripts

```bash
# Linux/Mac
./start_celery.sh

# Windows
start_celery.bat
```

#### Docker Compose

```bash
docker-compose up
```

This will start all services including Redis, Celery worker, beat scheduler, and Flower.

### API Usage

#### Starting a Background Task

```python
# Example: Process a document asynchronously
import requests

response = requests.post("http://localhost:8000/api/background-tasks/document/process", 
    json={
        "document_data": {
            "content": "Document content here...",
            "metadata": {
                "company": "AAPL",
                "document_type": "earnings_report",
                "filing_date": "2024-01-01T00:00:00Z"
            }
        }
    },
    headers={"Authorization": "Bearer your-token"}
)

task_id = response.json()["task_id"]
```

#### Checking Task Status

```python
# Check task status
status_response = requests.get(f"http://localhost:8000/api/background-tasks/task/{task_id}/status",
    headers={"Authorization": "Bearer your-token"}
)

status = status_response.json()
print(f"Task status: {status['status']}")
print(f"Result: {status['result']}")
```

#### Canceling a Task

```python
# Cancel a task
cancel_response = requests.delete(f"http://localhost:8000/api/background-tasks/task/{task_id}",
    headers={"Authorization": "Bearer your-token"}
)
```

### Monitoring

#### Flower Web Interface

Access the Flower monitoring interface at `http://localhost:5555` to:
- View active tasks
- Monitor worker status
- See task history
- View task details and results

#### API Endpoints

- `GET /api/background-tasks/tasks/active`: Get active tasks
- `GET /api/background-tasks/workers/stats`: Get worker statistics
- `GET /api/background-tasks/queues/lengths`: Get queue lengths
- `GET /api/background-tasks/health`: Health check

## Configuration

### Environment Variables

```bash
# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
CELERY_TASK_SERIALIZER=json
CELERY_RESULT_SERIALIZER=json
CELERY_TIMEZONE=UTC
CELERY_WORKER_PREFETCH_MULTIPLIER=1
CELERY_TASK_ACKS_LATE=true
CELERY_WORKER_MAX_TASKS_PER_CHILD=1000
FLOWER_PORT=5555
```

### Task Routing

Tasks are automatically routed to appropriate queues based on their type:
- Document tasks → `document_processing` queue
- Sentiment tasks → `sentiment_analysis` queue
- Anomaly tasks → `anomaly_detection` queue
- Forecasting tasks → `forecasting` queue
- Model training tasks → `model_training` queue
- Data processing tasks → `data_processing` queue
- Monitoring tasks → `monitoring` queue

### Rate Limiting

- Default rate limit: 100 tasks per minute
- Model training tasks: 10 tasks per hour
- Market data updates: 20 tasks per minute

## Error Handling

### Task Retry Logic

Tasks automatically retry on failure with exponential backoff:
- Maximum retries: 3
- Retry delay: 60 seconds (exponential backoff)
- Retry on specific exceptions only

### Error Monitoring

- Failed tasks are logged with full error details
- Task status includes error information
- Flower interface shows failed task details
- System health checks monitor task failure rates

## Performance Optimization

### Worker Configuration

- **Concurrency**: 4 workers per process (configurable)
- **Prefetch**: 1 task per worker (prevents memory issues)
- **Max tasks per child**: 1000 (prevents memory leaks)
- **Task acknowledgment**: Late acknowledgment for reliability

### Memory Management

- Workers restart after processing 1000 tasks
- Large results are stored in Redis with expiration
- Temporary files are cleaned up automatically

### Scaling

#### Horizontal Scaling

Add more worker processes:
```bash
# Start additional workers
celery -A app.celery_app worker --concurrency=8 --queues=document_processing
celery -A app.celery_app worker --concurrency=4 --queues=model_training
```

#### Queue-Specific Workers

Dedicate workers to specific queues:
```bash
# High-priority document processing worker
celery -A app.celery_app worker --queues=document_processing --concurrency=8

# Resource-intensive model training worker
celery -A app.celery_app worker --queues=model_training --concurrency=2
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Check Redis is running: `redis-cli ping`
   - Verify connection settings in environment variables

2. **Tasks Not Processing**
   - Check worker status: `celery -A app.celery_app inspect active`
   - Verify queue configuration
   - Check worker logs for errors

3. **High Memory Usage**
   - Reduce worker concurrency
   - Lower `CELERY_WORKER_MAX_TASKS_PER_CHILD`
   - Monitor task memory usage

4. **Task Timeouts**
   - Increase task timeout limits
   - Optimize task implementation
   - Split large tasks into smaller chunks

### Debugging

#### Enable Debug Logging

```bash
# Start worker with debug logging
celery -A app.celery_app worker --loglevel=debug
```

#### Monitor Task Execution

```python
# Get detailed task information
from app.celery_app import celery_app
inspect = celery_app.control.inspect()

# Active tasks
active = inspect.active()

# Scheduled tasks
scheduled = inspect.scheduled()

# Worker stats
stats = inspect.stats()
```

## Security Considerations

### Authentication

- All API endpoints require authentication
- Task results are isolated per user
- Sensitive data is encrypted in transit

### Data Protection

- Task arguments and results are serialized securely
- Temporary files are cleaned up automatically
- Access logs are maintained for audit purposes

### Network Security

- Redis should be secured with authentication
- Use TLS for production deployments
- Restrict network access to Celery services

## Best Practices

1. **Task Design**
   - Keep tasks idempotent
   - Handle failures gracefully
   - Use appropriate timeouts
   - Log important events

2. **Resource Management**
   - Monitor memory usage
   - Clean up temporary resources
   - Use connection pooling

3. **Monitoring**
   - Set up alerts for failed tasks
   - Monitor queue lengths
   - Track task execution times

4. **Deployment**
   - Use process managers (systemd, supervisor)
   - Implement health checks
   - Plan for graceful shutdowns