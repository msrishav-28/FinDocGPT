# API Reference

This document provides a comprehensive reference for the Financial Intelligence System REST API.

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base URL and Versioning](#base-url-and-versioning)
4. [Request/Response Format](#requestresponse-format)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [API Endpoints](#api-endpoints)
8. [WebSocket API](#websocket-api)
9. [SDK and Client Libraries](#sdk-and-client-libraries)

## Overview

The Financial Intelligence System API is a RESTful API that provides access to AI-powered financial document analysis, market data, forecasting, and investment recommendation services. The API is built with FastAPI and follows OpenAPI 3.0 specifications.

### Key Features

- **Document Processing**: Upload, analyze, and query financial documents
- **Sentiment Analysis**: Analyze sentiment in financial reports and commentary
- **Market Data**: Real-time and historical market data integration
- **Forecasting**: Time-series forecasting for stock prices and market trends
- **Investment Advisory**: AI-powered investment recommendations
- **Real-time Updates**: WebSocket support for live data streaming
- **Comprehensive Monitoring**: Health checks, metrics, and audit trails

## Authentication

The API uses JWT (JSON Web Token) based authentication with OAuth2 password flow.

### Authentication Flow

1. **Login**: POST `/api/auth/login` with username/password
2. **Receive Tokens**: Get access token and refresh token
3. **Use Access Token**: Include in Authorization header for API requests
4. **Refresh Token**: Use refresh token to get new access token when expired

### Example Authentication

```bash
# Login
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user@example.com",
    "password": "password"
  }'

# Response
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}

# Use token in requests
curl -X GET "http://localhost:8000/api/documents/search" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
```

### Token Management

- **Access Token**: Valid for 1 hour, used for API requests
- **Refresh Token**: Valid for 30 days, used to get new access tokens
- **Token Refresh**: POST `/api/auth/refresh` with refresh token

## Base URL and Versioning

- **Base URL**: `http://localhost:8000/api` (development)
- **Production URL**: `https://your-domain.com/api`
- **API Version**: v1 (current)
- **OpenAPI Docs**: `/docs` (Swagger UI)
- **ReDoc**: `/redoc` (Alternative documentation)

## Request/Response Format

### Content Types

- **Request**: `application/json` or `multipart/form-data` (file uploads)
- **Response**: `application/json`

### Standard Response Format

```json
{
  "data": {},
  "message": "Success",
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "uuid-string"
}
```

### Pagination

```json
{
  "data": [],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 100,
    "pages": 5
  }
}
```

## Error Handling

### HTTP Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Server Error

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "email",
      "issue": "Invalid email format"
    }
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "uuid-string"
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage and system stability.

### Rate Limits

- **Default**: 100 requests per minute
- **Authenticated**: 1000 requests per minute
- **File Upload**: 10 requests per minute
- **Heavy Operations**: 5 requests per minute

### Rate Limit Headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## API Endpoints

### Health and Status

#### GET /health
Basic health check for load balancers.

```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "service": "financial-intelligence-backend",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### GET /health/detailed
Comprehensive health check with system status.

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "celery": "healthy"
  },
  "metrics": {
    "uptime": 3600,
    "memory_usage": "45%",
    "cpu_usage": "12%"
  }
}
```

### Authentication Endpoints

#### POST /api/auth/login
Authenticate user and receive JWT tokens.

**Request:**
```json
{
  "username": "user@example.com",
  "password": "password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### POST /api/auth/refresh
Refresh access token using refresh token.

**Request:**
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

#### GET /api/auth/me
Get current user information.

**Headers:** `Authorization: Bearer <token>`

**Response:**
```json
{
  "id": "uuid",
  "username": "user@example.com",
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "roles": ["user"]
}
```

### Document Processing

#### POST /api/documents/upload
Upload and process a financial document.

**Request:** `multipart/form-data`
- `file`: Document file (PDF, TXT)
- `company`: Company name
- `document_type`: Document type (10-K, 10-Q, earnings_call, etc.)
- `filing_date`: Filing date (ISO format)
- `period`: Reporting period (optional)
- `source`: Data source

**Response:**
```json
{
  "document_id": "uuid",
  "message": "Document uploaded and processed successfully",
  "company": "Apple Inc.",
  "document_type": "10-K",
  "status": "processing"
}
```

#### POST /api/documents/search
Search documents using vector similarity or text search.

**Request:**
```json
{
  "query": "revenue growth",
  "companies": ["Apple Inc.", "Microsoft"],
  "document_types": ["10-K", "10-Q"],
  "date_from": "2023-01-01T00:00:00Z",
  "date_to": "2024-01-01T00:00:00Z",
  "min_confidence": 0.3,
  "use_vector_search": true
}
```

**Response:**
```json
{
  "results": [
    {
      "document_id": "uuid",
      "company": "Apple Inc.",
      "document_type": "10-K",
      "confidence": 0.95,
      "excerpt": "Revenue increased by 15% year-over-year...",
      "filing_date": "2023-10-01T00:00:00Z"
    }
  ]
}
```

#### POST /api/documents/qa
Ask questions about financial documents.

**Request:**
```json
{
  "question": "What was the revenue growth in Q3?",
  "company": "Apple Inc.",
  "document_types": ["10-Q"],
  "include_related": true,
  "max_documents": 10
}
```

**Response:**
```json
{
  "answer": "Apple's revenue grew by 15% in Q3 2023...",
  "confidence": 0.92,
  "sources": [
    {
      "document_id": "uuid",
      "company": "Apple Inc.",
      "excerpt": "Q3 revenue was $89.5 billion..."
    }
  ],
  "context": "Based on Apple's Q3 2023 10-Q filing"
}
```

### Sentiment Analysis

#### POST /api/sentiment/analyze
Analyze sentiment of financial text or documents.

**Request:**
```json
{
  "text": "The company reported strong earnings with revenue exceeding expectations",
  "document_id": "uuid",
  "analysis_type": "comprehensive"
}
```

**Response:**
```json
{
  "sentiment": {
    "overall": "positive",
    "score": 0.85,
    "confidence": 0.92
  },
  "aspects": {
    "earnings": {"sentiment": "positive", "score": 0.9},
    "revenue": {"sentiment": "positive", "score": 0.8}
  },
  "keywords": ["strong", "exceeding", "expectations"]
}
```

### Market Data

#### GET /api/market-data/quote/{symbol}
Get real-time quote for a stock symbol.

**Response:**
```json
{
  "symbol": "AAPL",
  "price": 150.25,
  "change": 2.15,
  "change_percent": 1.45,
  "volume": 45000000,
  "timestamp": "2024-01-01T16:00:00Z"
}
```

#### GET /api/market-data/historical/{symbol}
Get historical price data.

**Parameters:**
- `period`: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
- `interval`: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

**Response:**
```json
{
  "symbol": "AAPL",
  "data": [
    {
      "date": "2024-01-01",
      "open": 148.50,
      "high": 151.20,
      "low": 147.80,
      "close": 150.25,
      "volume": 45000000
    }
  ]
}
```

### Forecasting

#### POST /api/forecast/stock
Generate stock price forecasts.

**Request:**
```json
{
  "symbol": "AAPL",
  "forecast_days": 30,
  "model": "prophet",
  "include_confidence_intervals": true
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "forecast": [
    {
      "date": "2024-01-02",
      "predicted_price": 152.30,
      "lower_bound": 148.50,
      "upper_bound": 156.10,
      "confidence": 0.85
    }
  ],
  "model_metrics": {
    "mae": 2.15,
    "rmse": 3.42,
    "mape": 1.8
  }
}
```

### Investment Advisory

#### POST /api/investment/recommend
Get AI-powered investment recommendations.

**Request:**
```json
{
  "symbol": "AAPL",
  "analysis_type": "comprehensive",
  "risk_tolerance": "moderate",
  "investment_horizon": "long_term"
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "recommendation": "BUY",
  "confidence": 0.88,
  "target_price": 165.00,
  "reasoning": [
    "Strong financial fundamentals",
    "Positive market sentiment",
    "Growing market share in key segments"
  ],
  "risk_factors": [
    "Market volatility",
    "Regulatory changes"
  ],
  "metrics": {
    "pe_ratio": 25.4,
    "debt_to_equity": 0.31,
    "roe": 0.26
  }
}
```

### Monitoring and Analytics

#### GET /api/monitoring/health
Comprehensive system health check.

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "database": {"status": "healthy", "response_time": 15},
    "redis": {"status": "healthy", "response_time": 5},
    "celery": {"status": "healthy", "active_tasks": 3}
  },
  "metrics": {
    "uptime": 86400,
    "memory_usage": 45.2,
    "cpu_usage": 12.8,
    "disk_usage": 23.1
  }
}
```

#### GET /api/monitoring/metrics
Get system performance metrics.

**Response:**
```json
{
  "requests": {
    "total": 10000,
    "per_minute": 150,
    "success_rate": 99.5
  },
  "response_times": {
    "avg": 245,
    "p50": 180,
    "p95": 450,
    "p99": 800
  },
  "errors": {
    "total": 50,
    "rate": 0.5
  }
}
```

### Background Tasks

#### POST /api/tasks/submit
Submit a background task for processing.

**Request:**
```json
{
  "task_type": "document_analysis",
  "parameters": {
    "document_id": "uuid",
    "analysis_types": ["sentiment", "entities", "summary"]
  },
  "priority": "normal"
}
```

**Response:**
```json
{
  "task_id": "uuid",
  "status": "queued",
  "estimated_completion": "2024-01-01T12:05:00Z"
}
```

#### GET /api/tasks/{task_id}/status
Get background task status.

**Response:**
```json
{
  "task_id": "uuid",
  "status": "completed",
  "progress": 100,
  "result": {
    "sentiment_score": 0.85,
    "entities": ["Apple Inc.", "Q3 2023"],
    "summary": "Strong quarterly performance..."
  },
  "started_at": "2024-01-01T12:00:00Z",
  "completed_at": "2024-01-01T12:04:30Z"
}
```

## WebSocket API

The system provides real-time updates through WebSocket connections.

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

// Authentication
ws.send(JSON.stringify({
  type: 'auth',
  token: 'your-jwt-token'
}));
```

### Subscription Types

#### Market Data Updates
```javascript
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'market_data',
  symbols: ['AAPL', 'MSFT', 'GOOGL']
}));
```

#### Document Processing Updates
```javascript
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'document_processing',
  document_ids: ['uuid1', 'uuid2']
}));
```

#### System Alerts
```javascript
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'alerts',
  alert_types: ['system', 'market', 'portfolio']
}));
```

### Message Format

```javascript
{
  "type": "market_data",
  "data": {
    "symbol": "AAPL",
    "price": 150.25,
    "change": 2.15,
    "timestamp": "2024-01-01T16:00:00Z"
  },
  "timestamp": "2024-01-01T16:00:00Z"
}
```

## SDK and Client Libraries

### JavaScript/TypeScript SDK

```bash
npm install @financial-intelligence/sdk
```

```javascript
import { FinancialIntelligenceClient } from '@financial-intelligence/sdk';

const client = new FinancialIntelligenceClient({
  baseUrl: 'http://localhost:8000/api',
  apiKey: 'your-api-key'
});

// Upload document
const result = await client.documents.upload({
  file: fileBlob,
  company: 'Apple Inc.',
  documentType: '10-K',
  filingDate: new Date('2023-10-01')
});

// Search documents
const searchResults = await client.documents.search({
  query: 'revenue growth',
  companies: ['Apple Inc.']
});

// Get stock forecast
const forecast = await client.forecast.stock({
  symbol: 'AAPL',
  forecastDays: 30
});
```

### Python SDK

```bash
pip install financial-intelligence-sdk
```

```python
from financial_intelligence import Client

client = Client(
    base_url='http://localhost:8000/api',
    api_key='your-api-key'
)

# Upload document
result = client.documents.upload(
    file=open('document.pdf', 'rb'),
    company='Apple Inc.',
    document_type='10-K',
    filing_date='2023-10-01'
)

# Search documents
results = client.documents.search(
    query='revenue growth',
    companies=['Apple Inc.']
)

# Get investment recommendation
recommendation = client.investment.recommend(
    symbol='AAPL',
    risk_tolerance='moderate'
)
```

## API Limits and Quotas

### Request Limits

- **File Upload**: Maximum 100MB per file
- **Batch Operations**: Maximum 100 items per batch
- **Query Length**: Maximum 1000 characters
- **Response Size**: Maximum 10MB per response

### Usage Quotas

- **Free Tier**: 1,000 requests per month
- **Professional**: 50,000 requests per month
- **Enterprise**: Unlimited requests

## Webhooks

Configure webhooks to receive notifications about events.

### Webhook Events

- `document.processed` - Document processing completed
- `forecast.generated` - Forecast generation completed
- `alert.triggered` - System or market alert triggered
- `task.completed` - Background task completed

### Webhook Configuration

```json
{
  "url": "https://your-app.com/webhooks/financial-intelligence",
  "events": ["document.processed", "alert.triggered"],
  "secret": "webhook-secret-key"
}
```

### Webhook Payload

```json
{
  "event": "document.processed",
  "data": {
    "document_id": "uuid",
    "company": "Apple Inc.",
    "status": "completed",
    "processing_time": 45.2
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "signature": "sha256=..."
}
```

## Support and Resources

- **Interactive Documentation**: `/docs` (Swagger UI)
- **Alternative Documentation**: `/redoc`
- **OpenAPI Specification**: `/openapi.json`
- **Status Page**: `https://status.your-domain.com`
- **Support**: `support@your-domain.com`

For additional help and examples, please refer to the comprehensive documentation and example implementations in the project repository.