"""
OpenAPI documentation configuration and customization
"""

from typing import Dict, Any
from fastapi.openapi.utils import get_openapi


def custom_openapi_schema(app) -> Dict[str, Any]:
    """Generate custom OpenAPI schema with enhanced documentation"""
    
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Advanced Financial Intelligence System API",
        version="1.0.0",
        description="""
        ## Overview
        
        The Advanced Financial Intelligence System API provides comprehensive AI-powered financial analysis capabilities including:
        
        - **Document Processing**: Upload and analyze financial documents with advanced Q&A capabilities
        - **Sentiment Analysis**: Multi-dimensional sentiment analysis of financial communications
        - **Anomaly Detection**: Statistical anomaly detection in financial metrics and patterns
        - **Forecasting**: Multi-horizon financial forecasting using ensemble methods
        - **Investment Advisory**: Generate actionable investment recommendations with explainable reasoning
        - **Real-time Dashboard**: Live data visualization and interactive financial intelligence
        
        ## Authentication
        
        This API uses JWT (JSON Web Token) authentication. To access protected endpoints:
        
        1. Obtain an access token by calling `/api/auth/login` with valid credentials
        2. Include the token in the Authorization header: `Bearer <your_access_token>`
        3. Refresh tokens when they expire using `/api/auth/refresh`
        
        ## Rate Limiting
        
        API requests are rate-limited to ensure fair usage:
        
        - **Default**: 60 requests per minute
        - **Authenticated users**: 120 requests per minute  
        - **Admin users**: 300 requests per minute
        - **File uploads**: 10 requests per minute
        - **Forecasting**: 30 requests per minute
        
        Rate limit headers are included in responses:
        - `X-RateLimit-Limit`: Request limit per minute
        - `X-RateLimit-Remaining`: Remaining requests in current window
        - `X-RateLimit-Reset`: Unix timestamp when limit resets
        
        ## Error Handling
        
        The API uses standardized error responses with structured error codes:
        
        ```json
        {
            "error": true,
            "error_code": "VAL_001",
            "message": "Validation failed",
            "details": [
                {
                    "field": "ticker",
                    "message": "Invalid ticker format",
                    "code": "invalid_format"
                }
            ],
            "timestamp": "2024-01-01T12:00:00Z",
            "request_id": "abc123"
        }
        ```
        
        ## Data Models
        
        The API uses consistent data models across all endpoints. Key models include:
        
        - **User**: User account information and roles
        - **Document**: Financial document metadata and content
        - **SentimentAnalysis**: Sentiment scores and topic-specific analysis
        - **Anomaly**: Statistical anomaly detection results
        - **Forecast**: Multi-horizon prediction results with confidence intervals
        - **InvestmentRecommendation**: Buy/sell/hold recommendations with reasoning
        
        ## Pagination
        
        List endpoints support pagination with query parameters:
        - `page`: Page number (default: 1)
        - `page_size`: Items per page (default: 20, max: 100)
        
        ## Filtering and Sorting
        
        Many endpoints support filtering and sorting:
        - `filter`: JSON object with filter criteria
        - `sort`: Field name for sorting
        - `order`: Sort order ('asc' or 'desc')
        
        ## WebSocket Support
        
        Real-time data is available through WebSocket connections at `/ws/` endpoints.
        
        ## External Integrations
        
        The system integrates with external data sources:
        - Yahoo Finance API
        - Alpha Vantage API  
        - Quandl API
        - FinanceBench Dataset
        """,
        routes=app.routes,
        tags=[
            {
                "name": "Authentication",
                "description": "User authentication and authorization endpoints"
            },
            {
                "name": "Documents",
                "description": "Document upload, processing, and Q&A capabilities"
            },
            {
                "name": "Sentiment Analysis", 
                "description": "Multi-dimensional sentiment analysis of financial communications"
            },
            {
                "name": "Anomaly Detection",
                "description": "Statistical anomaly detection in financial metrics"
            },
            {
                "name": "Forecasting",
                "description": "Multi-horizon financial forecasting using ensemble methods"
            },
            {
                "name": "Investment Advisory",
                "description": "Investment recommendations with explainable reasoning"
            },
            {
                "name": "Market Data",
                "description": "Real-time and historical market data endpoints"
            },
            {
                "name": "Alerts",
                "description": "Alert management and notification system"
            },
            {
                "name": "WebSocket",
                "description": "Real-time data streaming endpoints"
            },
            {
                "name": "System",
                "description": "System health, monitoring, and administrative endpoints"
            }
        ]
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token obtained from /api/auth/login"
        }
    }
    
    # Add common response schemas
    openapi_schema["components"]["schemas"].update({
        "ErrorResponse": {
            "type": "object",
            "properties": {
                "error": {"type": "boolean", "example": True},
                "error_code": {"type": "string", "example": "VAL_001"},
                "message": {"type": "string", "example": "Validation failed"},
                "details": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string", "example": "ticker"},
                            "message": {"type": "string", "example": "Invalid format"},
                            "code": {"type": "string", "example": "invalid_format"}
                        }
                    }
                },
                "timestamp": {"type": "string", "format": "date-time"},
                "request_id": {"type": "string", "example": "abc123"}
            },
            "required": ["error", "error_code", "message"]
        },
        "PaginatedResponse": {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {}},
                "total": {"type": "integer", "example": 100},
                "page": {"type": "integer", "example": 1},
                "page_size": {"type": "integer", "example": 20},
                "pages": {"type": "integer", "example": 5}
            },
            "required": ["items", "total", "page", "page_size", "pages"]
        },
        "HealthCheck": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "example": "healthy"},
                "version": {"type": "string", "example": "1.0.0"},
                "timestamp": {"type": "string", "format": "date-time"},
                "services": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "example": "healthy"},
                            "response_time": {"type": "number", "example": 0.05}
                        }
                    }
                }
            }
        }
    })
    
    # Add common examples
    openapi_schema["components"]["examples"] = {
        "ValidationError": {
            "summary": "Validation Error Example",
            "value": {
                "error": True,
                "error_code": "VAL_001",
                "message": "Validation failed",
                "details": [
                    {
                        "field": "ticker",
                        "message": "Ticker must be 1-5 uppercase letters",
                        "code": "invalid_format"
                    }
                ],
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "abc123"
            }
        },
        "AuthenticationError": {
            "summary": "Authentication Error Example", 
            "value": {
                "error": True,
                "error_code": "AUTH_001",
                "message": "Authentication required",
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "def456"
            }
        },
        "RateLimitError": {
            "summary": "Rate Limit Error Example",
            "value": {
                "error": True,
                "error_code": "RATE_001", 
                "message": "Rate limit exceeded",
                "context": {
                    "limit": 60,
                    "remaining": 0,
                    "reset": 1704110400
                },
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "ghi789"
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def get_openapi_tags():
    """Get OpenAPI tags for endpoint organization"""
    return [
        {
            "name": "Authentication",
            "description": "User authentication and authorization"
        },
        {
            "name": "Documents", 
            "description": "Document processing and Q&A"
        },
        {
            "name": "Sentiment Analysis",
            "description": "Financial sentiment analysis"
        },
        {
            "name": "Anomaly Detection",
            "description": "Statistical anomaly detection"
        },
        {
            "name": "Forecasting",
            "description": "Financial forecasting and predictions"
        },
        {
            "name": "Investment Advisory",
            "description": "Investment recommendations"
        },
        {
            "name": "Market Data",
            "description": "Market data and real-time feeds"
        },
        {
            "name": "Alerts",
            "description": "Alert and notification management"
        },
        {
            "name": "WebSocket",
            "description": "Real-time WebSocket connections"
        },
        {
            "name": "System",
            "description": "System monitoring and health checks"
        }
    ]