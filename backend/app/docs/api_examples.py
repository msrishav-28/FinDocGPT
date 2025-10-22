"""
API documentation examples and response schemas
"""

# Authentication Examples
AUTH_EXAMPLES = {
    "login_request": {
        "summary": "Login Request",
        "value": {
            "username": "analyst_user",
            "password": "SecurePass123"
        }
    },
    "login_response": {
        "summary": "Login Response",
        "value": {
            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "token_type": "bearer",
            "expires_in": 1800
        }
    },
    "user_info": {
        "summary": "User Information",
        "value": {
            "id": "user123",
            "email": "analyst@company.com",
            "username": "analyst_user",
            "full_name": "Financial Analyst",
            "role": "analyst",
            "is_active": True,
            "created_at": "2024-01-01T00:00:00Z",
            "last_login": "2024-01-15T10:30:00Z"
        }
    }
}

# Document Processing Examples
DOCUMENT_EXAMPLES = {
    "upload_response": {
        "summary": "Document Upload Response",
        "value": {
            "document_id": "doc_abc123",
            "filename": "earnings_report_q4_2023.pdf",
            "company": "AAPL",
            "document_type": "earnings_report",
            "status": "processing",
            "upload_timestamp": "2024-01-15T10:30:00Z"
        }
    },
    "qa_request": {
        "summary": "Q&A Request",
        "value": {
            "question": "What was the revenue growth in Q4 2023?",
            "document_id": "doc_abc123",
            "context": {
                "company": "AAPL",
                "include_related_docs": True
            }
        }
    },
    "qa_response": {
        "summary": "Q&A Response",
        "value": {
            "answer": "Apple reported revenue growth of 2.1% year-over-year in Q4 2023, reaching $119.6 billion.",
            "confidence": 0.92,
            "sources": [
                {
                    "document_id": "doc_abc123",
                    "page": 2,
                    "section": "Financial Highlights"
                }
            ],
            "related_questions": [
                "What drove the revenue growth in Q4 2023?",
                "How does this compare to previous quarters?"
            ]
        }
    }
}

# Sentiment Analysis Examples
SENTIMENT_EXAMPLES = {
    "sentiment_response": {
        "summary": "Sentiment Analysis Response",
        "value": {
            "overall_sentiment": 0.65,
            "confidence": 0.88,
            "topic_sentiments": {
                "management_outlook": 0.72,
                "financial_performance": 0.58,
                "market_conditions": 0.45,
                "competitive_position": 0.68
            },
            "sentiment_explanation": "The document expresses positive sentiment overall, with particularly optimistic management outlook and strong competitive positioning.",
            "historical_comparison": 0.12,
            "analysis_timestamp": "2024-01-15T10:35:00Z"
        }
    }
}

# Forecasting Examples
FORECAST_EXAMPLES = {
    "forecast_request": {
        "summary": "Forecast Request",
        "value": {
            "ticker": "AAPL",
            "horizons": [1, 3, 6, 12],
            "include_confidence_intervals": True,
            "model_ensemble": ["prophet", "arima", "lstm"]
        }
    },
    "forecast_response": {
        "summary": "Forecast Response",
        "value": {
            "ticker": "AAPL",
            "forecasts": {
                "1": 185.50,
                "3": 192.30,
                "6": 198.75,
                "12": 210.20
            },
            "confidence_intervals": {
                "1": [180.25, 190.75],
                "3": [185.10, 199.50],
                "6": [188.20, 209.30],
                "12": [195.80, 224.60]
            },
            "model_performance": {
                "prophet": 0.85,
                "arima": 0.78,
                "lstm": 0.82
            },
            "last_updated": "2024-01-15T10:40:00Z"
        }
    }
}

# Investment Advisory Examples
ADVISORY_EXAMPLES = {
    "recommendation_response": {
        "summary": "Investment Recommendation",
        "value": {
            "ticker": "AAPL",
            "signal": "BUY",
            "confidence": 0.78,
            "target_price": 195.00,
            "risk_level": "moderate",
            "position_size": 0.05,
            "reasoning": "Strong fundamentals, positive sentiment, and favorable technical indicators support a buy recommendation.",
            "supporting_factors": [
                "Revenue growth acceleration",
                "Positive management outlook",
                "Strong competitive position",
                "Favorable market conditions"
            ],
            "risk_factors": [
                "Market volatility",
                "Regulatory concerns",
                "Supply chain risks"
            ],
            "time_horizon": "3-6 months",
            "recommendation_timestamp": "2024-01-15T10:45:00Z"
        }
    }
}

# Monitoring Examples
MONITORING_EXAMPLES = {
    "health_check": {
        "summary": "Health Check Response",
        "value": {
            "status": "healthy",
            "timestamp": "2024-01-15T10:50:00Z",
            "version": "1.0.0",
            "uptime": 86400,
            "services": {
                "database": {
                    "status": "healthy",
                    "last_check": "2024-01-15T10:50:00Z",
                    "response_time": 0.025
                },
                "redis": {
                    "status": "healthy",
                    "last_check": "2024-01-15T10:50:00Z",
                    "response_time": 0.008
                },
                "external_apis": {
                    "status": "healthy",
                    "last_check": "2024-01-15T10:50:00Z",
                    "response_time": 0.150
                }
            }
        }
    },
    "metrics_response": {
        "summary": "Performance Metrics",
        "value": {
            "request_metrics": {
                "total_requests": 15420,
                "requests_last_hour": 245,
                "avg_response_time": 0.125,
                "min_response_time": 0.008,
                "max_response_time": 2.350,
                "error_rate": 0.02,
                "requests_per_minute": 4.08
            },
            "system_metrics": {
                "cpu_usage": 25.5,
                "memory_usage": {
                    "total": 8589934592,
                    "available": 4294967296,
                    "percent": 50.0,
                    "used": 4294967296
                },
                "disk_usage": {
                    "total": 107374182400,
                    "used": 32212254720,
                    "free": 75161927680,
                    "percent": 30.0
                },
                "uptime": 86400
            }
        }
    }
}

# Error Examples
ERROR_EXAMPLES = {
    "validation_error": {
        "summary": "Validation Error",
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
            "timestamp": "2024-01-15T10:55:00Z",
            "request_id": "req_xyz789"
        }
    },
    "authentication_error": {
        "summary": "Authentication Error",
        "value": {
            "error": True,
            "error_code": "AUTH_001",
            "message": "Authentication required",
            "timestamp": "2024-01-15T10:55:00Z",
            "request_id": "req_abc123"
        }
    },
    "rate_limit_error": {
        "summary": "Rate Limit Error",
        "value": {
            "error": True,
            "error_code": "RATE_001",
            "message": "Rate limit exceeded",
            "context": {
                "limit": 60,
                "remaining": 0,
                "reset": 1705315200
            },
            "timestamp": "2024-01-15T10:55:00Z",
            "request_id": "req_def456"
        }
    }
}