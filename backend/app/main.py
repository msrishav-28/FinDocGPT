import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router
from .core.config import settings
from .middleware.rate_limit_middleware import RateLimitMiddleware
from .middleware.error_middleware import ErrorHandlingMiddleware
from .middleware.monitoring_middleware import MonitoringMiddleware
from .docs.openapi_config import custom_openapi_schema
from .database.init import initialize_database_system, check_database_health
from .services.websocket_service import connection_manager
from .services.market_data_service import market_data_service
from .services.alert_service import alert_service
from .services.audit_service import audit_service
from .services.compliance_service import compliance_service
from .services.cache_service import cache_service

# Import monitoring components
from .monitoring.logger import setup_logging, get_logger
from .monitoring.health import health_monitor
from .monitoring.dashboard import router as monitoring_router
from .routes.health_routes import router as health_router

# Setup comprehensive logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Financial Intelligence System...")
    
    try:
        # Initialize database system
        db_manager, migration_manager, vector_store = await initialize_database_system(
            run_migrations=True,
            initialize_vector_store=True
        )
        
        # Store instances in app state for access in routes
        app.state.db_manager = db_manager
        app.state.migration_manager = migration_manager
        app.state.vector_store = vector_store
        
        logger.info("Database system initialized successfully")
        
        # Check database health
        health = await check_database_health()
        logger.info(f"Database health check: {health.get('status', 'unknown')}")
        
        # Initialize cache service
        await cache_service.initialize()
        logger.info("Cache service initialized")
        
        # Initialize WebSocket connection manager
        await connection_manager.initialize()
        logger.info("WebSocket connection manager initialized")
        
        # Initialize market data service
        await market_data_service.initialize()
        logger.info("Market data service initialized")
        
        # Initialize alert service
        await alert_service.initialize()
        logger.info("Alert service initialized")
        
        # Initialize audit service
        await audit_service.initialize()
        logger.info("Audit service initialized")
        
        # Initialize compliance service
        await compliance_service.initialize()
        logger.info("Compliance service initialized")
        
        # Initialize background task service
        from .services.background_task_service import background_task_service
        background_task_health = await background_task_service.health_check()
        logger.info(f"Background task service health: {background_task_health.get('status', 'unknown')}")
        
        # Start health monitoring
        await health_monitor.start_monitoring()
        logger.info("Health monitoring started")
        
    except Exception as e:
        logger.error(f"Failed to initialize database system: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Financial Intelligence System...")
    
    try:
        # Cleanup WebSocket connections
        await connection_manager.cleanup()
        logger.info("WebSocket connections cleaned up")
        
        # Cleanup market data service
        await market_data_service.cleanup()
        logger.info("Market data service cleaned up")
        
        # Cleanup alert service
        await alert_service.cleanup()
        logger.info("Alert service cleaned up")
        
        # Cleanup cache service
        await cache_service.cleanup()
        logger.info("Cache service cleaned up")
        
        # Stop health monitoring
        await health_monitor.stop_monitoring()
        logger.info("Health monitoring stopped")
        
        # Close database connections
        from .database.connection import close_database
        await close_database()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


app = FastAPI(
    title="Advanced Financial Intelligence System",
    description="AI-powered platform for financial document analysis, market prediction, and investment decision-making",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for development and docker-compose usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Error handling middleware
app.add_middleware(ErrorHandlingMiddleware)

# Monitoring middleware
app.add_middleware(MonitoringMiddleware)

# Include API routes
app.include_router(router, prefix="/api")

# Include monitoring routes
app.include_router(monitoring_router, prefix="/api")

# Include health check routes
app.include_router(health_router)

# Set custom OpenAPI schema
app.openapi = lambda: custom_openapi_schema(app)


@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Advanced Financial Intelligence System",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Basic health check endpoint - use /api/monitoring/health for comprehensive checks"""
    try:
        health = await check_database_health()
        return {
            "status": "healthy",
            "database": health,
            "version": "1.0.0",
            "message": "Use /api/monitoring/health for comprehensive health checks"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "version": "1.0.0"
        }
