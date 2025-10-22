"""
Database optimization API routes
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from ..services.database_optimization_service import db_optimization_service
from ..services.auth_service import get_current_user

router = APIRouter(prefix="/database", tags=["database-optimization"])


class IndexCreationResponse(BaseModel):
    """Response model for index creation"""
    created: List[str]
    failed: List[Dict[str, str]]
    total_attempted: int


class TableStatsResponse(BaseModel):
    """Response model for table statistics"""
    stats: Dict[str, Any]


class ConnectionPoolResponse(BaseModel):
    """Response model for connection pool optimization"""
    current_stats: Dict[str, Any]
    pool_config: Dict[str, Any]
    database_config: Dict[str, Any]
    recommendations: List[str]
    cpu_count: int


class VacuumRequest(BaseModel):
    """Request model for vacuum operations"""
    table_name: Optional[str] = None


@router.post("/indexes/create", response_model=IndexCreationResponse)
async def create_advanced_indexes(current_user: dict = Depends(get_current_user)):
    """Create advanced indexes for optimal query performance"""
    try:
        # Only allow admin users to create indexes
        if current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        result = await db_optimization_service.create_advanced_indexes()
        return IndexCreationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create indexes: {str(e)}")


@router.get("/stats/tables", response_model=TableStatsResponse)
async def get_table_statistics(current_user: dict = Depends(get_current_user)):
    """Get comprehensive table statistics for optimization"""
    try:
        stats = await db_optimization_service.analyze_table_statistics()
        return TableStatsResponse(stats=stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get table statistics: {str(e)}")


@router.get("/connection-pool/optimize", response_model=ConnectionPoolResponse)
async def optimize_connection_pool(current_user: dict = Depends(get_current_user)):
    """Analyze and optimize database connection pool settings"""
    try:
        result = await db_optimization_service.optimize_connection_pool()
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return ConnectionPoolResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to optimize connection pool: {str(e)}")


@router.get("/queries/slow")
async def get_slow_queries(
    limit: int = Query(10, ge=1, le=50),
    current_user: dict = Depends(get_current_user)
):
    """Identify slow queries for optimization"""
    try:
        slow_queries = await db_optimization_service.identify_slow_queries(limit)
        return {
            "slow_queries": slow_queries,
            "count": len(slow_queries),
            "threshold_seconds": db_optimization_service._slow_query_threshold
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get slow queries: {str(e)}")


@router.post("/maintenance/vacuum")
async def vacuum_and_reindex(
    request: VacuumRequest,
    current_user: dict = Depends(get_current_user)
):
    """Perform vacuum and reindex operations"""
    try:
        # Only allow admin users to perform maintenance
        if current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        result = await db_optimization_service.vacuum_and_reindex(request.table_name)
        return {
            "message": "Vacuum and reindex completed",
            "results": result,
            "table_name": request.table_name or "all_tables"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to vacuum and reindex: {str(e)}")


@router.get("/performance/metrics")
async def get_performance_metrics(current_user: dict = Depends(get_current_user)):
    """Get comprehensive database performance metrics"""
    try:
        metrics = await db_optimization_service.get_database_performance_metrics()
        if "error" in metrics:
            raise HTTPException(status_code=500, detail=metrics["error"])
        
        return metrics
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@router.get("/performance/queries")
async def get_query_performance_stats(current_user: dict = Depends(get_current_user)):
    """Get query performance statistics from monitoring"""
    try:
        stats = await db_optimization_service.get_query_performance_stats()
        return {
            "query_stats": stats,
            "monitored_queries": len(stats),
            "slow_query_threshold": db_optimization_service._slow_query_threshold
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get query performance stats: {str(e)}")


@router.get("/health")
async def database_health_check():
    """Comprehensive database health check"""
    try:
        # Get basic database metrics
        metrics = await db_optimization_service.get_database_performance_metrics()
        
        if "error" in metrics:
            raise HTTPException(status_code=503, detail=f"Database health check failed: {metrics['error']}")
        
        # Analyze health indicators
        health_indicators = []
        warnings = []
        
        # Check cache hit ratio
        cache_stats = metrics.get("cache_stats", {})
        buffer_hit_ratio = cache_stats.get("buffer_cache_hit_ratio", 0)
        
        if buffer_hit_ratio < 95:
            warnings.append(f"Low buffer cache hit ratio: {buffer_hit_ratio}%")
        else:
            health_indicators.append(f"Good buffer cache hit ratio: {buffer_hit_ratio}%")
        
        # Check index usage
        index_usage = metrics.get("index_usage", {})
        index_ratio = index_usage.get("index_usage_ratio", 0)
        
        if index_ratio < 80:
            warnings.append(f"Low index usage ratio: {index_ratio}%")
        else:
            health_indicators.append(f"Good index usage ratio: {index_ratio}%")
        
        # Check transaction commit ratio
        transaction_stats = metrics.get("transaction_stats", {})
        commit_ratio = transaction_stats.get("commit_ratio", 0)
        
        if commit_ratio < 95:
            warnings.append(f"Low transaction commit ratio: {commit_ratio}%")
        else:
            health_indicators.append(f"Good transaction commit ratio: {commit_ratio}%")
        
        # Determine overall health status
        if len(warnings) == 0:
            status = "healthy"
        elif len(warnings) <= 2:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "status": status,
            "health_indicators": health_indicators,
            "warnings": warnings,
            "metrics": metrics,
            "timestamp": metrics.get("timestamp")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database health check failed: {str(e)}")