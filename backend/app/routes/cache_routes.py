"""
Cache management API routes
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from ..services.cache_service import cache_service
from ..services.auth_service import get_current_user

router = APIRouter(prefix="/cache", tags=["cache"])


class CacheInvalidationRequest(BaseModel):
    """Request model for cache invalidation"""
    patterns: List[str]


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics"""
    service_stats: Dict[str, Any]
    redis_info: Dict[str, Any]
    cache_warming_tasks: int
    ttl_settings: Dict[str, int]


@router.get("/stats", response_model=CacheStatsResponse)
async def get_cache_stats(current_user: dict = Depends(get_current_user)):
    """Get comprehensive cache statistics"""
    try:
        stats = await cache_service.get_stats()
        return CacheStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@router.post("/invalidate/document/{doc_id}")
async def invalidate_document_cache(
    doc_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Invalidate all cache related to a specific document"""
    try:
        await cache_service.invalidate_document_cache(doc_id)
        return {"message": f"Document cache invalidated for doc_id: {doc_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to invalidate document cache: {str(e)}")


@router.post("/invalidate/company/{company}")
async def invalidate_company_cache(
    company: str,
    current_user: dict = Depends(get_current_user)
):
    """Invalidate all cache related to a specific company"""
    try:
        await cache_service.invalidate_company_cache(company)
        return {"message": f"Company cache invalidated for: {company}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to invalidate company cache: {str(e)}")


@router.post("/invalidate/market/{ticker}")
async def invalidate_market_cache(
    ticker: str,
    current_user: dict = Depends(get_current_user)
):
    """Invalidate all cache related to a specific ticker"""
    try:
        await cache_service.invalidate_market_cache(ticker)
        return {"message": f"Market cache invalidated for ticker: {ticker}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to invalidate market cache: {str(e)}")


@router.post("/invalidate/patterns")
async def invalidate_cache_patterns(
    request: CacheInvalidationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Invalidate cache using custom patterns"""
    try:
        total_deleted = 0
        for pattern in request.patterns:
            deleted = await cache_service.delete_pattern(pattern)
            total_deleted += deleted
        
        return {
            "message": f"Cache invalidated for {len(request.patterns)} patterns",
            "total_keys_deleted": total_deleted,
            "patterns": request.patterns
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to invalidate cache patterns: {str(e)}")


@router.delete("/clear")
async def clear_all_cache(current_user: dict = Depends(get_current_user)):
    """Clear all cache (use with extreme caution)"""
    try:
        # Only allow admin users to clear all cache
        if current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        success = await cache_service.clear_all()
        if success:
            return {"message": "All cache cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear cache")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/health")
async def cache_health_check():
    """Check cache service health"""
    try:
        # Test basic cache operations
        test_key = "health_check_test"
        test_value = {"timestamp": "test", "status": "ok"}
        
        # Test set
        set_success = await cache_service.set(test_key, test_value, ttl=60)
        if not set_success:
            raise HTTPException(status_code=503, detail="Cache set operation failed")
        
        # Test get
        retrieved_value = await cache_service.get(test_key)
        if retrieved_value != test_value:
            raise HTTPException(status_code=503, detail="Cache get operation failed")
        
        # Test delete
        delete_success = await cache_service.delete(test_key)
        if not delete_success:
            raise HTTPException(status_code=503, detail="Cache delete operation failed")
        
        return {
            "status": "healthy",
            "message": "Cache service is operational",
            "operations_tested": ["set", "get", "delete"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cache health check failed: {str(e)}")