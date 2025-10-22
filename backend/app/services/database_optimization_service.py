"""
Database optimization service for the Financial Intelligence System

This service provides:
- Advanced indexing strategies for optimal query performance
- Query optimization for complex analytical queries
- Connection pooling and management optimization
- Database performance monitoring and tuning
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import asyncpg
from asyncpg import Pool, Connection

from ..config import get_settings
from ..database.connection import db_manager

logger = logging.getLogger(__name__)
settings = get_settings()


class QueryOptimizer:
    """Query optimization utilities"""
    
    @staticmethod
    def build_document_search_query(
        company: Optional[str] = None,
        document_type: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[str, List[Any]]:
        """Build optimized document search query with proper indexing"""
        
        base_query = """
        SELECT 
            id, company, document_type, filing_date, period, source,
            summary, processing_status, created_at
        FROM documents
        """
        
        conditions = []
        params = []
        param_count = 0
        
        if company:
            param_count += 1
            conditions.append(f"company = ${param_count}")
            params.append(company)
        
        if document_type:
            param_count += 1
            conditions.append(f"document_type = ${param_count}")
            params.append(document_type)
        
        if date_range:
            param_count += 1
            conditions.append(f"filing_date >= ${param_count}")
            params.append(date_range[0])
            
            param_count += 1
            conditions.append(f"filing_date <= ${param_count}")
            params.append(date_range[1])
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        # Add ordering for consistent results and index usage
        base_query += " ORDER BY filing_date DESC, created_at DESC"
        
        # Add pagination
        param_count += 1
        base_query += f" LIMIT ${param_count}"
        params.append(limit)
        
        param_count += 1
        base_query += f" OFFSET ${param_count}"
        params.append(offset)
        
        return base_query, params
    
    @staticmethod
    def build_sentiment_trend_query(
        company: str,
        timeframe: str = "30d",
        topic: Optional[str] = None
    ) -> Tuple[str, List[Any]]:
        """Build optimized sentiment trend analysis query"""
        
        # Calculate date range based on timeframe
        if timeframe == "7d":
            days = 7
        elif timeframe == "30d":
            days = 30
        elif timeframe == "90d":
            days = 90
        elif timeframe == "1y":
            days = 365
        else:
            days = 30
        
        start_date = datetime.now() - timedelta(days=days)
        
        query = """
        WITH daily_sentiment AS (
            SELECT 
                DATE(sa.created_at) as sentiment_date,
                AVG(sa.overall_sentiment) as avg_sentiment,
                AVG(sa.confidence) as avg_confidence,
                COUNT(*) as document_count
            FROM sentiment_analysis sa
            JOIN documents d ON sa.document_id = d.id
            WHERE d.company = $1 
                AND sa.created_at >= $2
        """
        
        params = [company, start_date]
        param_count = 2
        
        if topic:
            param_count += 1
            query += f" AND sa.topic_sentiments ? ${param_count}"
            params.append(topic)
        
        query += """
            GROUP BY DATE(sa.created_at)
            ORDER BY sentiment_date DESC
        )
        SELECT 
            sentiment_date,
            avg_sentiment,
            avg_confidence,
            document_count,
            LAG(avg_sentiment) OVER (ORDER BY sentiment_date) as previous_sentiment
        FROM daily_sentiment
        ORDER BY sentiment_date DESC
        """
        
        return query, params
    
    @staticmethod
    def build_anomaly_correlation_query(
        company: str,
        time_window_hours: int = 24
    ) -> Tuple[str, List[Any]]:
        """Build optimized anomaly correlation analysis query"""
        
        query = """
        WITH recent_anomalies AS (
            SELECT 
                id, metric_name, current_value, expected_value, 
                deviation_score, severity, created_at
            FROM anomalies
            WHERE company = $1 
                AND created_at >= NOW() - INTERVAL '%s hours'
                AND status IN ('detected', 'investigating')
        ),
        anomaly_pairs AS (
            SELECT 
                a1.id as anomaly1_id,
                a2.id as anomaly2_id,
                a1.metric_name as metric1,
                a2.metric_name as metric2,
                a1.deviation_score as score1,
                a2.deviation_score as score2,
                ABS(EXTRACT(EPOCH FROM (a1.created_at - a2.created_at))/3600) as time_diff_hours
            FROM recent_anomalies a1
            CROSS JOIN recent_anomalies a2
            WHERE a1.id != a2.id
                AND ABS(EXTRACT(EPOCH FROM (a1.created_at - a2.created_at))/3600) <= $2
        )
        SELECT 
            metric1, metric2,
            COUNT(*) as co_occurrence_count,
            AVG(time_diff_hours) as avg_time_diff,
            CORR(score1, score2) as correlation_coefficient
        FROM anomaly_pairs
        GROUP BY metric1, metric2
        HAVING COUNT(*) >= 2
        ORDER BY correlation_coefficient DESC NULLS LAST
        """ % time_window_hours
        
        return query, [company, time_window_hours]
    
    @staticmethod
    def build_forecast_performance_query(
        model_name: Optional[str] = None,
        horizon_days: Optional[int] = None,
        limit: int = 100
    ) -> Tuple[str, List[Any]]:
        """Build optimized forecast performance analysis query"""
        
        query = """
        SELECT 
            f.model_used,
            f.horizon_days,
            f.forecast_type,
            COUNT(*) as total_forecasts,
            AVG(f.accuracy_score) as avg_accuracy,
            STDDEV(f.accuracy_score) as accuracy_stddev,
            AVG(fp.mae) as avg_mae,
            AVG(fp.rmse) as avg_rmse,
            AVG(fp.mape) as avg_mape,
            AVG(fp.directional_accuracy) as avg_directional_accuracy
        FROM forecasts f
        LEFT JOIN forecast_performance fp ON f.id = fp.forecast_id
        WHERE f.actual_value IS NOT NULL
        """
        
        conditions = []
        params = []
        param_count = 0
        
        if model_name:
            param_count += 1
            conditions.append(f"f.model_used = ${param_count}")
            params.append(model_name)
        
        if horizon_days:
            param_count += 1
            conditions.append(f"f.horizon_days = ${param_count}")
            params.append(horizon_days)
        
        if conditions:
            query += " AND " + " AND ".join(conditions)
        
        query += """
        GROUP BY f.model_used, f.horizon_days, f.forecast_type
        ORDER BY avg_accuracy DESC
        """
        
        param_count += 1
        query += f" LIMIT ${param_count}"
        params.append(limit)
        
        return query, params


class DatabaseOptimizationService:
    """Service for database optimization and performance monitoring"""
    
    def __init__(self):
        self.query_optimizer = QueryOptimizer()
        self._performance_stats = {}
        self._slow_query_threshold = 1.0  # seconds
    
    async def create_advanced_indexes(self):
        """Create advanced indexes for optimal query performance"""
        
        indexes = [
            # Composite indexes for common query patterns
            {
                "name": "idx_documents_company_date_type",
                "table": "documents",
                "definition": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_company_date_type ON documents(company, filing_date DESC, document_type)"
            },
            {
                "name": "idx_sentiment_document_created",
                "table": "sentiment_analysis", 
                "definition": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sentiment_document_created ON sentiment_analysis(document_id, created_at DESC)"
            },
            {
                "name": "idx_anomalies_company_status_created",
                "table": "anomalies",
                "definition": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_anomalies_company_status_created ON anomalies(company, status, created_at DESC)"
            },
            {
                "name": "idx_forecasts_ticker_horizon_target",
                "table": "forecasts",
                "definition": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_forecasts_ticker_horizon_target ON forecasts(ticker, horizon_days, target_date DESC)"
            },
            
            # Partial indexes for active/recent data
            {
                "name": "idx_documents_active_processing",
                "table": "documents",
                "definition": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_active_processing ON documents(processing_status, created_at) WHERE processing_status IN ('pending', 'processing')"
            },
            {
                "name": "idx_anomalies_unresolved",
                "table": "anomalies", 
                "definition": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_anomalies_unresolved ON anomalies(company, severity, created_at DESC) WHERE status IN ('detected', 'investigating')"
            },
            {
                "name": "idx_forecasts_recent_accuracy",
                "table": "forecasts",
                "definition": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_forecasts_recent_accuracy ON forecasts(model_used, accuracy_score DESC) WHERE created_at >= NOW() - INTERVAL '30 days'"
            },
            
            # Expression indexes for computed values
            {
                "name": "idx_anomalies_abs_deviation",
                "table": "anomalies",
                "definition": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_anomalies_abs_deviation ON anomalies(company, ABS(current_value - expected_value))"
            },
            {
                "name": "idx_forecasts_confidence_width",
                "table": "forecasts",
                "definition": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_forecasts_confidence_width ON forecasts(ticker, (confidence_upper - confidence_lower))"
            },
            
            # Full-text search indexes
            {
                "name": "idx_documents_content_fts",
                "table": "documents",
                "definition": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_content_fts ON documents USING gin(to_tsvector('english', processed_content))"
            },
            {
                "name": "idx_documents_summary_fts", 
                "table": "documents",
                "definition": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_summary_fts ON documents USING gin(to_tsvector('english', summary))"
            },
            
            # Hash indexes for exact matches on high-cardinality columns
            {
                "name": "idx_documents_id_hash",
                "table": "documents",
                "definition": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_id_hash ON documents USING hash(id)"
            },
            {
                "name": "idx_sentiment_document_id_hash",
                "table": "sentiment_analysis",
                "definition": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sentiment_document_id_hash ON sentiment_analysis USING hash(document_id)"
            }
        ]
        
        created_indexes = []
        failed_indexes = []
        
        for index in indexes:
            try:
                async with db_manager.get_connection() as conn:
                    await conn.execute(index["definition"])
                    created_indexes.append(index["name"])
                    logger.info(f"Created index: {index['name']}")
                    
            except Exception as e:
                failed_indexes.append({"name": index["name"], "error": str(e)})
                logger.error(f"Failed to create index {index['name']}: {e}")
        
        return {
            "created": created_indexes,
            "failed": failed_indexes,
            "total_attempted": len(indexes)
        }
    
    async def analyze_table_statistics(self) -> Dict[str, Any]:
        """Analyze table statistics for query optimization"""
        
        tables = [
            "documents", "sentiment_analysis", "anomalies", 
            "forecasts", "sentiment_trends", "forecast_performance"
        ]
        
        stats = {}
        
        for table in tables:
            try:
                async with db_manager.get_connection() as conn:
                    # Update table statistics
                    await conn.execute(f"ANALYZE {table}")
                    
                    # Get table size and row count
                    table_stats = await conn.fetchrow(f"""
                        SELECT 
                            schemaname,
                            tablename,
                            attname,
                            n_distinct,
                            correlation
                        FROM pg_stats 
                        WHERE tablename = '{table}'
                        LIMIT 1
                    """)
                    
                    # Get table size information
                    size_info = await conn.fetchrow(f"""
                        SELECT 
                            pg_size_pretty(pg_total_relation_size('{table}')) as total_size,
                            pg_size_pretty(pg_relation_size('{table}')) as table_size,
                            pg_size_pretty(pg_total_relation_size('{table}') - pg_relation_size('{table}')) as index_size,
                            (SELECT reltuples::bigint FROM pg_class WHERE relname = '{table}') as estimated_rows
                    """)
                    
                    # Get index usage statistics
                    index_stats = await conn.fetch(f"""
                        SELECT 
                            indexrelname as index_name,
                            idx_tup_read,
                            idx_tup_fetch,
                            idx_scan
                        FROM pg_stat_user_indexes 
                        WHERE relname = '{table}'
                        ORDER BY idx_scan DESC
                    """)
                    
                    stats[table] = {
                        "size_info": dict(size_info) if size_info else {},
                        "index_usage": [dict(idx) for idx in index_stats],
                        "last_analyzed": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                logger.error(f"Failed to analyze table {table}: {e}")
                stats[table] = {"error": str(e)}
        
        return stats
    
    async def optimize_connection_pool(self) -> Dict[str, Any]:
        """Optimize database connection pool settings"""
        
        try:
            async with db_manager.get_connection() as conn:
                # Get current connection statistics
                conn_stats = await conn.fetchrow("""
                    SELECT 
                        count(*) as total_connections,
                        count(*) FILTER (WHERE state = 'active') as active_connections,
                        count(*) FILTER (WHERE state = 'idle') as idle_connections,
                        count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                """)
                
                # Get connection pool configuration
                pool_config = await conn.fetchrow("""
                    SELECT 
                        setting as max_connections
                    FROM pg_settings 
                    WHERE name = 'max_connections'
                """)
                
                # Calculate optimal pool size based on current usage
                active_connections = conn_stats['active_connections']
                total_connections = conn_stats['total_connections']
                max_connections = int(pool_config['max_connections'])
                
                # Recommend pool size (typically 2-4x CPU cores, but not more than 25% of max_connections)
                import os
                cpu_count = os.cpu_count() or 4
                recommended_pool_size = min(cpu_count * 3, max_connections // 4)
                
                current_pool_size = db_manager._pool.get_size() if db_manager._pool else 0
                current_max_size = db_manager._pool.get_max_size() if db_manager._pool else 0
                
                optimization_recommendations = []
                
                if current_max_size < recommended_pool_size:
                    optimization_recommendations.append(
                        f"Consider increasing max pool size from {current_max_size} to {recommended_pool_size}"
                    )
                
                if active_connections > current_max_size * 0.8:
                    optimization_recommendations.append(
                        "High connection utilization detected - consider increasing pool size"
                    )
                
                if conn_stats['idle_in_transaction'] > 0:
                    optimization_recommendations.append(
                        "Idle in transaction connections detected - review transaction handling"
                    )
                
                return {
                    "current_stats": dict(conn_stats),
                    "pool_config": {
                        "current_size": current_pool_size,
                        "max_size": current_max_size,
                        "recommended_size": recommended_pool_size
                    },
                    "database_config": dict(pool_config),
                    "recommendations": optimization_recommendations,
                    "cpu_count": cpu_count
                }
                
        except Exception as e:
            logger.error(f"Failed to optimize connection pool: {e}")
            return {"error": str(e)}
    
    async def identify_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Identify slow queries for optimization"""
        
        try:
            async with db_manager.get_connection() as conn:
                # Enable pg_stat_statements if available
                try:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
                except:
                    pass  # Extension might not be available
                
                # Get slow queries from pg_stat_statements
                slow_queries = await conn.fetch(f"""
                    SELECT 
                        query,
                        calls,
                        total_time,
                        mean_time,
                        max_time,
                        stddev_time,
                        rows,
                        100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                    FROM pg_stat_statements
                    WHERE mean_time > {self._slow_query_threshold * 1000}  -- Convert to milliseconds
                    ORDER BY mean_time DESC
                    LIMIT {limit}
                """)
                
                return [dict(query) for query in slow_queries]
                
        except Exception as e:
            logger.error(f"Failed to identify slow queries: {e}")
            return []
    
    async def vacuum_and_reindex(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Perform vacuum and reindex operations"""
        
        tables_to_process = []
        
        if table_name:
            tables_to_process = [table_name]
        else:
            # Get all user tables
            async with db_manager.get_connection() as conn:
                tables = await conn.fetch("""
                    SELECT tablename 
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                """)
                tables_to_process = [table['tablename'] for table in tables]
        
        results = {}
        
        for table in tables_to_process:
            try:
                async with db_manager.get_connection() as conn:
                    # Vacuum analyze
                    await conn.execute(f"VACUUM ANALYZE {table}")
                    
                    # Get table bloat information
                    bloat_info = await conn.fetchrow(f"""
                        SELECT 
                            schemaname,
                            tablename,
                            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                        FROM pg_tables 
                        WHERE tablename = '{table}'
                    """)
                    
                    results[table] = {
                        "vacuum_completed": True,
                        "size": bloat_info['size'] if bloat_info else "unknown",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    logger.info(f"Vacuumed and analyzed table: {table}")
                    
            except Exception as e:
                logger.error(f"Failed to vacuum table {table}: {e}")
                results[table] = {
                    "vacuum_completed": False,
                    "error": str(e)
                }
        
        return results
    
    async def get_database_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive database performance metrics"""
        
        try:
            async with db_manager.get_connection() as conn:
                # Database size and activity
                db_stats = await conn.fetchrow("""
                    SELECT 
                        pg_database_size(current_database()) as db_size_bytes,
                        pg_size_pretty(pg_database_size(current_database())) as db_size_pretty,
                        (SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()) as active_connections,
                        (SELECT setting FROM pg_settings WHERE name = 'shared_buffers') as shared_buffers,
                        (SELECT setting FROM pg_settings WHERE name = 'effective_cache_size') as effective_cache_size
                """)
                
                # Cache hit ratios
                cache_stats = await conn.fetchrow("""
                    SELECT 
                        round(100.0 * sum(blks_hit) / (sum(blks_hit) + sum(blks_read)), 2) as buffer_cache_hit_ratio,
                        round(100.0 * sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)), 2) as table_cache_hit_ratio
                    FROM pg_statio_user_tables
                """)
                
                # Index usage statistics
                index_usage = await conn.fetchrow("""
                    SELECT 
                        round(100.0 * sum(idx_tup_fetch) / (sum(seq_tup_read) + sum(idx_tup_fetch)), 2) as index_usage_ratio
                    FROM pg_stat_user_tables
                    WHERE seq_tup_read + idx_tup_fetch > 0
                """)
                
                # Transaction statistics
                transaction_stats = await conn.fetchrow("""
                    SELECT 
                        xact_commit,
                        xact_rollback,
                        round(100.0 * xact_commit / (xact_commit + xact_rollback), 2) as commit_ratio
                    FROM pg_stat_database 
                    WHERE datname = current_database()
                """)
                
                # Lock statistics
                lock_stats = await conn.fetch("""
                    SELECT 
                        mode,
                        count(*) as lock_count
                    FROM pg_locks 
                    WHERE database = (SELECT oid FROM pg_database WHERE datname = current_database())
                    GROUP BY mode
                    ORDER BY lock_count DESC
                """)
                
                return {
                    "database_stats": dict(db_stats),
                    "cache_stats": dict(cache_stats) if cache_stats else {},
                    "index_usage": dict(index_usage) if index_usage else {},
                    "transaction_stats": dict(transaction_stats) if transaction_stats else {},
                    "lock_stats": [dict(lock) for lock in lock_stats],
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get database performance metrics: {e}")
            return {"error": str(e)}
    
    @asynccontextmanager
    async def query_performance_monitor(self, query_name: str):
        """Context manager to monitor query performance"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            yield
        finally:
            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time
            
            # Store performance stats
            if query_name not in self._performance_stats:
                self._performance_stats[query_name] = []
            
            self._performance_stats[query_name].append({
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last 100 measurements
            if len(self._performance_stats[query_name]) > 100:
                self._performance_stats[query_name] = self._performance_stats[query_name][-100:]
            
            # Log slow queries
            if execution_time > self._slow_query_threshold:
                logger.warning(f"Slow query detected: {query_name} took {execution_time:.3f}s")
    
    async def get_query_performance_stats(self) -> Dict[str, Any]:
        """Get query performance statistics"""
        
        stats = {}
        
        for query_name, measurements in self._performance_stats.items():
            if measurements:
                execution_times = [m["execution_time"] for m in measurements]
                stats[query_name] = {
                    "count": len(measurements),
                    "avg_time": sum(execution_times) / len(execution_times),
                    "min_time": min(execution_times),
                    "max_time": max(execution_times),
                    "slow_queries": len([t for t in execution_times if t > self._slow_query_threshold])
                }
        
        return stats


# Global database optimization service instance
db_optimization_service = DatabaseOptimizationService()


async def get_db_optimization_service() -> DatabaseOptimizationService:
    """Get database optimization service instance"""
    return db_optimization_service