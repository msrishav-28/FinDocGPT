"""
Database connection management and pooling utilities
"""

import os
import asyncio
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
import asyncpg
from asyncpg import Pool, Connection
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection manager with connection pooling"""
    
    def __init__(self):
        self._pool: Optional[Pool] = None
        self._connection_string: Optional[str] = None
        
    async def initialize(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        username: str = None,
        password: str = None,
        connection_string: str = None,
        min_connections: int = 5,
        max_connections: int = 20,
        **kwargs
    ):
        """Initialize database connection pool"""
        
        if connection_string:
            self._connection_string = connection_string
        else:
            # Build connection string from parameters or environment variables
            host = host or os.getenv("DB_HOST", "localhost")
            port = port or int(os.getenv("DB_PORT", "5432"))
            database = database or os.getenv("DB_NAME", "financial_intelligence")
            username = username or os.getenv("DB_USER", "postgres")
            password = password or os.getenv("DB_PASSWORD", "postgres")
            
            self._connection_string = (
                f"postgresql://{username}:{password}@{host}:{port}/{database}"
            )
        
        try:
            # Create connection pool
            self._pool = await asyncpg.create_pool(
                self._connection_string,
                min_size=min_connections,
                max_size=max_connections,
                command_timeout=60,
                server_settings={
                    'jit': 'off',  # Disable JIT for better performance with short queries
                    'application_name': 'financial_intelligence_system'
                },
                **kwargs
            )
            
            logger.info(f"Database pool initialized with {min_connections}-{max_connections} connections")
            
            # Test connection
            async with self._pool.acquire() as conn:
                await conn.execute("SELECT 1")
                logger.info("Database connection test successful")
                
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self._pool:
            await self._pool.close()
            logger.info("Database pool closed")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[Connection, None]:
        """Get database connection from pool"""
        if not self._pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self._pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                logger.error(f"Database operation failed: {e}")
                raise
    
    @asynccontextmanager
    async def get_transaction(self) -> AsyncGenerator[Connection, None]:
        """Get database connection with transaction"""
        async with self.get_connection() as conn:
            async with conn.transaction():
                yield conn
    
    async def execute(self, query: str, *args) -> str:
        """Execute a query and return status"""
        async with self.get_connection() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args) -> list:
        """Fetch multiple rows"""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args) -> Optional[dict]:
        """Fetch single row"""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None
    
    async def fetchval(self, query: str, *args):
        """Fetch single value"""
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args)
    
    async def executemany(self, query: str, args_list: list) -> None:
        """Execute query multiple times with different parameters"""
        async with self.get_connection() as conn:
            await conn.executemany(query, args_list)
    
    async def copy_records_to_table(self, table_name: str, records: list, columns: list = None):
        """Bulk insert records using COPY"""
        async with self.get_connection() as conn:
            await conn.copy_records_to_table(table_name, records=records, columns=columns)
    
    async def health_check(self) -> dict:
        """Check database health"""
        try:
            async with self.get_connection() as conn:
                # Check basic connectivity
                await conn.execute("SELECT 1")
                
                # Get pool stats
                pool_stats = {
                    "size": self._pool.get_size(),
                    "min_size": self._pool.get_min_size(),
                    "max_size": self._pool.get_max_size(),
                    "idle_connections": self._pool.get_idle_size(),
                }
                
                # Get database stats
                db_stats = await conn.fetchrow("""
                    SELECT 
                        pg_database_size(current_database()) as db_size,
                        (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                        version() as version
                """)
                
                return {
                    "status": "healthy",
                    "pool": pool_stats,
                    "database": dict(db_stats),
                    "timestamp": asyncio.get_event_loop().time()
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
    
    async def execute_with_monitoring(self, query: str, *args, query_name: str = "unknown") -> str:
        """Execute query with performance monitoring"""
        from ..services.database_optimization_service import db_optimization_service
        
        async with db_optimization_service.query_performance_monitor(query_name):
            return await self.execute(query, *args)
    
    async def fetch_with_monitoring(self, query: str, *args, query_name: str = "unknown") -> list:
        """Fetch with performance monitoring"""
        from ..services.database_optimization_service import db_optimization_service
        
        async with db_optimization_service.query_performance_monitor(query_name):
            return await self.fetch(query, *args)


# Global database manager instance
db_manager = DatabaseManager()


async def get_database() -> DatabaseManager:
    """Get database manager instance"""
    return db_manager


async def init_database(**kwargs):
    """Initialize database connection"""
    await db_manager.initialize(**kwargs)


async def close_database():
    """Close database connection"""
    await db_manager.close()


# Context manager for database operations
@asynccontextmanager
async def database_transaction():
    """Context manager for database transactions"""
    async with db_manager.get_transaction() as conn:
        yield conn