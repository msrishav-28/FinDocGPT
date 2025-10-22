"""
Database initialization utilities
"""

import os
import asyncio
import logging
from typing import Optional

from .connection import DatabaseManager, init_database
from .migrations import MigrationManager
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


async def initialize_database_system(
    connection_string: Optional[str] = None,
    run_migrations: bool = True,
    initialize_vector_store: bool = True
) -> tuple[DatabaseManager, MigrationManager, VectorStore]:
    """
    Initialize the complete database system
    
    Args:
        connection_string: Database connection string (optional, uses env vars if not provided)
        run_migrations: Whether to run pending migrations
        initialize_vector_store: Whether to initialize vector store
    
    Returns:
        Tuple of (DatabaseManager, MigrationManager, VectorStore)
    """
    
    logger.info("Initializing database system...")
    
    # Initialize database connection
    if connection_string:
        await init_database(connection_string=connection_string)
    else:
        await init_database()
    
    # Get database manager instance
    from .connection import db_manager
    
    # Initialize migration manager
    migration_manager = MigrationManager(db_manager)
    await migration_manager.initialize()
    
    # Run migrations if requested
    if run_migrations:
        logger.info("Running database migrations...")
        await migration_manager.migrate_up()
        logger.info("Database migrations completed")
    
    # Initialize vector store
    vector_store = VectorStore(db_manager)
    if initialize_vector_store:
        await vector_store.initialize()
    
    logger.info("Database system initialization completed")
    
    return db_manager, migration_manager, vector_store


async def create_sample_data():
    """Create sample data for development and testing"""
    from .connection import db_manager
    
    logger.info("Creating sample data...")
    
    try:
        # Insert sample external data sources
        sample_sources = [
            {
                'source_name': 'Yahoo Finance',
                'api_endpoint': 'https://query1.finance.yahoo.com/v8/finance/chart/',
                'api_key_required': False,
                'rate_limit': 2000,
                'data_types': ['stock_prices', 'market_data'],
                'reliability_score': 0.95
            },
            {
                'source_name': 'Alpha Vantage',
                'api_endpoint': 'https://www.alphavantage.co/query',
                'api_key_required': True,
                'rate_limit': 5,
                'data_types': ['stock_prices', 'fundamentals', 'technical_indicators'],
                'reliability_score': 0.90
            },
            {
                'source_name': 'Quandl',
                'api_endpoint': 'https://www.quandl.com/api/v3/',
                'api_key_required': True,
                'rate_limit': 50,
                'data_types': ['economic_data', 'financial_data'],
                'reliability_score': 0.88
            }
        ]
        
        for source in sample_sources:
            await db_manager.execute(
                """
                INSERT INTO external_data_sources 
                (source_name, api_endpoint, api_key_required, rate_limit, data_types, reliability_score)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (source_name) DO NOTHING
                """,
                source['source_name'],
                source['api_endpoint'],
                source['api_key_required'],
                source['rate_limit'],
                source['data_types'],
                source['reliability_score']
            )
        
        logger.info("Sample data created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create sample data: {e}")
        raise


async def check_database_health() -> dict:
    """Check database system health"""
    from .connection import db_manager
    
    try:
        # Get basic health check
        health = await db_manager.health_check()
        
        # Add migration status
        migration_manager = MigrationManager(db_manager)
        await migration_manager.initialize()
        migration_status = await migration_manager.get_migration_status()
        
        # Add vector store stats
        vector_store = VectorStore(db_manager)
        embedding_stats = await vector_store.get_embedding_stats()
        
        return {
            **health,
            'migrations': migration_status,
            'embeddings': embedding_stats
        }
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


async def reset_database():
    """Reset database (WARNING: This will delete all data!)"""
    from .connection import db_manager
    
    logger.warning("RESETTING DATABASE - ALL DATA WILL BE LOST!")
    
    try:
        # Get all table names
        tables_query = """
        SELECT tablename FROM pg_tables 
        WHERE schemaname = 'public' 
        AND tablename != 'schema_migrations'
        """
        
        tables = await db_manager.fetch(tables_query)
        
        # Drop all tables except schema_migrations
        for table in tables:
            table_name = table['tablename']
            await db_manager.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
            logger.info(f"Dropped table: {table_name}")
        
        # Reset migrations
        await db_manager.execute("DELETE FROM schema_migrations")
        
        logger.warning("Database reset completed")
        
    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        raise


if __name__ == "__main__":
    # CLI interface for database operations
    import sys
    
    async def main():
        if len(sys.argv) < 2:
            print("Usage: python -m app.database.init <command>")
            print("Commands: init, migrate, reset, health, sample-data")
            return
        
        command = sys.argv[1]
        
        if command == "init":
            await initialize_database_system()
            print("Database system initialized")
            
        elif command == "migrate":
            db_manager, migration_manager, _ = await initialize_database_system(run_migrations=False)
            await migration_manager.migrate_up()
            print("Migrations completed")
            
        elif command == "reset":
            await initialize_database_system(run_migrations=False)
            await reset_database()
            print("Database reset completed")
            
        elif command == "health":
            await initialize_database_system(run_migrations=False)
            health = await check_database_health()
            print(f"Database health: {health}")
            
        elif command == "sample-data":
            await initialize_database_system()
            await create_sample_data()
            print("Sample data created")
            
        else:
            print(f"Unknown command: {command}")
    
    asyncio.run(main())