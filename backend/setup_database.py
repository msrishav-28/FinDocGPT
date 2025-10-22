#!/usr/bin/env python3
"""
Database setup script for development
"""

import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.database.init import initialize_database_system, create_sample_data, check_database_health


async def setup_development_database():
    """Set up database for development"""
    print("Setting up development database...")
    
    try:
        # Initialize database system
        print("1. Initializing database system...")
        db_manager, migration_manager, vector_store = await initialize_database_system(
            run_migrations=True,
            initialize_vector_store=True
        )
        print("   ✓ Database system initialized")
        
        # Create sample data
        print("2. Creating sample data...")
        await create_sample_data()
        print("   ✓ Sample data created")
        
        # Check health
        print("3. Checking database health...")
        health = await check_database_health()
        print(f"   ✓ Database status: {health.get('status', 'unknown')}")
        
        # Print summary
        print("\n" + "="*50)
        print("DATABASE SETUP COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Database: {health.get('database', {}).get('database', {}).get('db_size', 'N/A')} bytes")
        print(f"Migrations: {health.get('migrations', {}).get('applied_count', 0)} applied")
        print(f"Tables created: documents, sentiment_analysis, anomalies, forecasts, etc.")
        print("\nYou can now start the FastAPI server with:")
        print("  cd backend && python -m uvicorn app.main:app --reload")
        
    except Exception as e:
        print(f"\n❌ Database setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(setup_development_database())