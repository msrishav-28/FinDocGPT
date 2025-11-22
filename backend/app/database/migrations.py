"""
Database migration management system
"""

import os
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import logging
from pathlib import Path

from .connection import DatabaseManager

logger = logging.getLogger(__name__)


class Migration:
    """Single database migration"""
    
    def __init__(self, version: str, name: str, up_sql: str, down_sql: str = None):
        self.version = version
        self.name = name
        self.up_sql = up_sql
        self.down_sql = down_sql
        self.applied_at: Optional[datetime] = None
    
    def __str__(self):
        return f"Migration {self.version}: {self.name}"


class MigrationManager:
    """Database migration manager"""
    
    def __init__(self, db_manager: DatabaseManager, migrations_dir: str = None):
        self.db_manager = db_manager
        self.migrations_dir = migrations_dir or os.path.join(os.path.dirname(__file__), "migrations")
        self.migrations: List[Migration] = []
    
    async def initialize(self):
        """Initialize migration system"""
        await self._create_migrations_table()
        await self._load_migrations()
    
    async def _create_migrations_table(self):
        """Create migrations tracking table"""
        query = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR(50) PRIMARY KEY,
            name VARCHAR(200) NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            checksum VARCHAR(64)
        );
        """
        await self.db_manager.execute(query)
    
    async def _load_migrations(self):
        """Load migration files from directory"""
        migrations_path = Path(self.migrations_dir)
        if not migrations_path.exists():
            migrations_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created migrations directory: {migrations_path}")
        
        # Load migration files
        migration_files = sorted(migrations_path.glob("*.sql"))
        
        for file_path in migration_files:
            try:
                migration = self._parse_migration_file(file_path)
                self.migrations.append(migration)
            except Exception as e:
                logger.error(f"Failed to parse migration file {file_path}: {e}")
                raise
        
        logger.info(f"Loaded {len(self.migrations)} migrations")
    
    def _parse_migration_file(self, file_path: Path) -> Migration:
        """Parse migration file"""
        filename = file_path.stem
        
        # Extract version and name from filename (e.g., "001_create_documents_table")
        parts = filename.split("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid migration filename format: {filename}")
        
        version, name = parts
        
        # Read file content
        content = file_path.read_text(encoding='utf-8')
        
        # Split up and down migrations if present
        if "-- DOWN" in content:
            up_sql, down_sql = content.split("-- DOWN", 1)
            down_sql = down_sql.strip()
        else:
            up_sql = content
            down_sql = None
        
        up_sql = up_sql.replace("-- UP", "").strip()
        
        return Migration(version, name.replace("_", " ").title(), up_sql, down_sql)
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions"""
        query = "SELECT version FROM schema_migrations ORDER BY version"
        rows = await self.db_manager.fetch(query)
        return [row['version'] for row in rows]
    
    async def get_pending_migrations(self) -> List[Migration]:
        """Get list of pending migrations"""
        applied = await self.get_applied_migrations()
        return [m for m in self.migrations if m.version not in applied]
    
    async def apply_migration(self, migration: Migration):
        """Apply a single migration"""
        logger.info(f"Applying migration: {migration}")
        
        async with self.db_manager.get_transaction() as conn:
            try:
                # Execute migration SQL
                await conn.execute(migration.up_sql)
                
                # Record migration as applied
                await conn.execute(
                    "INSERT INTO schema_migrations (version, name) VALUES ($1, $2)",
                    migration.version, migration.name
                )
                
                migration.applied_at = datetime.utcnow()
                logger.info(f"Successfully applied migration: {migration}")
                
            except Exception as e:
                logger.error(f"Failed to apply migration {migration}: {e}")
                raise
    
    async def rollback_migration(self, migration: Migration):
        """Rollback a single migration"""
        if not migration.down_sql:
            raise ValueError(f"Migration {migration.version} has no rollback SQL")
        
        logger.info(f"Rolling back migration: {migration}")
        
        async with self.db_manager.get_transaction() as conn:
            try:
                # Execute rollback SQL
                await conn.execute(migration.down_sql)
                
                # Remove migration record
                await conn.execute(
                    "DELETE FROM schema_migrations WHERE version = $1",
                    migration.version
                )
                
                logger.info(f"Successfully rolled back migration: {migration}")
                
            except Exception as e:
                logger.error(f"Failed to rollback migration {migration}: {e}")
                raise
    
    async def migrate_up(self, target_version: str = None):
        """Apply all pending migrations up to target version"""
        pending = await self.get_pending_migrations()
        
        if target_version:
            pending = [m for m in pending if m.version <= target_version]
        
        if not pending:
            logger.info("No pending migrations to apply")
            return
        
        logger.info(f"Applying {len(pending)} migrations")
        
        for migration in pending:
            await self.apply_migration(migration)
        
        logger.info("All migrations applied successfully")
    
    async def migrate_down(self, target_version: str):
        """Rollback migrations down to target version"""
        applied = await self.get_applied_migrations()
        
        # Find migrations to rollback (in reverse order)
        to_rollback = []
        for version in reversed(applied):
            if version > target_version:
                migration = next((m for m in self.migrations if m.version == version), None)
                if migration:
                    to_rollback.append(migration)
        
        if not to_rollback:
            logger.info("No migrations to rollback")
            return
        
        logger.info(f"Rolling back {len(to_rollback)} migrations")
        
        for migration in to_rollback:
            await self.rollback_migration(migration)
        
        logger.info("Rollback completed successfully")
    
    async def get_migration_status(self) -> Dict:
        """Get current migration status"""
        applied = await self.get_applied_migrations()
        pending = await self.get_pending_migrations()
        
        return {
            "total_migrations": len(self.migrations),
            "applied_count": len(applied),
            "pending_count": len(pending),
            "applied_versions": applied,
            "pending_versions": [m.version for m in pending],
            "current_version": applied[-1] if applied else None
        }
    
    def create_migration_file(self, name: str, up_sql: str, down_sql: str = None) -> str:
        """Create a new migration file"""
        # Generate version number
        existing_versions = [int(m.version) for m in self.migrations if m.version.isdigit()]
        next_version = str(max(existing_versions, default=0) + 1).zfill(3)
        
        # Create filename
        filename = f"{next_version}_{name.lower().replace(' ', '_')}.sql"
        file_path = Path(self.migrations_dir) / filename
        
        # Create file content
        content = f"-- UP\n{up_sql}\n"
        if down_sql:
            content += f"\n-- DOWN\n{down_sql}\n"
        
        # Write file
        file_path.write_text(content, encoding='utf-8')
        
        logger.info(f"Created migration file: {file_path}")
        return str(file_path)