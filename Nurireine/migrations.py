"""
Database Migration System

Manages schema versioning and migrations to prevent data loss.
Migrations are applied automatically on startup if needed.
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """Represents a single database migration."""
    version: int
    description: str
    up_sql: str  # SQL to apply migration
    down_sql: Optional[str] = None  # SQL to rollback (optional)


class MigrationRunner:
    """
    Handles database schema migrations.
    
    Migrations are tracked in the schema_version table.
    Each migration has a version number, description, and SQL statements.
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize migration runner.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.migrations: List[Migration] = []
        self._init_schema_version_table()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(str(self.db_path))
    
    def _init_schema_version_table(self) -> None:
        """Initialize the schema_version table if it doesn't exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY,
                        description TEXT NOT NULL,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                logger.info("event=schema_version_table_initialized")
        except Exception as e:
            logger.error(f"event=schema_version_init_failed error={str(e)}")
            raise
    
    def get_current_version(self) -> int:
        """
        Get current schema version.
        
        Returns:
            Current version number (0 if no migrations applied)
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(version) FROM schema_version")
                result = cursor.fetchone()
                version = result[0] if result[0] is not None else 0
                logger.debug(f"event=current_version_retrieved version={version}")
                return version
        except Exception as e:
            logger.error(f"event=get_version_failed error={str(e)}")
            return 0
    
    def register_migration(
        self, 
        version: int, 
        description: str, 
        up_sql: str,
        down_sql: Optional[str] = None
    ) -> None:
        """
        Register a migration.
        
        Args:
            version: Migration version number (must be sequential)
            description: Human-readable description
            up_sql: SQL to apply migration
            down_sql: SQL to rollback (optional)
        """
        migration = Migration(
            version=version,
            description=description,
            up_sql=up_sql,
            down_sql=down_sql
        )
        self.migrations.append(migration)
        logger.debug(
            f"event=migration_registered "
            f"version={version} "
            f"description={description}"
        )
    
    def apply_migration(self, migration: Migration) -> bool:
        """
        Apply a single migration.
        
        Args:
            migration: Migration to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Execute migration SQL
                cursor.executescript(migration.up_sql)
                
                # Record migration in schema_version
                cursor.execute(
                    "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                    (migration.version, migration.description)
                )
                
                conn.commit()
                
                logger.info(
                    f"event=migration_applied "
                    f"version={migration.version} "
                    f"description={migration.description}"
                )
                return True
                
        except Exception as e:
            logger.error(
                f"event=migration_failed "
                f"version={migration.version} "
                f"error={str(e)}"
            )
            return False
    
    def run_migrations(self) -> Tuple[int, int]:
        """
        Run all pending migrations.
        
        Returns:
            Tuple of (migrations_applied, total_migrations)
        """
        current_version = self.get_current_version()
        pending_migrations = [
            m for m in sorted(self.migrations, key=lambda x: x.version)
            if m.version > current_version
        ]
        
        if not pending_migrations:
            logger.info(
                f"event=no_pending_migrations "
                f"current_version={current_version}"
            )
            return 0, len(self.migrations)
        
        logger.info(
            f"event=migrations_starting "
            f"current_version={current_version} "
            f"pending_count={len(pending_migrations)}"
        )
        
        applied = 0
        for migration in pending_migrations:
            if self.apply_migration(migration):
                applied += 1
            else:
                logger.error(
                    f"event=migration_sequence_failed "
                    f"stopped_at_version={migration.version}"
                )
                break
        
        new_version = self.get_current_version()
        logger.info(
            f"event=migrations_completed "
            f"applied={applied} "
            f"new_version={new_version}"
        )
        
        return applied, len(self.migrations)
    
    def get_migration_history(self) -> List[Tuple[int, str, str]]:
        """
        Get migration history.
        
        Returns:
            List of (version, description, applied_at) tuples
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT version, description, applied_at 
                    FROM schema_version 
                    ORDER BY version
                """)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"event=get_history_failed error={str(e)}")
            return []


# =============================================================================
# Migration Definitions
# =============================================================================

def get_migrations() -> List[Migration]:
    """
    Define all database migrations.
    
    Returns:
        List of Migration objects
    """
    migrations = []
    
    # Migration 001: Initial schema (baseline)
    migrations.append(Migration(
        version=1,
        description="Initial schema with active_channels, channel_summaries, chat_logs",
        up_sql="""
            -- Baseline migration: Tables created by initial DatabaseManager._init_db()
            -- This migration records the existing schema state for version tracking
            -- No schema changes needed as tables already exist
            SELECT 1 WHERE 1=0;  -- No-op: tables created by DatabaseManager
        """
    ))
    
    # Migration 002: Add channel_policies table for response modes
    migrations.append(Migration(
        version=2,
        description="Add channel_policies table for per-channel response modes",
        up_sql="""
            CREATE TABLE IF NOT EXISTS channel_policies (
                channel_id INTEGER PRIMARY KEY,
                response_mode TEXT DEFAULT 'balanced',
                mood_adjustment BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CHECK(response_mode IN ('strict', 'balanced', 'chatty'))
            );
        """,
        down_sql="DROP TABLE IF EXISTS channel_policies;"
    ))
    
    # Migration 003: L3 memory metadata enhancement (programmatic)
    migrations.append(Migration(
        version=3,
        description="L3 memory lifecycle fields (timestamp, score, ttl_days) - enhanced programmatically in memory module",
        up_sql="""
            -- Note: L3 memory is stored in ChromaDB (not SQLite)
            -- Metadata fields (timestamp, score, ttl_days, access_count) are added
            -- programmatically when saving facts in memory.save_facts()
            -- This migration exists for documentation and version tracking only
            SELECT 1 WHERE 1=0;  -- No-op: ChromaDB metadata managed in memory.py
        """
    ))
    
    return migrations


def run_auto_migration(db_path: Path) -> bool:
    """
    Run automatic migrations on startup.
    
    Args:
        db_path: Path to database file
        
    Returns:
        True if successful or no migrations needed
    """
    try:
        runner = MigrationRunner(db_path)
        
        # Register all migrations
        for migration in get_migrations():
            runner.register_migration(
                migration.version,
                migration.description,
                migration.up_sql,
                migration.down_sql
            )
        
        # Run pending migrations
        applied, total = runner.run_migrations()
        
        if applied > 0:
            logger.info(
                f"event=auto_migration_completed "
                f"applied={applied} total={total}"
            )
        
        return True
        
    except Exception as e:
        logger.error(f"event=auto_migration_failed error={str(e)}")
        return False
