"""Persistence layer (SQLite by default, optional PostgreSQL)."""

from database.db import Database, get_database

__all__ = ["Database", "get_database"]
