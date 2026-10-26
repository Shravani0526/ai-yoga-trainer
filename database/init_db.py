"""Create (or verify) the database schema.

    python -m database.init_db                       # uses DATABASE_URL or data/yoga_trainer.db
    DATABASE_URL=postgresql://user:pw@host/db python -m database.init_db
"""

from __future__ import annotations

import sys

from database.db import Database


def main() -> int:
    db = Database()
    print(f"Database ready: {db.describe()}")
    print(f"Users stored: {len(db.list_users())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
