"""Clear all stale pending_uploads entries from the database."""

import sqlite3
import os

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'data', 'processed', 'channel_forge.db'
)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM pending_uploads")
count = cursor.fetchone()[0]
print(f"Pending uploads before clear: {count}")
cursor.execute("DELETE FROM pending_uploads")
conn.commit()
cursor.execute("SELECT COUNT(*) FROM pending_uploads")
count = cursor.fetchone()[0]
print(f"Pending uploads after clear: {count}")
conn.close()
print("Done.")
