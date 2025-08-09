import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "history.db"

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            user_id TEXT NOT NULL,
            session_id TEXT,
            payload TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def add_event(event_type, user_id, session_id=None, payload=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    payload_json = json.dumps(payload) if payload is not None else None
    c.execute("""
        INSERT INTO history (event_type, user_id, session_id, payload, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (event_type, user_id, session_id, payload_json, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def get_history(limit=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    history_list = []
    for row in rows:
        history_list.append({
            "id": row[0],
            "event_type": row[1],
            "user_id": row[2],
            "session_id": row[3],
            "payload": json.loads(row[4]) if row[4] else None,
            "timestamp": row[5]
        })
    return history_list

def clear_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM history")
    conn.commit()
    conn.close()
def search_history(event_type=None, user_id=None, keyword=None, limit=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    query = "SELECT * FROM history WHERE 1=1"
    params = []

    if event_type:
        query += " AND event_type = ?"
        params.append(event_type)

    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)

    if keyword:
        query += " AND payload LIKE ?"
        params.append(f"%{keyword}%")

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    c.execute(query, tuple(params))
    rows = c.fetchall()
    conn.close()

    history_list = []
    for row in rows:
        history_list.append({
            "id": row[0],
            "event_type": row[1],
            "user_id": row[2],
            "session_id": row[3],
            "payload": json.loads(row[4]) if row[4] else None,
            "timestamp": row[5]
        })
    return history_list
