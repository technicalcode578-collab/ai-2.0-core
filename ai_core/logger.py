import json, time, os
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
EVENT_FILE = DATA_DIR / "events.jsonl"

def capture_event(event: dict):
    # attach provenance fields
    record = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": "ai2.0.v1",
        "provenance": {
            "source": event.get("source","client"),
            "session_id": event.get("session_id"),
            "user_id": event.get("user_id"),
            "client": event.get("client","codespace")
        },
        "payload": event.get("payload", {})
    }
    with open(EVENT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\\n")
    return record
