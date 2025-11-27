from datetime import datetime

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def log_event(session: dict, event_type: str, detail: dict):
    entry = {"ts": now_iso(), "type": event_type, "detail": detail}
    session["events"].append(entry)
    print("LOG:", entry)
