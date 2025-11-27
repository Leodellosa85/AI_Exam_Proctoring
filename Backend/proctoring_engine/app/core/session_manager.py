import time
import json
import os
from .logger import log_event, now_iso
from ..config.settings import LOG_DIR

SESSIONS = {}

def create_session():
    import uuid
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "created_at": now_iso(),
        "events": [],
        "terminated": False,
        "last_face_seen": time.time(),
        "lookaway_counter": 0,
        "flag_lookaway": False,
        "flag_multi": False
    }
    log_event(SESSIONS[session_id], "session_started", {})
    return session_id, SESSIONS[session_id]

def get_session(session_id: str):
    return SESSIONS.get(session_id)

def save_session_log(session_id: str):
    session = SESSIONS.get(session_id)
    if not session:
        return None
    path = os.path.join(LOG_DIR, f"{session_id}.json")
    with open(path, "w") as f:
        json.dump(session, f, indent=2)
    return path
