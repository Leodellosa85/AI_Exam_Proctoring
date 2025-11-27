def build_report(session: dict, session_id: str):
    events = session.get("events", [])

    warnings = [e for e in events if e["type"].startswith("warning")]
    flags = [e for e in events if e["type"].startswith("flag")]
    term = [e for e in events if "terminate" in e["type"]]

    return {
        "session_id": session_id,
        "created_at": session["created_at"],
        "total_events": len(events),
        "warnings": len(warnings),
        "flags": len(flags),
        "terminated": session.get("terminated", False) or bool(term),
        "events": events
    }
