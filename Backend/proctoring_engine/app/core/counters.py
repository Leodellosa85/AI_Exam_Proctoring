from .logger import log_event
from ..config.settings import LOOKAWAY_THRESHOLD

def handle_multiperson(session: dict, num_faces: int):
    multi_detected = False
    if num_faces > 1:
        multi_detected = True
        if not session["flag_multi"]:
            log_event(session, "warning_multiperson", {"faces": num_faces})
            log_event(session, "flag_multiperson", {"faces": num_faces})
            session["flag_multi"] = True
    else:
        session["flag_multi"] = False
    return multi_detected

def handle_lookaway(session: dict, yaw: float, pitch: float, min_angle: float):
    is_looking_away = abs(yaw) > min_angle or abs(pitch) > min_angle
    if not is_looking_away:
        return False

    session["lookaway_counter"] += 1
    log_event(session, "warning_lookaway", {
        "count": session["lookaway_counter"],
        "yaw": yaw, "pitch": pitch
    })

    if session["lookaway_counter"] > LOOKAWAY_THRESHOLD and not session["flag_lookaway"]:
        log_event(session, "flag_lookaway", {"count": session["lookaway_counter"]})
        session["flag_lookaway"] = True

    return True
