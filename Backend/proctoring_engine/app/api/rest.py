from fastapi import APIRouter, UploadFile, File, Form
from ..core.session_manager import create_session, get_session, save_session_log
from ..core.logger import log_event
from ..core.counters import handle_multiperson, handle_lookaway
from ..core.report import build_report
from ..utils.image import pil_from_bytes, base64_to_bytes
from ..detection.mediapipe_detector import MediapipeDetector
from ..config.settings import FACE_ABSENCE_TIMEOUT, LOOKAWAY_MIN_ANGLE

import time

router = APIRouter()
# detector is an instance of a face detection/head pose analyzer that will be reused for each frame.
detector = MediapipeDetector()

"""
To track the student's activity over time (face presence, looking away, multi-person detection),
 we need a persistent in-memory object that stores all relevant info until the exam ends."""

@router.post("/sessions/start")
async def start_session():
    session_id, session = create_session()
    return {"session_id": session_id, "started_at": session["created_at"]}

"""Client sends one frame (image) at a time to /frame.
get_session() retrieves the current session state.
If session doesn't exist (e.g., expired or wrong ID), return an error.
"""
@router.post("/sessions/{session_id}/frame")
async def post_frame(session_id: str, image_file: UploadFile = File(...)):
    session = get_session(session_id)
    if not session:
        return {"error": "Session not found"}

    """Reads the uploaded image.
       Converts it to PIL.Image.
       Detects number of faces in the frame using MediapipeDetector."""
    bytes_data = await image_file.read()

    """pil_from_bytes() does:
    Image.open(io.BytesIO(bytes)).convert("RGB")"""
    img = pil_from_bytes(bytes_data)

    num_faces = detector.detect_faces(img)

    # face absence
    now = time.time()
    if num_faces == 0:
        if now - session["last_face_seen"] >= FACE_ABSENCE_TIMEOUT:
            session["terminated"] = True
            log_event(session, "terminate_exam", {"reason": "face_absent"})
            save_session_log(session_id)
            return {"terminate": True}
    else:
        session["last_face_seen"] = now

    # multiperson
    multi = handle_multiperson(session, num_faces)

    # lookaway
    yaw, pitch = detector.analyze_head_pose(img)
    handle_lookaway(session, yaw, pitch, LOOKAWAY_MIN_ANGLE)

    return {
        "num_faces": num_faces,
        "yaw": yaw, "pitch": pitch,
        "flags": {
            "looking_away": session["flag_lookaway"],
            "multiperson": session["flag_multi"]
        },
        "terminate": session["terminated"]
    }

@router.post("/sessions/{session_id}/frame_base64")
async def post_b64(session_id: str, b64: str = Form(...)):
    session = get_session(session_id)
    if not session:
        return {"error": "Session not found"}
    
    if b64.startswith("data:image"):
        b64 = b64.split(",")[1]

    img = pil_from_bytes(base64_to_bytes(b64))

    num_faces = detector.detect_faces(img)

    now = time.time()
    if num_faces == 0:
        if now - session["last_face_seen"] >= FACE_ABSENCE_TIMEOUT:
            session["terminated"] = True
            log_event(session, "terminate_exam", {"reason": "face_absent"})
            save_session_log(session_id)
            return {"terminate": True}
    else:
        session["last_face_seen"] = now

    multi = handle_multiperson(session, num_faces)

    yaw, pitch = detector.analyze_head_pose(img)
    handle_lookaway(session, yaw, pitch, LOOKAWAY_MIN_ANGLE)

    return {
        "num_faces": num_faces,
        "yaw": yaw, "pitch": pitch,
        "flags": {
            "looking_away": session["flag_lookaway"],
            "multiperson": session["flag_multi"]
        },
        "terminate": session["terminated"]
    }

@router.post("/sessions/{session_id}/end")
async def end_session(session_id: str):
    session = get_session(session_id)
    if not session:
        return {"error": "not found"}

    log_event(session, "session_ended", {})
    path = save_session_log(session_id)
    return {"saved": True, "path": path}

@router.get("/sessions/{session_id}/report")
async def report(session_id: str):
    session = get_session(session_id)
    if not session:
        return {"error": "not found"}

    return build_report(session, session_id)



