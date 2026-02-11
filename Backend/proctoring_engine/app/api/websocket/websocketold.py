from fastapi import WebSocket, WebSocketDisconnect
from ...core.session_manager import get_session, create_session, save_session_log
from ...core.logger import log_event
from ...config.settings import FACE_ABSENCE_TIMEOUT
from ...core.liveness_engine import LivenessEngine
import time

liveness_engine = LivenessEngine()

class DirectionWebSocket:

    def __init__(self, detector=None):
        self.detector = detector

    async def handle(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        session = get_session(session_id)

        if not session:
            session_id, session = create_session()

        liveness_engine.reset_session(session_id)

        log_event(session, "ws_connected", {})
        print("WebSocket connected:", session_id)


        try:
            while True:
                data = await websocket.receive_json()

                if data["type"] == "liveness_features":
                    features = data["features"]  # [yaw, pitch, roll, motion]

                    score = liveness_engine.add_features(session_id, features)
                    avg_score = liveness_engine.get_smoothed_score(session_id)
                    motion = features[3]

                    liveness_status = "unknown"

                    # if score is not None:
                    #     if score > 0.9:
                    #         liveness_status = "real"
                    #         session["flag_spoof"] = False
                    #     elif score < 0.7:
                    #         liveness_status = "fake"
                    #         session["flag_spoof"] = True
                    #         log_event(session, "spoof_detected", {"score": score})
                    #     else:
                    #         liveness_status = "suspicious"
                    liveness_status = "unknown"

                    if score is not None:

                        # 🧊 Photo detection
                        if score < 0.70 and motion < 0.03:
                            liveness_status = "fake"
                            session["flag_spoof"] = True

                        # 🚨 Hard spoof
                        elif score < 0.45:
                            liveness_status = "fake"
                            session["flag_spoof"] = True

                        # ✅ Stable real
                        elif avg_score and avg_score > 0.78:
                            liveness_status = "real"
                            session["flag_spoof"] = False

                        # 🧠 Natural motion
                        elif avg_score and avg_score > 0.65 and motion > 0.12:
                            liveness_status = "real"
                            session["flag_spoof"] = False

                        else:
                            liveness_status = "suspicious"
                            session["flag_spoof"] = False

                    # ---- Cooldown override (PUT HERE) ----
                    now = time.time()

                    if session.get("last_status") == "real" and liveness_status == "suspicious":
                        if now - session.get("last_real_time", 0) < 3:
                            liveness_status = "real"

                    # ---- Save state for next frame ----
                    session["last_status"] = liveness_status
                    if liveness_status == "real":
                        session["last_real_time"] = now

                    await websocket.send_json({
                        "liveness": liveness_status,
                        "score": score,
                        "flags": {
                            "spoof": session.get("flag_spoof", False)
                        },
                        "terminate": session.get("terminated", False)
                    })

                elif data["type"] == "heartbeat":
                    session["last_face_seen"] = time.time()

                if session.get("terminated"):
                    break

        except WebSocketDisconnect:
            log_event(session, "ws_disconnected", {})
            save_session_log(session_id)
