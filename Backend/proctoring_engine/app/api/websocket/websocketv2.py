from fastapi import WebSocket, WebSocketDisconnect
from collections import deque
from datetime import datetime
import os

import cv2
import numpy as np

from ...core.session_manager import (
    get_session,
    create_session,
)
from ...core.spoof_detector import SpoofDetector


# ======================================================
# Configuration
# ======================================================

SCORE_BUFFER_SIZE = 5
FAKE_THRESHOLD = 0.70
MIN_STABLE_FRAMES = 3

DEBUG_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "debug_facesminifasnet",
)
os.makedirs(DEBUG_DIR, exist_ok=True)

spoof_detector = SpoofDetector()


# ======================================================
# Helpers
# ======================================================

def save_debug_image(image_bytes: bytes, prefix: str, session_id: str):
    """
    Decode binary image bytes and save as JPEG for debugging / audit.
    """
    img_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        return

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{session_id}_{timestamp}.png"  # 🔥 PNG
    filepath = os.path.join(DEBUG_DIR, filename)

    # 🔥 Force lossless write
    cv2.imwrite(filepath, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# ======================================================
# WebSocket Handler
# ======================================================

class DirectionWebSocketV2:

    async def handle(self, websocket: WebSocket, session_id: str):
        await websocket.accept()

        # Retrieve or create session
        session = get_session(session_id)
        if session is None:
            session_id, session = create_session()

        # Rolling buffer for score smoothing
        session.setdefault(
            "score_buffer",
            deque(maxlen=SCORE_BUFFER_SIZE),
        )

        try:
            while True:
                # 1️⃣ Receive control message
                header = await websocket.receive_json()

                if header.get("type") != "liveness_payload":
                    continue

                expect_mini = header.get("mini", False)
                expect_facebag = header.get("facebag", False)

                blobs = []

                # 2️⃣ Receive MiniFASNet crops
                if expect_mini:
                    blob_v2 = await websocket.receive_bytes()
                    blob_v1se = await websocket.receive_bytes()
                    blobs.extend([blob_v2, blob_v1se])

                # 3️⃣ Receive FaceBagNet crop
                if expect_facebag:
                    blob_facebag = await websocket.receive_bytes()
                    blobs.append(blob_facebag)

                # 4️⃣ Run MiniFASNet (example)
                raw_score = spoof_detector.predict(blob_v2, blob_v1se)
                if raw_score is None:
                    continue
                
                # 4️⃣ Temporal smoothing
                score_buffer = session["score_buffer"]
                score_buffer.append(raw_score)

                stable_score = sum(score_buffer) / len(score_buffer)

                # 5️⃣ Decision logic
                is_fake = (
                    len(score_buffer) >= MIN_STABLE_FRAMES
                    and stable_score > FAKE_THRESHOLD
                )

                # 6️⃣ Save images when spoof detected
                if is_fake:
                    save_debug_image(blob_v2, "v2_fake", session_id)
                    save_debug_image(blob_v1se, "v1se_fake", session_id)
                else:
                    save_debug_image(blob_v2, "v2_real", session_id)
                    save_debug_image(blob_v1se, "v1se_real", session_id)
                    save_debug_image(blob_facebag, "facebag_real", session_id)


                # 7️⃣ Send result to client
                await websocket.send_json({
                    "type": "spoof_result",
                    "liveness": "fake" if is_fake else "real",
                    "spoof_score": round(float(stable_score), 4),
                })

        except WebSocketDisconnect:
            print(f"[WebSocket] Session {session_id} disconnected")
