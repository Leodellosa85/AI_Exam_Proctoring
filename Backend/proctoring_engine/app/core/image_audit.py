import os
import cv2
import numpy as np
from datetime import datetime
from typing import Optional

CORE_DIR = os.path.dirname(__file__)                 # app/core/models
APP_DIR = os.path.dirname(os.path.dirname(CORE_DIR)) # app
AUDIT_ROOT = os.path.join(APP_DIR, "audit_logs")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
    if not image_bytes:
        return None
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return img


def save_audit_image(
    image_bytes: bytes,
    *,
    session_id: str,
    model_name: str,
    decision: str,
    score: float
):
    """
    decision: real | fake | suspicious
    model_name: minifasnet_v2 | minifasnet_v1se | facebagnet
    """

    img = decode_image(image_bytes)
    if img is None:
        return

    decision = decision.lower()
    ensure_dir(AUDIT_ROOT)

    save_dir = os.path.join(AUDIT_ROOT, decision, model_name)
    ensure_dir(save_dir)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{session_id}_{score:.4f}_{ts}.png"

    cv2.imwrite(
        os.path.join(save_dir, filename),
        img,
        [cv2.IMWRITE_PNG_COMPRESSION, 0]
    )
