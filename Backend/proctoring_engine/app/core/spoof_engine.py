from collections import deque

import cv2
from .models.minifasnet import MiniFASNetDetector
from .models.facebagnet import FaceBagNetDetector
from .decision_policy import DecisionPolicy
import numpy as np
from .image_audit import save_audit_image

class SpoofEngine:
    """
    Unified spoof detection engine.
    Runs MiniFASNet + FaceBagNet and applies temporal decision logic.
    """

    def __init__(self):
        self.minifasnet = MiniFASNetDetector()
        self.facebagnet = FaceBagNetDetector()
        self.policy = DecisionPolicy()

    def _decode_image(self, blob: bytes):
        if blob is None:
            return None
        arr = np.frombuffer(blob, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def process(self, *, mini_v2=None, facebag=None, session,session_id: str):
        """
        Returns:
        {
          liveness: "real" | "suspicious" | "fake",
          spoof_score: float,
          debug: dict
        }
        """

        scores = {}

        if mini_v2:
            scores["minifasnet"] = self.minifasnet.predict(mini_v2)

        if facebag is not None:
            face_img = self._decode_image(facebag)
            if face_img is not None:
                scores["facebagnet"] = self.facebagnet.predict(face_img)

        if not scores:
            return {
                "liveness": "unstable",
                "spoof_score": None,
                "components": {}
            }

        result = self.policy.update(scores, session)

        if mini_v2:
            save_audit_image(
                mini_v2,
                session_id=session_id,
                model_name="minifasnet_v2",
                decision=result["liveness"],
                score=scores["minifasnet"]
            )

        if facebag:
            save_audit_image(
                facebag,
                session_id=session_id,
                model_name="facebagnet",
                decision=result["liveness"],
                score=scores["facebagnet"]
            )

        return result
