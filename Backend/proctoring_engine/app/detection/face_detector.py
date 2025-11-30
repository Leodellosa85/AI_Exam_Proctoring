import io
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import AutoModelForObjectDetection, AutoProcessor, AutoModelForImageClassification
import mediapipe as mp
from ..config.settings import HF_FACE_DETECT_MODEL, HF_LIVENESS_MODEL


class HF_MediaPipe_Detector:
    """
    Hybrid AI Proctoring Detector:

    ✔ Face Detection (HF model if available, fallback: MediaPipe)
    ✔ Liveness Detection (AI anti-spoof model)
    ✔ Mirror / Reflection detection (via spoof detection)
    """

    def __init__(self, score_threshold=0.5):
        self.score_threshold = score_threshold

        # -------------------------
        # Load HuggingFace FACE DETECTION MODEL
        # -------------------------
        self.hf_face_enabled = False
        if HF_FACE_DETECT_MODEL:
            try:
                self.fd_processor = AutoProcessor.from_pretrained(HF_FACE_DETECT_MODEL)
                self.fd_model = AutoModelForObjectDetection.from_pretrained(HF_FACE_DETECT_MODEL)
                self.fd_model.eval()
                self.hf_face_enabled = True
                print(f"[HF] Loaded face detection model: {HF_FACE_DETECT_MODEL}")
            except Exception as e:
                print("[HF] Failed to load face detection model:", e)

        # -------------------------
        # Load HuggingFace LIVENESS MODEL
        # -------------------------
        self.hf_liveness_enabled = False
        if HF_LIVENESS_MODEL:
            try:
                self.lv_processor = AutoProcessor.from_pretrained(HF_LIVENESS_MODEL)
                self.lv_model = AutoModelForImageClassification.from_pretrained(HF_LIVENESS_MODEL)
                self.lv_model.eval()
                self.hf_liveness_enabled = True
                print(f"[HF] Loaded liveness model: {HF_LIVENESS_MODEL}")
            except Exception as e:
                print("[HF] Failed to load liveness model:", e)
                

        # Fallback face detector (MediaPipe)
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

    # --------------------------------------------------------
    # FACE DETECTION (HF model → MediaPipe fallback)
    # --------------------------------------------------------
    def detect_faces(self, img: Image.Image) -> int:
        """Return number of faces detected."""
        
        # ---- HF MODEL ----
        if self.hf_face_enabled:
            try:
                inputs = self.fd_processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    out = self.fd_model(**inputs)

                if hasattr(out, "logits"):
                    scores = out.logits.softmax(-1)[..., :-1].max(-1).values
                    return int((scores > self.score_threshold).sum().item())

                if hasattr(out, "scores"):
                    return int((out.scores > self.score_threshold).sum().item())

            except Exception as e:
                print("[HF] Face detection error:", e)

        # ---- FALLBACK: MediaPipe ----
        np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        result = self.mp_face_detection.process(np_img)
        if result.detections:
            return len(result.detections)
        return 0

    # --------------------------------------------------------
    # LIVENESS / ANTI-SPOOFING (Real vs Fake face)
    # --------------------------------------------------------
    def check_liveness(self, face_crop: Image.Image) -> str:
        """
        Returns:
            "real" or "spoof"
        """

        if not self.hf_liveness_enabled:
            return "unknown"

        try:
            inputs = self.lv_processor(images=face_crop, return_tensors="pt")
            with torch.no_grad():
                out = self.lv_model(**inputs)

            pred = out.logits.softmax(-1).argmax().item()
            label = self.lv_model.config.id2label[pred].lower()

            if "real" in label:
                return "real"
            return "spoof"

        except Exception as e:
            print("[HF] Liveness error:", e)
            return "error"
