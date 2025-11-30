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
    Hybrid AI Proctoring Detector with bounding boxes:

    ✔ Face Detection (HF model if available, fallback: MediaPipe)
    ✔ Liveness Detection (AI anti-spoof model)
    """

    def __init__(self, score_threshold=0.5):
        self.score_threshold = score_threshold

        # Lazy load flags
        self.fd_model_loaded = False
        self.lv_model_loaded = False

        # Model references
        self.fd_processor = None
        self.fd_model = None
        self.lv_processor = None
        self.lv_model = None

        # MediaPipe fallback
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

    # Lazy-load HF Face Detection
    def _load_face_model(self):
        if self.fd_model_loaded:
            return -1
        if HF_FACE_DETECT_MODEL:
            try:
                self.fd_processor = AutoProcessor.from_pretrained(HF_FACE_DETECT_MODEL)
                self.fd_model = AutoModelForObjectDetection.from_pretrained(HF_FACE_DETECT_MODEL)
                self.fd_model.eval()
                self.fd_model_loaded = True
                print(f"[HF] Loaded face detection model: {HF_FACE_DETECT_MODEL}")
            except Exception as e:
                print("[HF] Failed to load face detection model:", e)

    # Lazy-load HF Liveness model
    def _load_liveness_model(self):
        if self.lv_model_loaded:
            return -1
        if HF_LIVENESS_MODEL:
            try:
                self.lv_processor = AutoProcessor.from_pretrained(HF_LIVENESS_MODEL)
                self.lv_model = AutoModelForImageClassification.from_pretrained(HF_LIVENESS_MODEL)
                self.lv_model.eval()
                self.lv_model_loaded = True
                print(f"[HF] Loaded liveness model: {HF_LIVENESS_MODEL}")
            except Exception as e:
                print("[HF] Failed to load liveness model:", e)

    # -----------------------------
    # Face Detection + return bounding boxes
    # -----------------------------
    def detect_faces(self, img: Image.Image):
        """
        Returns:
            num_faces (int)
            boxes (list of dict: x, y, width, height)
        """
        self._load_face_model()
        boxes = []

        # ---- HF MODEL ----
        if self.fd_model_loaded:
            try:
                inputs = self.fd_processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    out = self.fd_model(**inputs)

                if hasattr(out, "logits"):
                    scores = out.logits.softmax(-1)[..., :-1].max(-1).values
                    boxes = [
                        {"x": int(b[0]), "y": int(b[1]),
                         "width": int(b[2]-b[0]), "height": int(b[3]-b[1])}
                        for i, b in enumerate(out.pred_boxes[0]) if scores[i] > self.score_threshold
                    ]
                    return len(boxes), boxes

                if hasattr(out, "scores"):
                    boxes = [
                        {"x": int(b[0]), "y": int(b[1]),
                         "width": int(b[2]-b[0]), "height": int(b[3]-b[1])}
                        for i, b in enumerate(out.pred_boxes) if out.scores[i] > self.score_threshold
                    ]
                    return len(boxes), boxes

            except Exception as e:
                print("[HF] Face detection error:", e)

        # ---- MediaPipe fallback ----
        print("[HF] Using MediaPipe fallback for face detection.")
        np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        result = self.mp_face_detection.process(np_img)
        if result.detections:
            h, w, _ = np_img.shape
            for det in result.detections:
                bbox = det.location_data.relative_bounding_box
                boxes.append({
                    "x": int(bbox.xmin * w),
                    "y": int(bbox.ymin * h),
                    "width": int(bbox.width * w),
                    "height": int(bbox.height * h)
                })
            return len(boxes), boxes

        return 0, []

    # -----------------------------
    # Liveness check
    # -----------------------------
    def check_liveness(self, face_crop: Image.Image) -> str:
        self._load_liveness_model()
        if not self.lv_model_loaded:
            return "unknown"

        try:
            inputs = self.lv_processor(images=face_crop, return_tensors="pt")
            with torch.no_grad():
                out = self.lv_model(**inputs)

            pred = out.logits.softmax(-1).argmax().item()
            label = self.lv_model.config.id2label[pred].lower()
            return "real" if "real" in label else "spoof"

        except Exception as e:
            print("[HF] Liveness error:", e)
            return "error"
