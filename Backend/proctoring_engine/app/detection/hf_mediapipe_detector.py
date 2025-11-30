import io
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import AutoModelForObjectDetection, AutoProcessor
import mediapipe as mp
from ..config.settings import HF_MODEL_REPO_ID


class HF_MediaPipe_Detector:
    """
    Hybrid detector for AI Proctoring:
    - Hugging Face model handles face detection (AI-powered requirement)
    - MediaPipe handles head pose (yaw/pitch)
    """

    def __init__(self, score_threshold=0.5):
        self.enabled_hf = False
        self.score_threshold = score_threshold
        if HF_MODEL_REPO_ID:
            try:
                self.processor = AutoProcessor.from_pretrained(HF_MODEL_REPO_ID)
                self.model = AutoModelForObjectDetection.from_pretrained(HF_MODEL_REPO_ID)
                self.model.eval()
                self.enabled_hf = True
                print(f"[HF] Loaded model: {HF_MODEL_REPO_ID}")
            except Exception as e:
                print("[HF] Failed to load model:", e)

        else:
            print("[HF] No model ID provided")

        # -----------------------------
        #  Load MediaPipe components
        # -----------------------------
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True
        )

    # ------------------------------------------------------------
    # 🔵 FACE DETECTION USING HUGGING FACE (AI MODEL)
    # ------------------------------------------------------------
    def detect_faces(self, img: Image.Image) -> int:
        """Return number of faces detected in a PIL Image using HF model."""
        if not self.enabled_hf:
            return -1
        try:
            inputs = self.processor(images=img, return_tensors="pt")

            with torch.no_grad():
                out = self.model(**inputs)

            # If logits exist → DETR/YOLO-like model
            if hasattr(out, "logits"):
                scores = out.logits.softmax(-1)[..., :-1].max(-1).values
                num_faces = (scores > self.score_threshold).sum().item()
                return int(num_faces)

            # If scores exist → SSD-style model
            if hasattr(out, "scores"):
                num_faces = (out.scores > self.score_threshold).sum().item()
                return int(num_faces)

            return 0
        except Exception as e:
            print("[HF] Detection error:", e)
            return -1

    # ------------------------------------------------------------
    # 🔵 HEAD POSE USING MEDIAPIPE
    # ------------------------------------------------------------
    def analyze_head_pose(self, img: Image.Image):
        """
        Returns:
            yaw (float): left-right rotation
            pitch (float): up-down rotation
        """
        rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w, _ = rgb.shape

        res = self.mp_face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            return (0.0, 0.0)

        lm = res.multi_face_landmarks[0].landmark

        # Helper: extract (x,y)
        def xy(i):
            return (lm[i].x * w, lm[i].y * h)

        nose = xy(1)
        left_eye = xy(33)
        right_eye = xy(263)

        eye_dist = abs(left_eye[0] - right_eye[0]) + 1e-6

        yaw = ((nose[0] - ((left_eye[0] + right_eye[0]) / 2)) / eye_dist) * 50
        pitch = ((nose[1] - ((left_eye[1] + right_eye[1]) / 2)) / eye_dist) * 50

        return float(yaw), float(pitch)
