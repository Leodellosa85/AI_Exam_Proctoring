import onnxruntime as ort
import numpy as np
import cv2
import os

CORE_DIR = os.path.dirname(__file__)                 # app/core/models
APP_DIR = os.path.dirname(os.path.dirname(CORE_DIR)) # app
MODELS_DIR = os.path.join(APP_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "facebagnet_color_96.onnx")


class FaceBagNetDetector:

    def __init__(self, model_path=MODEL_PATH):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = self.session.get_inputs()[0].shape[2]

        print("✅ FaceBagNet loaded")
        print("Input size:", self.img_size)
    
    def preprocess(self, face_bgr):
        # 1. Resize and convert to RGB
        img = cv2.resize(face_bgr, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Scale to [0, 1]
        img = img.astype(np.float32) / 255.0

        # 3. Vectorized Normalization (much faster)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # 4. HWC to CHW and add Batch dimension
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img

    def predict(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            return None

        inp = self.preprocess(face_bgr)
        output = self.session.run(None, {self.input_name: inp})[0][0]
        score_diff = output[0] - output[1]
        
        normalized_spoof_score = 1.0 / (1.0 + np.exp(-score_diff / 250.0))
        print("Normalized Spoof Score:", normalized_spoof_score)
        return float(normalized_spoof_score)
