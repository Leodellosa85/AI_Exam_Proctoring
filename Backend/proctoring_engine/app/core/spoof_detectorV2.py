import onnxruntime as ort
import numpy as np
import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # app/
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "facebagnet_color_96.onnx")

class SpoofDetector:

    def __init__(self, model_path=MODEL_PATH):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = self.session.get_inputs()[0].shape[2]

        print("✅ FaceBagNet loaded")
        print("Input size:", self.img_size)

    # def preprocess(self, face_bgr):
    #     img = cv2.resize(face_bgr, (self.img_size, self.img_size))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = img.astype(np.float32) / 255.0

    #     mean = [0.485, 0.456, 0.406]
    #     std  = [0.229, 0.224, 0.225]

    #     img[:,:,0] = (img[:,:,0] - mean[0]) / std[0]
    #     img[:,:,1] = (img[:,:,1] - mean[1]) / std[1]
    #     img[:,:,2] = (img[:,:,2] - mean[2]) / std[2]

    #     img = np.transpose(img, (2, 0, 1))
    #     img = np.expand_dims(img, axis=0)

    #     return img.astype(np.float32)
    
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

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    # def predict(self, face_bgr):
    #     if face_bgr is None or face_bgr.size == 0:
    #         return None

    #     inp = self.preprocess(face_bgr)
    #     output = self.session.run(None, {self.input_name: inp})[0][0]
    #     print("RAW logits:", output)

    #     probs = self.softmax(output)
    #     print("probs:", probs)

    #     fake_score = float(probs[0])  # index 1 = spoof

    #     return fake_score
    
    def predict(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            return None

        inp = self.preprocess(face_bgr)
        output = self.session.run(None, {self.input_name: inp})[0][0]
        # print("RAW logits:", output)
        
        # # Apply softmax to get 0-1 probabilities
        # probs = self.softmax(output)
        # print("probs:", probs)
        
        # # Logit index 0: Real, Index 1: Spoof
        # liveness_score = float(probs[0]) # Higher means more likely REAL
        # spoof_score = float(probs[1])   # Higher means more likely FAKE

        # return spoof_score  # Or return a dict with both
        # RAW Logit Difference logic
        # If Index 0 = Real and Index 1 = Spoof
        # A positive diff means REAL, a negative diff means SPOOF
        score_diff = output[0] - output[1]
        
        # Normalizing the diff to a 0-1 scale for your app
        # Since your range is around 300-400, we scale it
        # normalized_spoof_score = 1.0 / (1.0 + np.exp(score_diff / 100.0)) 
        # normalized_spoof_score = 1.0 / (1.0 + np.exp(score_diff / 500.0))
        normalized_spoof_score = 1.0 / (1.0 + np.exp(score_diff / 250.0))
        print("Normalized Spoof Score:", normalized_spoof_score)
        return normalized_spoof_score
