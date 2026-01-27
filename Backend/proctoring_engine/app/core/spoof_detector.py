import onnxruntime as ort
import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # app/
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "MiniFASNetV2.onnx")

IMG_SIZE = 80  

class SpoofDetector:
    def __init__(self):
        self.session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        print("✅ spoof detector ready")

    def preprocess(self, face_bgr):
        img = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        # img = np.transpose(img, (2, 0, 1))
        # img = np.expand_dims(img, axis=0)
        # return img
        
        # Just convert to float32
        img = img.astype(np.float32) 
        
        # 4. Transpose to CHW (Correct)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, face_bgr):
        if face_bgr is None:
            return None

        if face_bgr.size == 0:
            return None

        h, w = face_bgr.shape[:2]
        if h < 10 or w < 10:
            return None

        inp = self.preprocess(face_bgr)

        # output = self.session.run(None, {self.input_name: inp})[0]

        # if output.shape[-1] == 1:
        #     logit = float(output[0][0])
        # else:
        #     logit = float(output[0][1])

        # # Convert logit → probability
        # score = 1.0 / (1.0 + np.exp(-logit))

        # print("logit:", logit, "prob:", score)
       
        # return score


         # Get raw logits (shape: 1, 2)
        # logits = self.session.run(None, {self.input_name: inp})[0][0]
    
        # fake_logit = logits[0]
        # real_logit = logits[1]

        # # Calculate probability specifically between Real and Fake
        # # This ignores the background logit (Index 2)
        # exps = np.exp([fake_logit, real_logit])
        # probs = exps / np.sum(exps)
        
        # # probs[0] is probability of Fake
        # # probs[1] is probability of Real
        # fake_prob = float(probs[0])
        # real_prob = float(probs[1])

        # print(f"RAW Logits: {logits}")
        # print(f"Relative Probs -> Real: {real_prob:.4f}, Fake: {fake_prob:.4f}")
        
        # return fake_prob
        
        logits = self.session.run(None, {self.input_name: inp})[0][0]
    
        # 2. Calculate Softmax over ALL 3 classes (Fake, Real, Fake/Background)
        # Subtracting np.max for numerical stability
        exps = np.exp(logits - np.max(logits))
        probs = exps / np.sum(exps)
        
        # MiniFASNet Mapping: Index 1 is REAL.
        # Therefore, Spoof Prob = (Prob of Index 0) + (Prob of Index 2)
        # Or simply: 1.0 - (Prob of Index 1)
        real_probability = probs[1]
        spoof_score = 1.0 - real_probability
        
        # Debug prints to see the shift
        print(f"RAW Logits: {logits}")
        print(f"Real Prob (Idx 1): {real_probability:.4f}")
        
        return float(spoof_score)
        

