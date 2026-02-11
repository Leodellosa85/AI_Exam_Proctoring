import onnxruntime as ort
import cv2
import numpy as np
import os


# minifasnet.py lives in: app/core/models/
CORE_DIR = os.path.dirname(__file__)            # app/core/models
APP_DIR = os.path.dirname(os.path.dirname(CORE_DIR))  # app
MODELS_DIR = os.path.join(APP_DIR, "models")    # app/models


MODEL_PATH_V2 = os.path.join(MODELS_DIR, "MiniFASNetV2.onnx")

class MiniFASNetDetector:
    def __init__(self, model_path_v2=MODEL_PATH_V2):
        self.session_v2 = ort.InferenceSession(model_path_v2, providers=["CPUExecutionProvider"])
        print("MiniFASNetV2 loaded")

    def _preprocess(self, blob):
        # 1. Decode
        arr = np.frombuffer(blob, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None: return None
        
        # 2. Color Swap (Crucial: MiniFASNet was trained on RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. Resize (80x80)
        # img = cv2.resize(img, (80, 80), interpolation=cv2.INTER_AREA)

        h, w, _ = img.shape
        if (h, w) != (80, 80):
            raise ValueError(f"Unexpected input size: {w}x{h}, expected 80x80")
        
        # 4. Correct Normalization for MiniFASNet (MUST be float32)
        # This converts 0-255 to roughly -1.0 to 1.0
        img = (img.astype(np.float32) - 127.5) / 128.0
        
        # 5. HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def _get_score(self, session, inp):
        logits = session.run(None, {session.get_inputs()[0].name: inp})[0]

        # Softmax
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        # Class order: [fake, real, background]
        fake_prob = probs[0][0]
        real_prob = probs[0][1]
        bg_prob = probs[0][2]

        # 🔥 Ignore background and renormalize
        denom = fake_prob + real_prob
        if denom < 1e-6:
            return 0.0

        real_norm = real_prob / denom
        return float(real_norm)

    def predict(self, blob_v2):
        in_v2 = self._preprocess(blob_v2)
        
        if in_v2 is None: return None

        real_v2 = self._get_score(self.session_v2, in_v2)

        print(f"Debug: V2_Real={real_v2:.4f}")
        spoof_score = 1.0 - real_v2
        print(f"Debug: V2_Spoof_Score={spoof_score:.4f}")
        return spoof_score
