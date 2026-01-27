import os
import numpy as np
import joblib
import tensorflow as tf
from collections import deque

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # app/
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "liveness_model.tflite")
SCALER_PATH = os.path.join(MODELS_DIR, "liveness_scaler.pkl")


class LivenessEngine:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH, window_size=16):
        print("Loading scaler from:", scaler_path)
        self.scaler = joblib.load(scaler_path)

        print("Loading TFLite model from:", model_path)
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.window_size = window_size
        self.buffers = {}
        self.score_buffers = {} 

        print("✅ Liveness engine ready")


    def reset_session(self, session_id):
        self.buffers[session_id] = deque(maxlen=self.window_size)
        self.score_buffers[session_id] = deque(maxlen=8)  # smoothing window


    def add_features(self, session_id, features):
        """
        features = [yaw, pitch, roll, motion]
        """
        if session_id not in self.buffers:
            self.reset_session(session_id)

        self.buffers[session_id].append(features)

        if len(self.buffers[session_id]) < self.window_size:
            return None  # not enough data yet
        
        score = self.predict(session_id)

        self.score_buffers[session_id].append(score)

        return score

    def get_smoothed_score(self, session_id):
        buf = self.score_buffers.get(session_id)
        if not buf or len(buf) == 0:
            return None
        return float(sum(buf) / len(buf))


    def predict(self, session_id):
        window = np.array(self.buffers[session_id], dtype=np.float32)

        # Normalize
        flat = window.reshape(-1, 4)
        flat = self.scaler.transform(flat)
        window = flat.reshape(1, self.window_size, 4).astype(np.float32)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], window)
        self.interpreter.invoke()

        score = float(self.interpreter.get_tensor(self.output_details[0]['index'])[0][0])
        return score
