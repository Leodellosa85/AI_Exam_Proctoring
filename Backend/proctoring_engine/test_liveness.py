from app.core.liveness_engine import LivenessEngine

engine = LivenessEngine(
    "app/models/liveness_model.tflite",
    "app/models/liveness_scaler.pkl"
)

print("Liveness engine loaded successfully!")
