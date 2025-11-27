import os
from dotenv import load_dotenv

load_dotenv()

FACE_ABSENCE_TIMEOUT = float(os.getenv("FACE_ABSENCE_TIMEOUT", 8))
LOOKAWAY_THRESHOLD = int(os.getenv("LOOKAWAY_THRESHOLD", 3))
LOOKAWAY_MIN_ANGLE = float(os.getenv("LOOKAWAY_MIN_ANGLE", 25.0))
DETECTION_FPS = int(os.getenv("DETECTION_FPS", 6))

LOG_DIR = os.getenv("LOG_DIR", "./session_logs")
HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", None)

os.makedirs(LOG_DIR, exist_ok=True)
