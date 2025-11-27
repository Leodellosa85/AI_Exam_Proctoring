from PIL import Image

class BaseDetector:
    def detect_faces(self, img: Image.Image) -> int:
        """Return num faces"""
        raise NotImplementedError

    def analyze_head_pose(self, img: Image.Image):
        """Return (yaw, pitch)"""
        raise NotImplementedError
