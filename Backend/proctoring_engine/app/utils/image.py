import base64
import io
import cv2
import numpy as np
from PIL import Image

def base64_to_bytes(b64: str) -> bytes:
    return base64.b64decode(b64)

def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def cv2_from_pil(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
