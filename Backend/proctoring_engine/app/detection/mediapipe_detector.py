import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from .base_detector import BaseDetector

"""
    cv2 - OpenCV (image conversions)
    mediapipe - Google’s face detection & face mesh library
    numpy - used for image array conversion
    Pillow (PIL) - images come in PIL format from Django/FastAPI
"""

mp_face = mp.solutions.face_detection
mp_mesh = mp.solutions.face_mesh

class MediapipeDetector(BaseDetector):
    def detect_faces(self, img: Image.Image) -> int:
        """ PIL uses RGB
            OpenCV uses BGR
            MediaPipe expects RGB from OpenCV
        """
        #Before detection, the image is converted
        rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        with mp_face.FaceDetection(0.5) as fd:
           # MediaPipe expects RGB, so internally:
            result = fd.process(rgb)
            faces = result.detections or []
            return len(faces)

    def analyze_head_pose(self, img: Image.Image):
        rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w, _ = rgb.shape
        """static_image_mode=True → process single image (not video stream)
            max_num_faces=1 → we only care about the main user
            refine_landmarks=True → more accurate eyes/nose
        """
        with mp_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True
        ) as fm:
            res = fm.process(rgb)
            if not res.multi_face_landmarks:
                return (0.0, 0.0)

            lm = res.multi_face_landmarks[0].landmark
            def xy(i): return (lm[i].x * w, lm[i].y * h)

            """MediaPipe gives 468 landmarks.
            You extract:
            1 → nose tip
            33 → left eye
            263 → right eye"""

            nose = xy(1)
            left_eye = xy(33)
            right_eye = xy(263)

            """Distance between eyes → used as a scale reference
            (so head pose works even if the face is close/far)"""
            eye_dist = abs(left_eye[0] - right_eye[0]) + 1e-6

            """If nose moves left of center → user looks left
            If nose moves right → user looks right
            Multiply by 50 to scale numbers into a human-readable ±30 range."""
            yaw = ((nose[0] - ((left_eye[0] + right_eye[0]) / 2)) / eye_dist) * 50

            """If nose moves down → user is looking down
            If nose moves up → user is looking up
            Same scale factor."""
            pitch = ((nose[1] - ((left_eye[1] + right_eye[1]) / 2)) / eye_dist) * 50

            return float(yaw), float(pitch)


"""Stage	            Format	                Notes
Frontend Camera	        raw pixel	            video frame
→ Canvas	            RGB	                    browser RGB pixel data
→ Blob (file mode)	    JPEG binary	            compressed
→ Base64 (b64 mode)	    Base64 JPEG text	    encoded string
Django receives	        file or text	        depends on mode
Django → FastAPI	    binary file OR base64	forwarded
FastAPI → PIL	        PIL RGB	                unified representation
PIL → NumPy (OpenCV)	BGR	                    OpenCV standard
NumPy → MediaPipe	    RGB (interpreted)	    expected by MediaPipe"""