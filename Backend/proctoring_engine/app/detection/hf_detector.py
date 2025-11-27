from PIL import Image
import torch
from transformers import AutoModelForObjectDetection, AutoProcessor
from ..config.settings import HF_MODEL_REPO_ID

class HFDetector:
    """
    Hugging Face-based face detector for AI Proctoring POC.

    Responsibilities:
    - Detect faces in a single frame.
    - Count faces → supports Face Presence & Multi-Person Detection.
    """

    def __init__(self):
        if HF_MODEL_REPO_ID is None:
            self.enabled = False
            print("HFDetector: No model configured, disabled.")
            return

        try:
            # Load processor & model
            self.processor = AutoProcessor.from_pretrained(HF_MODEL_REPO_ID)
            self.model = AutoModelForObjectDetection.from_pretrained(HF_MODEL_REPO_ID)
            self.model.eval()
            self.enabled = True
            print(f"HFDetector: Loaded model {HF_MODEL_REPO_ID}")
        except Exception as e:
            print(f"HFDetector: Failed to load model {HF_MODEL_REPO_ID}: {e}")
            self.enabled = False

    def detect_faces(self, img: Image.Image) -> int:
        """
        Detect faces in a PIL image.

        Returns:
            int: Number of faces detected. Returns -1 if detector disabled.
        """
        if not self.enabled:
            return -1

        # Convert image & run through HF object detection model
        inputs = self.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Model outputs vary; this is a generic way to get boxes + scores
        # We assume class 1 = face (depends on model!)
        # Adjust threshold as needed
        threshold = 0.5

        # Get logits/probabilities
        if hasattr(outputs, "logits"):
            # Some HF models return logits of shape [batch, num_queries, num_classes]
            scores = outputs.logits.softmax(-1)[..., :-1].max(-1).values  # exclude "no-object" class
            num_faces = int((scores > threshold).sum().item())
        elif hasattr(outputs, "scores"):
            # Some models return 'scores' directly
            num_faces = int((outputs.scores > threshold).sum().item())
        else:
            # fallback
            num_faces = 0

        return num_faces
