# .pth (PyTorch) and .onnx (Open Neural Network Exchange):
- In the world of AI models, these are simply two different formats of the same "brain":
### .pth (PyTorch):
- This is the "raw" format used during development. To use it, you would need to install the full PyTorch library (which is very large, usually >1GB) and have the original Python class code that defines the model's architecture.
### .onnx (Open Neural Network Exchange):
- This is the "distributed" format. It contains both the architecture and the weights in one file. It is designed to be high-performance, lightweight, and does not require PyTorch.

#### CDCN++ (Central Difference Convolutional Network) (Best Accuracy)
#### Neural Architecture Search (NAS) based backbone with Central Difference Convolutions (CDC).

https://github.com/SeuTao/FaceBagNet
https://github.com/yakhyo/face-anti-spoofing


Phase 1: Frontend Acquisition (The "Smart Capture")
Localized Face Tracking: The system first confirms a face is present in the frame before sending any data.
Reasoning: Reduces server load by preventing the processing of empty frames.
Normalized 1.2x Square Cropping: The face is cropped using a fixed 1.2x scale to create a "Face Chip."
Reasoning: FaceBagNet is a texture-specialist. This tight crop ensures the highest pixel density per square inch of skin, which is vital for detecting micro-artifacts.
Binary Stream Transmission: Data is converted from a standard image string into a raw binary ArrayBuffer.
Reasoning: Improves performance by reducing data size by 33% and lowering latency for real-time feedback.

Phase 2: Backend Sanitization (The "Quality Gate")
Hardware-Accelerated Decoding: Raw bytes are decoded directly into a numerical matrix (NumPy) for the engine.
Reasoning: Eliminates the "decoding overhead" found in traditional web applications, allowing for higher frame-per-second (FPS) analysis.
Automated Image Quality Audit (IQA): The backend runs a "Sharpness & Exposure" check using Laplacian Variance.
Reasoning: Prevents "False Fakes." If an image is blurry or overexposed, the system rejects it as "Unstable" rather than "Fake," prompting the user to fix their lighting or hold still.

Phase 3: Texture Analysis (The "Deep Engine")
Neural Standardization: Images are resized to 128x128 and normalized using a Mean-Shift algorithm (x - 127.5 / 128).
Reasoning: Aligns the live camera data with the specific mathematical distribution of the model’s training set, ensuring the detector "sees" the same texture depth it learned during development.
Patch-Based Feature Extraction: The FaceBagNet model analyzes the face in small, overlapping patches.
Reasoning: By looking at local textures (Bag-of-Features) rather than the whole face, the model can ignore the person's identity and focus purely on finding spoofing patterns like Moiré mesh or printed ink artifacts.

Phase 4: Decision Intelligence (The "Final Verdict")
Temporal Result Smoothing: The system uses a Moving Average Buffer to analyze the last 5 consecutive frames.
Reasoning: Eliminates "flickering" results. It ensures a single frame of bad lighting doesn't cause a rejection, requiring consistent proof of liveness over time.
Multi-Tiered Classification:
Real (>0.94): Confirmed liveness with high-fidelity skin texture.
Suspicious (0.88 - 0.94): Validates the face but flags environmental interference (low quality/grain).
Fake (<0.88): Identifies non-human structural patterns.
Reasoning: Provides a "Safety Gap" that allows the system to ask for a retry instead of a hard lockout, significantly improving the User Experience (UX).