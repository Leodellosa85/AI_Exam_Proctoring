Option	                                    Simple Explanation
num_faces=1	                                Detect only ONE face.
min_face_detection_confidence=0.4	        Ignore very weak detections — must be at least 40% confident a face is present.
min_face_presence_confidence=0.4	        Avoid false positives (fake faces).
min_tracking_confidence=0.4	                Tracks the face between frames (for video).
output_face_blendshapes=False	            Don’t calculate emotions (like smiling, blinking).
output_facial_transformation_matrixes=True	YES → Give the 3D face orientation matrix (used for yaw/pitch/roll).
running_mode=IMAGE	                        Process one image at a time (not webcam video).

1️⃣ Base options
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
- What it does:
    - Tells MediaPipe which pre-trained model to use.
    - Here, face_landmarker.task is a frozen MediaPipe face landmark model.
- Why you set it:
- You are using MediaPipe’s validated model for face detection and head pose.
- No training required; weights are already included in .task.
- Sample:
    - This model detects landmarks and computes head orientation.
    - Can handle 1–5 faces in a frame, depending on your settings.

2️⃣ Number of faces
num_faces=1
- What it does:
    - Limits detection to only the first face in the image.
- Why you set it:
    - For exam proctoring, only the examinee’s face matters.
    - Speeds up processing by ignoring other faces in background.
- Sample output:
    - len(result.face_landmarks) = 1
    - Only the main face landmarks are returned.

3️⃣ Minimum face detection confidence
min_face_detection_confidence=0.4
- What it does:
    - Threshold for the initial face detection.
    - 0.4 = fairly permissive; any detection > 40% confidence is considered valid.
- Why you set it low:
    - Webcam images can be blurry or poorly lit.
    - A low threshold reduces false negatives, ensuring the model tries to detect even partially visible faces.
- Sample output:
    - result.face_landmarks exists even if face is slightly occluded.

4️⃣ Minimum face presence confidence
min_face_presence_confidence=0.4
- What it does:
    - After initial detection, MediaPipe checks how confident it is that a detected face is still present.
- Why you set it low:
    - Exam conditions may include head movements or partial occlusion, so we want tolerant detection.
- Sample output:
    - face_present = 1 if confidence ≥ 0.4
- Can handle faces turning slightly away.

5️⃣ Minimum tracking confidence
min_tracking_confidence=0.4
- What it does:
    - Confidence threshold for landmark tracking between frames.
    - Applies mostly in VIDEO or STREAM mode, but also affects robustness in IMAGE mode for repeated frames.
- Why you set it:
    - Helps reduce false landmark jitter on low-quality webcam input.
- Sample output:
    - Landmarks are reasonably stable across frames even in noisy conditions.

6️⃣ Output face blendshapes
output_face_blendshapes=False
- What it does:
    - Blendshapes are facial expression coefficients (e.g., smile, frown, blink).
- Why you set it off:
    - Your behavioral model only uses head pose and face presence, not expression.
    - Saves memory and computation.
- Sample output:
    - result.face_blendshapes = None

7️⃣ Output facial transformation matrixes
output_facial_transformation_matrixes=True
- What it does:
    - Returns a 4×4 transformation matrix mapping face coordinates → camera coordinates.
- Why you set it on:
    - You need this to compute yaw, pitch, roll for head pose.
- Sample output:
transform_mat.shape = (4,4)
R = transform_mat[:3,:3]  # 3x3 rotation
yaw, pitch, roll = rotation_matrix_to_euler_angles(R)

8️⃣ Running mode
running_mode=mp.tasks.vision.RunningMode.IMAGE
- What it does:
    - Indicates MediaPipe processes each image independently.
- Why you set it:
    - You are processing saved images, not live video frames yet.
    - IMAGE mode is simpler, no internal frame tracking needed.
- Sample:
    - Each call to landmarker.detect(img) → returns landmarks for that single image.

9️⃣ Create the landmarker
landmarker = vision.FaceLandmarker.create_from_options(options)
- What it does:
    - Instantiates the MediaPipe Face Landmarker with all your settings.
Sample usage:
result = landmarker.detect(mp_image)
print(len(result.face_landmarks))  # should be 0 or 1
print(result.facial_transformation_matrixes[0])  # 4x4 matrix