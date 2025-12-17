Who develops MediaPipe?
- Google / Google Research
- MediaPipe is:
    - Open-source
    - Production-tested in Google products
    - Used in:
        - ARCore
        - Google Meet
        - YouTube effects
        - Pixel camera features
Validation (what this actually means)
- MediaPipe Face Landmarker was validated internally by Google using:
- Large-scale multi-demographic datasets
- Controlled head pose benchmarks
- Internal QA pipelines
While exact datasets are not public, the model is:
✔ Peer-reviewed in concept
✔ Widely deployed
✔ Maintained by Google

References you can cite (important)
- Official documentation
MediaPipe Face Landmarker:
https://developers.google.com/mediapipe/solutions/vision/face_landmarker
- Academic foundation (head pose & landmarks)
    - Kazemi & Sullivan, One Millisecond Face Alignment with an Ensemble of Regression Trees
    - Bulat & Tzimiropoulos, How Far Are We from Solving Face Alignment?
- Weak supervision precedent
- Ratner et al., Snorkel: Rapid Training Data Creation with Weak Supervision

📌 Important wording
“MediaPipe is used as a standardized, well-maintained facial geometry estimator, not as a ground-truth oracle.”


With MediaPipe Face Landmarker Task API, result can contain:
✅ Available result fields
- Depending on how the task was configured:
1. result.face_landmarks
    - Type: List[List[NormalizedLandmark]]
    - What you already use
    - 468 landmarks per face (x, y, z in normalized coordinates)
2. result.facial_transformation_matrixes
    - Type: List[List[float]]
    - 4×4 transformation matrix per face
    - Used for head pose (pitch, yaw, roll) ✅
    - You are using this correctly
3. result.face_blendshapes (optional, if enabled in options)
    - Type: List[FaceBlendshapes]
    - Expression-related scores (mouth open, smile, blink, etc.)
    - Each blendshape has a .score ∈ [0, 1]
4. result.face_geometry (less commonly used)
    - 3D mesh geometry info

Can you get a confidence score?
⚠️ Important truth:
- There is NO explicit “face detection confidence” score in FaceLandmarker.
- Unlike object detectors, Face Landmarker:
    - Either detects a face or doesn’t
    - It does not return a single detection probability

✅ What you can use as a proxy confidence
Option 1: Face presence (binary confidence)
face_detected = len(result.face_landmarks) > 0
This is effectively:
- 1.0 → face detected
- 0.0 → no face

Can you get F1 score from this code?
❌ No — and here’s why
- F1 score is NOT a per-frame output.
- F1 score requires:
    - Ground truth labels
    - Multiple predictions
    - Dataset-level evaluation
    - F1 formula:
        - F1 = 2 * (precision * recall) / (precision + recall)
- You CANNOT compute F1:
    - From a single image
    - From MediaPipe output alone
    - Without ground-truth annotations

When does F1 score make sense here?
- You can compute F1 score if:
- Example tasks:
    - “Is the head turned left/right/center?”
    - “Is the face present or not?”
- Example:
    - Ground truth: head turned left
    - Prediction: head turned left
- Across a dataset:
from sklearn.metrics import f1_score
f1_score(y_true, y_pred)

➡️ This is model evaluation, not inference output.

MediaPipe Face Landmarker is NOT meant to be “trained” further.
It is a feature extractor.

What you train is your own AI on top of MediaPipe outputs.
So your goal should be:
- Use MediaPipe to generate ground-truth–like signals → train a proctoring classifier/regressor on those signals.
- This is exactly how production proctoring systems work.

You can CREATE your own ground truth using MediaPipe
Why this is valid
- MediaPipe head pose is already trained and validated
- You use it as a pseudo–ground truth generator
- Then train a lighter, task-specific model
- This is called:
    - Teacher–Student / Weak Supervision

FPS = Frames Per Second
0.5 seconds = 2 FPS
- Good enough for head pose & face presence
- Low bandwidth
- Lower privacy risk than video

Why FPS matters
FPS	    Meaning
30 FPS	Video (overkill, high privacy risk)
10 FPS	High-resolution tracking
2 FPS	Behavior monitoring (perfect for proctoring)
<1 FPS	Too sparse

Bias when using MediaPipe as a truth generator (real risks)
MediaPipe can introduce bias in three main ways:
1. Demographic bias
- Skin tone
- Facial structure
- Glasses / hijab / masks
- Lighting differences
➡️ Result:
- Face presence false negatives
- Noisy landmarks
- Inaccurate pose
2. Hardware & environment bias
- Low-quality webcams
- Poor lighting
- Motion blur
- Laptop angle
➡️ Result:
- Higher face loss for certain users
3. Behavioral bias
- Cultural norms (eye contact styles)
- Disabilities
- Neurodivergence
➡️ Result:
- False “suspicious” flags

How to MITIGATE bias (this is key)
Rule #1: MediaPipe never makes the final decision
MediaPipe only produces:
- low-level geometric signals
- Your system decides behavior, not MediaPipe.
1. Use MediaPipe as a sensor, not a judge
What MediaPipe gives:
- yaw, pitch, roll
- face_present (binary)
What YOU decide:
- Is this behavior suspicious over time?
This separation is critical for defense.

2. Aggregate over time (this reduces bias massively)
Instead of:
- One bad frame = violation ❌
Use:
- face_presence_ratio = frames_with_face / total_frames
- avg_yaw = mean(yaw)
- yaw_std = std(yaw)
Example:
- Face missing for 1 frame → ignore
- Face missing for 10 seconds → flag
✔️ Reduces false positives
✔️ Defensible statistically

3. Use relative thresholds, not absolute ones
BAD (biased):
- if yaw > 30: suspicious
GOOD:
- baseline_yaw = median(yaw_first_30s)
- if abs(yaw - baseline_yaw) > threshold:
    suspicious
✔️ Adapts to:
- Camera angle
- User posture
- Physical differences

4. Calibration phase (VERY IMPORTANT)
Add a 30–60 second calibration step before exam:
- "Please look at the screen normally"
Store:
- baseline_pitch, baseline_yaw
- baseline_variance
Then judge deviation, not absolute pose.
This is one of your strongest bias defenses.

5. Multi-signal confirmation
Never rely on just head pose.
Combine:
- Face presence
- Pose stability
- Temporal duration
Optional: gaze proxy (eye landmarks only)
Flag only when multiple signals agree.

D. How to defend using MediaPipe as a “truth generator”
- This is how you justify it formally and ethically.
1. Correct terminology (IMPORTANT)
❌ Do NOT say:
“MediaPipe gives ground truth”
✅ Say:
“MediaPipe provides pseudo-labels or weak supervision signals”
This matters a lot.

“Due to the lack of publicly available datasets with accurate yaw, pitch, and roll annotations under real exam conditions, we use MediaPipe Face Landmarker as a weak supervision signal to generate pseudo-labels. These labels are not treated as absolute ground truth but are aggregated temporally and validated against human-reviewed behavioral annotations.”

Bias defense statement (very important)
You can explicitly state:

“To mitigate demographic and environmental bias, decisions are never made at the single-frame level. All judgments are based on relative deviations from a user-specific calibration baseline and require consistent behavior over a temporal window.”

This shows intentional bias mitigation.

Bottom line (this is your defense in one paragraph)

MediaPipe is not used as an authority but as a standardized facial geometry extractor. Bias is mitigated through user-specific calibration, temporal aggregation, and behavior-level decision making rather than frame-level classification. The system prioritizes fairness, explainability, and human oversight over automated judgment.

Calibration phase (first 30–60 seconds)
User instruction:
“Please sit naturally and look at your screen.”
Collect:
yaw_list, pitch_list, roll_list

Compute baseline:
baseline = {
    "yaw_mean": np.median(yaw_list),
    "yaw_std": np.std(yaw_list),
    "pitch_mean": np.median(pitch_list),
    "pitch_std": np.std(pitch_list)
}
Store per session.
Runtime deviation scoring
For each window:
yaw_dev = abs(mean_yaw - baseline["yaw_mean"])
pitch_dev = abs(mean_pitch - baseline["pitch_mean"])

Normalize:
yaw_score = yaw_dev / (baseline["yaw_std"] + eps)

This removes:
- Camera angle bias
- Posture bias
- Facial structure bias

Decision logic (example)
if yaw_score > 3 and duration > 8s:
    flag = "looking_away"

This is auditable and explainable.

“Pseudo-labels” / “Weak supervision” explained
What is ground truth?
Ground truth = perfect, manually verified labels
You do not have:
- Exact yaw/pitch/roll annotations
- Motion capture systems

What MediaPipe gives you
MediaPipe provides:
- Approximate but consistent pose estimates
- These are:
pseudo-labels

Temporal aggregation (5–10 second windows)
What this means
- Instead of:
Frame → decision ❌
- You do:
5–10 seconds of frames → decision ✅


A. MediaPipe the Python library
- import mediapipe as mp
This is:
- Just a runtime framework
- Graph execution + utilities
- NOT the AI model itself
Think of it as:
- TensorFlow Lite runtime + helpers

B. MediaPipe .task file (THIS is the AI model)
- face_landmarker.task
This contains:
- Neural network weights
- Architecture
- Pretrained parameters
This file is:
- The actual AI model
- Trained by Google
- Frozen (not finetunable)
So when you say:
“We are using MediaPipe”
You should clarify:
“We are using MediaPipe runtime with a pretrained face landmark model.”

- Why this distinction matters
Because your system actually has:
- [Pretrained face model]  ← frozen
- [Your trained behavioral model] ← trainable
That fully satisfies:
- “We use AI”
- “We train a model”
- “We control behavior logic”

“MediaPipe provides a pretrained facial geometry model that extracts head pose and face presence. This model is not fine-tuned, by design, to avoid bias and instability. On top of these features, we train our own AI model that learns exam behavior patterns from our dataset. This second model is fully trainable, auditable, and can be sourced from standard ML libraries or Hugging Face.”

What “Train your own behavioral model” actually means
It does NOT mean:
- Training a face detector
- Training a landmark model
- Training head-pose estimation from pixels
Those are low-level perception problems.

It DOES mean:
- You train a model that answers questions like:
- “Based on face presence and head pose over time, is the exam behavior acceptable?”
- This model is:
    - Task-specific (exam proctoring)
    - Fully trainable
    - Uses your dataset
    - Explainable

Example (very concrete)
- Input to your trained model
- Not images — features:
    - mean_yaw_5s
    - std_yaw_5s
    - mean_pitch_5s
    - face_presence_ratio_5s
- Output
0 = normal
1 = suspicious
2 = violation
That is a real AI model, trained on your data


What is a “behavioral model”?
- A behavioral model does NOT look at faces directly.
- Instead, it answers questions like:
“Given head pose and face presence over time, is the behavior normal or suspicious?”
- Input to a behavioral model
- Numerical features (from MediaPipe):
yaw, pitch, roll
face_present
face_presence_ratio_5s
yaw_std_5s
pitch_std_5s

- Output
0 = normal
1 = suspicious
2 = violation

Behavioral models you can use (RANKED)
⭐ OPTION A (BEST): Classical ML (Tree-based)
Models
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)
Why they are ideal

✔ Handle noisy features
✔ No GPU required
✔ Explainable
✔ Easy bias analysis
✔ Accepted in regulated environments

Example use
Input: 5-second aggregated features
Output: suspicious / normal

Recommended architecture
MediaPipe (frozen)
   ↓
Head pose & face presence
   ↓
Temporal aggregation (5–10s)
   ↓
Behavioral model (trainable)
   ↓
Soft flags + audit logs

Bottom line (very clear)
✔ A behavioral model learns patterns over time, not faces
✔ YOLO ≠ behavioral model
✔ MediaPipe + behavioral ML is the industry-standard approach
✔ Hugging Face can be used at the behavior level, not face level

NEXT STEP 1: Temporal aggregation (MOST IMPORTANT STEP)
- Why this step exists
- Single frames are noisy and unfair:
    - Person blinks
    - Slight head movement
    - Camera jitter
- Proctoring must be time-based, not frame-based.

What “Temporal aggregation (5–10s)” means
- You group multiple frames into a time window.
Example
- Capture rate: 0.5s → 2 FPS
- 5 seconds → 10 frames
- 10 seconds → 20 frames
You compute statistics over the window, not per frame.

Features you compute per window
1. Face presence
face_presence_ratio = (# frames with face) / (total frames)

2. Head direction
yaw_mean
yaw_std
pitch_mean
pitch_std

3. Optional stability features
yaw_range = max(yaw) - min(yaw)
pitch_range = max(pitch) - min(pitch)

Why this matters for exams
Behavior	    Temporal signal
Quick glance	low std, short duration
Looking away	high mean yaw
Leaving seat	low face_presence_ratio

Example code (aggregation)
def aggregate_window(df):
    return {
        "face_presence_ratio": df["face_present"].mean(),
        "yaw_mean": df["yaw"].mean(),
        "yaw_std": df["yaw"].std(),
        "pitch_mean": df["pitch"].mean(),
        "pitch_std": df["pitch"].std(),
    }

Each output row = 1 behavioral sample

NEXT STEP 2: Define behavioral labels (ground truth)
- This is where YOU define behavior
- You are not labeling faces — you are labeling behavior.
Example labels
Label	Meaning
0	    Normal
1	    Suspicious
2	    Violation

Example rules for initial labeling (bootstrapping)
- You can start with rules, then refine with human review.
IF face_presence_ratio < 0.6 → violation
IF |yaw_mean| > 30° for >5s → suspicious
ELSE → normal

- These labels become training data.
- This is called weak supervision.

NEXT STEP 3: Train the behavioral model (THIS IS YOUR AI)
- This is the trainable model your manager wants.
- Input to the model
face_presence_ratio
yaw_mean
yaw_std
pitch_mean
pitch_std

- Output
behavior_label

NEXT STEP 4: Soft flags (NOT hard decisions)
- Why soft flags
- You never auto-fail a student based on AI.
- Instead:
Generate scores
Log evidence

NEXT STEP 5: Audit logs (CRITICAL for production)
- Every decision must be traceable.
- Store:
Aggregated features
Model prediction
Threshold used
Timestamp


1️⃣ WINDOW AGGREGATION (FROM YOUR CSV)
- Assumptions about your per-frame CSV
- Your existing CSV has rows like:
session_id, timestamp, face_present, yaw, pitch, roll
- Captured every 0.5s (2 FPS)

2️⃣ LABELING RULES (WEAK SUPERVISION)
- You do not label frames.
- You label behavior windows.
Behavioral definitions (VERY IMPORTANT)
Label	Meaning
0	    Normal
1	    Suspicious
2	    Violation

3️⃣ TRAINING THE BEHAVIORAL MODEL
This is the AI your manager wants.

4️⃣ EVALUATION METRICS (WHAT YOU REPORT)
- What NOT to overemphasize
- Accuracy (misleading)

What to report (DEFENSIBLE)
Metric	            Why
Precision	        Avoid false accusations
Recall	            Catch real violations
F1-score	        Balance
Confusion matrix	Transparency

Key statement you can use:
“We prioritize precision over recall to minimize false accusations, consistent with proctoring ethics.”

5️⃣ THRESHOLD DESIGN (THIS IS CRITICAL)
- Your model outputs probabilities, not final decisions.
proba = model.predict_proba(X)[i]
if proba[2] > 0.8:
    flag = "violation"
elif proba[1] > 0.7:
    flag = "suspicious"
else:
    flag = "normal"

Why these thresholds are defensible
- Academic references you CAN cite:
1️⃣ ISO/IEC 23894:2023 (AI Risk Management)
- Recommends conservative thresholds in human-impact systems
2️⃣ IEEE Ethically Aligned Design
- Avoid high-confidence automated sanctions
3️⃣ Buolamwini & Gebru (2018)
- Higher thresholds reduce demographic bias impact
4️⃣ Microsoft Responsible AI Guidelines
- Human-in-the-loop decisioning

Exact defense sentence (USE THIS):
“Thresholds were selected conservatively to prioritize precision and reduce false positives. Automated outputs generate soft flags for human review rather than hard enforcement.”

6️⃣ AUDIT LOG DESIGN (MANDATORY)
- Every window creates a log
{
  "session_id": "xyz",
  "window": "10:05:00–10:05:05",
  "features": {
    "yaw_mean": 35.1,
    "face_presence_ratio": 0.95
  },
  "model_output": "suspicious",
  "confidence": 0.78,
  "action": "flag_for_review"
}



1️⃣ What “real-time” means in exam proctoring (important)
- Real-time does NOT mean:
    - Frame-by-frame punishment
    - Instant warnings on a single glance
- Real-time DOES mean:
    - Continuous monitoring
    - Decisions every few seconds
    - Timely alerts with low false positives

💡 In regulated systems, 5–10 seconds latency is intentional.

3️⃣ How real-time processing works (conceptually)
Sliding window approach
- Capture frame every 0.5s
- Keep the last N frames in memory
- Every new frame:
    - Update the window
    - Recompute features
    - Re-run the behavioral model

This is called online inference.

4️⃣ Real-time timeline example
Time	What happens
0.0s	First frame
0.5s	Second frame
...	...
5.0s	First window complete → inference
5.5s	Window slides → inference
6.0s	Window slides → inference

So after the first 5 seconds, the system updates every 0.5 seconds.
That is real-time.

6️⃣ Real-time response strategy (VERY IMPORTANT)
❌ What NOT to do
- Show raw yaw/pitch
- Immediate punishment
- Per-frame alerts

✅ What TO do
- Soft real-time feedback options
- Internal flag (backend only)
- Proctor dashboard highlight
- Silent logging
- Optional gentle reminder

Example policy
IF suspicious for 3 consecutive windows
→ notify proctor
This reduces false positives dramatically.

7️⃣ Latency analysis (this will help in defense)
Component	            Time
MediaPipe inference	    ~5–10 ms
Feature aggregation	    ~1 ms
Behavioral model	    <1 ms
Total	                ~15 ms

The only delay is the intentional 5s window.

8️⃣ Why real-time + windows is REQUIRED (defense point)
You can say:
“Real-time decisions are based on short temporal windows to ensure stability and fairness. Frame-level reactions are avoided to reduce false positives and bias.”

This aligns with:
- IEEE standards
- Online behavior analysis literature
- Commercial proctoring systems



1️⃣ What cross_val_score already does
- This block:
scores = cross_val_score(
    model,
    X,
    y,
    cv=3,
    scoring="f1_weighted"
)
means:
- The data is split into 3 folds
- Each fold is used once as validation
- The model is trained 3 separate times
- You already get out-of-sample F1 scores
So you already have:
- Validation
- Generalization estimate
- A defensible metric
- There is no X_test and y_test here.

2️⃣ Why your second block does NOT apply
- This part:
y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred)

requires:
- A single held-out test set
- A model trained on X_train
- But when you use cross-validation:
- The model is trained internally multiple times
- There is no single test set
- model at the end is not trained yet
- So using both would be conceptually wrong.






