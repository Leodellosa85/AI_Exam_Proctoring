Pitch: 14.38°
Yaw:   -50.44°
Roll:  6.35°

These three values represent the orientation of the person's head in 3D, measured in DEGREES.

1. Yaw — Turning Left / Right
- Yaw means rotating your head left or right, like saying “NO”.
- Positive yaw → person looking right
- Negative yaw → person looking left

In your output:
❗ Yaw = -50° → The person is looking strongly to the LEFT

2. Pitch — Looking Up / Down
- Pitch means rotating your head up or down, like saying “YES”.
- Positive pitch → looking up
- Negative pitch → looking down

In your output:
👉 Pitch = +14° → Slightly looking UP

3. Roll — Tilting the head sideways
- Roll means tilting your head to the side, like when you're curious.
- Positive roll → tilting right
- Negative roll → tilting left

In your output:
👉 Roll = +6° → Slight tilt to the RIGHT

pitch = float(pose[0]) * 180/np.pi
yaw   = float(pose[1]) * 180/np.pi
roll  = float(pose[2]) * 180/np.pi

This is simply converting radians to degrees.
Why?
- The .mat file stores angles in radians
- but humans understand degrees, so we convert.
- Radians → Degrees
Formula:
degrees = radians × (180 / π)
1 rad = 180 degrees
2 rad = 369 degrees
radians = degrees × (π/180)
degrees = radians × (180/π)


Example:

1 radian ≈ 57.2958 degrees

So:
yaw_deg = yaw_radians * (180 / 3.14159)

A radian measures how far you move along the edge of a circle, using the circle’s radius as the unit of distance.

SolvePnP (Perspective-n-Point)
OpenCV solvePnP() takes:


You need a “face presence + movement + angle” threshold system that considers environmental factors like camera quality and lighting. Here’s a structured approach:

1️⃣ Face Presence Threshold
- Even if a model detects a face, in real-world scenarios you might need confidence thresholds:
- Confidence Score: Most face detectors (YOLO, MediaPipe, RetinaFace) output a probability/confidence.
- Example: confidence = 0.85 → face is detected with 85% certainty.
- Threshold: Set a lower limit to reduce false positives.
- Recommended: >= 0.7 for typical webcams.
- For low-quality cameras or poor lighting, you might lower to 0.6 but combine with other checks.
- Bounding Box Size: Reject detections that are too small (e.g., < 50px in height), as they may be false positives or far away faces.

2️⃣ Head Movement Thresholds
- Head movement can indicate attention or suspicious behavior. Use yaw/pitch/roll from head pose estimation:
Parameter	        Typical Range	        Suggested Thresholds for Alert
Yaw (left-right)	-90° → 90°	            Alert if > ±45° for > 3 sec
Pitch (up-down)	    -90° → 90°	            Alert if > ±30° for > 3 sec
Roll (tilt)	        -45° → 45°	            Usually ignore unless extreme tilt

You can smooth the angles over time (moving average over 0.5–1 sec) to avoid false alerts due to small jitter.

3️⃣ Face Movement Detection
- Track the face across frames: compare bounding box positions or landmark positions.
- Thresholds for movement:
- Minimal pixel displacement → stable attention
- Large sudden displacement → attention diverted / suspicious
- Consider frame rate: If your webcam is 15–30 fps, movement thresholds must be scaled per frame.

4️⃣ Environmental Considerations
- Camera Quality
    - Low-res cameras: set lower confidence thresholds for detection.
    - Avoid tiny bounding boxes (<50px).
- Lighting Conditions
    - Poor lighting can reduce detection confidence.
    - You can use brightness/contrast metrics:
        - If frame brightness < 50 (on 0–255 scale), lower thresholds or issue warning.
- Occlusions
    - Glasses, masks, hair can reduce landmark detection.
    - Consider fallback detection using simpler face bounding box (YOLO) instead of landmarks.

- Distance / Zoom
    - If the face occupies <10–15% of the frame, detection may be unreliable.

5️⃣ Combining Thresholds for Decision Logic
Example pseudo-code:
if face_detected and confidence >= 0.7 and bbox_height >= 50:
    # check pose
    if abs(yaw) > 45 or abs(pitch) > 30:
        alert("Look away too long")
    elif face_movement > movement_threshold:
        alert("Suspicious movement")
else:
    alert("No face detected")

- You can add time windows (e.g., 3–5 seconds) to avoid false positives for temporary occlusions.

6️⃣ Optional: Dynamic Threshold Adjustment
- Adjust thresholds dynamically based on camera quality and environment lighting.
For example:
Low light → reduce pose thresholds
High light, high-res → stricter thresholds

7️⃣ Data-Driven Approach
- Test your thresholds using sample exam videos.
- Measure:
    - Detection confidence
    - Angle variation
    - Frame-to-frame movement
- Adjust thresholds to balance:
    - False positives (raising unnecessary alerts)
    - False negatives (missing real suspicious behavior)


✅ Summary

Feature	        Metric	                        Suggested Threshold
Face presence	    Detection confidence	    ≥ 0.7
Face size	        Bounding box height	        ≥ 50 px
Head yaw	        Left/right	                ±45° max for normal attention
Head pitch	        Up/down	                    ±30° max for normal attention
Roll	            Tilt	                    ±30° optional
Face movement	    Pixel displacement/frame	Small jitter ignored, large > threshold alerts
Environment	        Brightness	                Warn if too dark (<50)
Occlusion / blur	Optional	                Lower confidence → manual check


1️⃣ What AFLW2000-3D Provides
- Images: 2,000 face images from AFLW dataset
- Labels / Annotations:
- 3D landmarks (68+ points per face)
- Head pose: yaw, pitch, roll in radians/degrees
- Use case: Head pose estimation, face alignment, and landmark-based applications

✅ It already provides:
- Head pose (yaw/pitch/roll) → you can compute face orientation and angles
- Landmark locations → you can draw axes on face, measure movement

❌ It does NOT provide:
- Face presence / no-face labels → all images have faces
- Environmental labels (lighting, occlusion, camera quality)
- Face movement across frames → it’s a static image dataset

2️⃣ What You Can Use Directly from AFLW2000-3D
- Head pose estimation: you can train a regression model to predict yaw/pitch/roll
- Angle-based thresholds: for “LOOK LEFT/RIGHT/UP/DOWN/CENTER”
- Axes visualization: draw 3D axes on the face for testing / reporting
- Essentially, AFLW2000-3D solves the head pose part for static images.

3️⃣ What You Still Need to Add / Handle Yourself
- Face presence detection
    - AFLW2000-3D only has faces; your model will never learn “no-face” cases.
    - You need a face/no-face dataset like:
        - WIDER FACE
        - FDDB
        - LFW + random non-face images
- Face movement detection
    - AFLW2000-3D has only single images → no temporal movement info
    - For tracking movement, you need:
        - Webcam frames or video sequences
        - Or a dataset with multiple frames per subject
- Environmental factors
    - Lighting, camera quality, occlusions → must simulate or collect in your own environment

4️⃣ Practical Strategy for Your POC
Module	                    Dataset to Use	                    Notes
Head pose estimation	    AFLW2000-3D	                        Already labeled, good for initial training
Face presence detection	    WIDER FACE / LFW + no-face images	Needed for detecting “no face present”
Movement tracking	        Webcam / video frames	            Calculate displacement between frames using landmarks or bounding boxes
Thresholds / lighting	    Custom data	                        Optional: collect sample exam videos to tune thresholds


Each augmented image is now saved to its respective folder:

landmarks_axes/ - Images with face landmarks and pose angles drawn
gamma/ - Brightness-adjusted images (3 per original)
occlusion/ - Images with random rectangular blocks (3 per original)
lowres/ - Low-resolution simulations (2 per original)
noise/ - Gaussian noise added images (3 per original)
combined/ - Multiple augmentations applied together (3 per original)


You're using MediaPipe — a Google library that detects faces and landmarks using AI models.
This line downloads a file named face_landmarker.task, which is basically the AI model MediaPipe uses to detect the face and compute its orientation (yaw, pitch, roll).

Option	                                    Simple Explanation
num_faces=1	                                Detect only ONE face.
min_face_detection_confidence=0.4	        Ignore very weak detections — must be at least 40% confident a face is present.
min_face_presence_confidence=0.4	        Avoid false positives (fake faces).
min_tracking_confidence=0.4	                Tracks the face between frames (for video).
output_face_blendshapes=False	            Don’t calculate emotions (like smiling, blinking).
output_facial_transformation_matrixes=True	YES → Give the 3D face orientation matrix (used for yaw/pitch/roll).
running_mode=IMAGE	                        Process one image at a time (not webcam video).

This uses the old MediaPipe Solutions API.
The models are already included inside the mediapipe package, so:

✔ no downloads
✔ easy to use
✔ works for standard face landmarks
✖ has limited features
✖ does NOT give yaw/pitch/roll or the 3D matrix

You are using the Tasks API, which requires a .task file.

👉 The .task file is the actual machine learning model — it must be downloaded manually.

This version supports:

✔ 3D face orientation (yaw, pitch, roll)
✔ transformation matrix
✔ output_face_blendshapes (emotions)
✔ can run on GPU / optimized hardware
✔ newer, better accuracy

🔥 BIGGEST DIFFERENCE (Most important)
Old MediaPipe (no download)
- Good for landmarks (468 points)
- Has 3D-ish data, BUT no official yaw/pitch/roll
- Cannot output transformation matrix
- Much harder to compute head rotation

New MediaPipe Tasks (download .task)
- Model is fully separate and must be provided
- Gives you:
- 3D rotation matrix
- accurate yaw, pitch, roll
- face blendshapes (emotion weights)
- Faster and more stable

def rotation_matrix_to_euler_angles(R):
Takes a 3×3 rotation matrix and converts it into:
- pitch (up/down head tilt)
- yaw (left/right head turn)
- roll (tilt sideways)
MediaPipe gives you a rotation matrix, but you need Euler angles to know head rotation in degrees.

🧠 What is a Rotation Matrix? 
Imagine you point your face forward.
Now you turn your head:
- left or right → yaw
- up or down → pitch
- tilt sideways → roll

These movements are rotations.
MediaPipe gives these rotations in a complicated format called a:
👉 Rotation Matrix (3×3 matrix)
It looks like this:
[ 0.94  -0.10   0.30 ]
[ 0.12   0.99  -0.05 ]
[ -0.29  0.09   0.95 ]

This is a mathematical block of numbers that represents how your head is rotated in 3D.
But…
🚫 Humans don’t understand rotation matrices
🚫 You cannot directly say “how many degrees is this?”
So we need to convert this matrix into something useful.

🎯 What are Euler Angles? (easy version)
Euler angles give rotation in degrees, like:
- Yaw → head turning left/right (°)
- Pitch → head looking up/down (°)
- Roll → head tilting sideways (°)
pitch = 10°   (looking up)
yaw   = -20°  (turning left)
roll  = 5°    (slightly tilted)

R is the rotation matrix that MediaPipe gives you.
[ R00 R01 R02 ]
[ R10 R11 R12 ]
[ R20 R21 R22 ]

Each number tells how a 3D axis is rotated.
- We pick the values that change when:
- head tilts → row 2 columns 1 & 2
- head turns → row 2 column 0
- head rolls → row 0 & 1 column 0

These formulas come from a standard 3D math rule but already simplified for MediaPipe.

R = 3×3 rotation matrix
R[a,b] = number at row a, column b
These numbers represent how your head rotated in 3D
We use specific R cells to compute pitch, yaw, roll

Imagine your head has 3 arrows:
- X arrow: points to your right
- Y arrow: points up
- Z arrow: points forward

When you move your head, these arrows rotate.
- The rotation matrix R stores the new directions of all arrows.
- Your code reads certain numbers from R to compute:
Movement	        Uses which R values?	    Why
Pitch (up/down)	    R[2,1], R[2,2]	            looks at tilting forward/up
Yaw (left/right)	R[2,0]	                    face turning sideways
Roll (tilt)	        R[1,0], R[0,0]	            looks at how head tilts around Z

We plug these numbers into atan2 and asin to calculate pitch, yaw, roll.


✅ Function 1: get_head_pose(img_bgr)
🎯 Goal:
- Take an image → detect face → get:
- pitch, yaw, roll
- face landmarks (points)

transform_mat = np.array(result.facial_transformation_matrixes[0], dtype=np.float32).reshape(4, 4)
Get the 4×4 transformation matrix
[ R | T ]
[ 0 | 1 ]
R = rotation matrix
T = translation (where the head is)

[[ 0.9802   -0.0511    0.1915    0.0123 ]
 [ 0.0580    0.9980   -0.0207   -0.0052 ]
 [-0.1898    0.0316    0.9813    0.4501 ]
 [ 0.0000    0.0000    0.0000    1.0000 ]]

🟥 Rows 0–2, Columns 0–2 → Rotation Matrix (3×3)
[ 0.9802  -0.0511   0.1915 ]
[ 0.0580   0.9980  -0.0207 ]
[-0.1898   0.0316   0.9813 ]

🟦 Column 3 (the last column) → Translation (head position in 3D)
[ 0.0123 ]
[-0.0052 ]
[ 0.4501 ]

This tells where the head is in space (forward/back, left/right, up/down), not the angle.
Most people ignore this unless doing 3D face tracking.

🟩 Last row → always this:
[ 0.  0.  0.  1. ]
This is standard in 4×4 transformation matrices.


# Convert landmarks to image coordinates
# MediaPipe landmarks are 0 to 1, so we scale them.
h, w = img_bgr.shape[:2]
image_landmarks = [(int(p.x * w), int(p.y * h)) for p in landmarks]
# This gives coordinates like (320, 150) that you can draw.

MediaPipe gives you face landmarks like:
- p.x = 0.52
- p.y = 0.23
These values are not pixels.
They are normalized — meaning:
- 0.0 = left/top of the image
- 1.0 = right/bottom of the image
So you need to convert them to real pixel coordinates so that OpenCV can draw them

h, w = img_bgr.shape[:2]
- This gets your image height and width:
- Example:
- If your image is 640×480 → h=480, w=640.

image_landmarks = [(int(p.x * w), int(p.y * h)) for p in landmarks]
For each landmark p, convert:
- pixel_x = p.x * w
- pixel_y = p.y * h

Example:
- Suppose the 0–1 landmark is:
- p.x = 0.50, p.y = 0.25
- On a 640×480 image:
pixel_x = 0.50 * 640 = 320
pixel_y = 0.25 * 480 = 120

So the converted point is:
(320, 120)

🟢 Why this is needed?
- Because OpenCV uses pixels, not normalized values.
- This converted result is what you pass to:
cv2.circle(image, (x, y), 2, (0,255,0), -1)

🟦 What is cv2?
cv2 = OpenCV, the most popular computer-vision library.
You use it to:
- Draw circles
- Draw text
- Show images
- Detect objects
- Process video frames
Basically, it's your main tool for working with images in Python.

cv2.circle(img_copy, (x, y), 1, (0, 255, 0), -1)
This draws a small green dot at the landmark location.
Parameter	        Meaning
img_copy	        The image where you draw
(x, y)	            Pixel position of the dot
1	                Radius of the circle (very small)
(0, 255, 0)	        Color = green (BGR format)
-1	                Filled circle

cv2.putText(img_copy, f"Pitch: {pitch:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
Meaning:
- img_copy → draw text on this image
- "Pitch: {pitch:.1f}" → shows pitch value like:
Pitch: -12.4
- (10, 25) → position of the text on screen
- cv2.FONT_HERSHEY_SIMPLEX → font style
- 0.7 → font size scale
- (0,255,255) → text color = yellow
- 2 → thickness of letters







2️⃣ Comparison with MediaPipe Tasks API
Feature	                            Old Face Mesh + solvePnP	                    MediaPipe Face Landmarker Task API
Requires download .task?	        No	                                            Yes
Gives direct 3D rotation matrix	    No, you compute it with solvePnP	            Yes
Gives yaw/pitch/roll	            Not directly, you compute from rvec	            Directly available
Accuracy	                        Depends on how well 3D model matches the face	More stable + optimized
Extra features	                    Only landmarks	                                Blendshapes, transformations, 3D rotation

PnP (Perspective-n-Point)



🔍 What is gamma correction?
- Gamma correction changes the brightness of an image non-linearly (curved, not straight).
- It is often used to fix:
 - dark faces
 - shadows
- overly bright images
- Instead of simply “adding brightness,” gamma correction adjusts brightness according to how human vision works.

1️⃣ inv_gamma = 1.0 / gamma
- If you choose gamma = 1.5, then:
- inv_gamma = 1 / 1.5 = 0.666...
- This controls how much brightness is changed.
gamma > 1 → brighten image
gamma < 1 → darken image

2️⃣ Build a lookup table (LUT)
- table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
- This builds a list of 256 values (0 to 255), but gamma-corrected.
- Example:
If i = 100 (a pixel brightness):
100/255
raise to power inv_gamma
multiply by 255

- This gives a new brightness value
📌 The LUT is simply a “brightness conversion rule”.

3️⃣ Convert image using the table
cv2.LUT(img, table)
- LUT = Look-Up Table
- This means:
“For every pixel brightness value, replace it with the new gamma-corrected value.”
- It applies the table to the entire image very fast.

📸 What does the output look like?
- If gamma = 1.5 → brighten image
- Before gamma correction:
- Dark face, shadows, low contrast

After gamma correction:
- More visible face, brighter mid-tones
- If gamma = 0.7 → darken image

Before:
- Over-exposed, too bright
After:
- More balanced, darker

🟢 Simple summary
Line of code	        Meaning
inv_gamma = 1/gamma	    Controls how strong effect is
Build table	            Precomputes brightness adjustments
cv2.LUT(img, table)	    Applies correction fast to all pixels


def add_random_occlusion(img):
D. Draw rectangle
cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, -1)
- (x1, y1) = top-left
- (x2, y2) = bottom-right
- color = fill color
- -1 = fill the rectangle
So this produces a filled block.

🟧 Example before/after
- Original image:
- [normal face photo]
- After occlusion:
- face with 1–3 random colored blocks that hide parts of it
- This helps train models to be more robust.

🟢 Simple Summary
- This function:
✔ Chooses 1–3 random rectangles
✔ Places them at random positions
✔ With random sizes
✔ With random colors
✔ Draws them on the image
✔ Returns the blocked/occluded version

🟦 Why is this useful?
- For AI training:
- Face partially covered by glasses
- Face behind hair
- Hand covering mouth
- Shadows or objects blocking parts
- Improves robustness
Your AI learns:
“Even if part of the face is missing, I can still detect head pose or landmarks.”


def reduce_resolution(img, scale=0.5):
🟩 What the function does
- This function simulates a low-resolution camera by:
- Shrinking the image to a smaller size
- Upscaling it back to the original size
- This makes the image look blurry / pixelated, just like cheap webcams or low-quality CCTV.
If scale = 0.5, then:
- new width  = 640 × 0.5 = 320
- new height = 480 × 0.5 = 240
So the image becomes 320×240 → half the original size.
This removes details and sharpness.
💡 This destroys details and creates blurry / pixelated images.

3️⃣ Upscale back to original size
- return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
- Now the low-resolution image (320×240) is stretched back to 640×480.
- This creates the “low-res camera look”:
- blurry
- pixelated
- less facial detail
- softer edges

🟥 Why do this?
- To train AI models that work with bad camera quality.
- Examples:
- cheap webcams
- old smartphone cameras
- long-distance CCTV
- video calls with low bandwidth
- If your AI can survive low resolution training, it becomes more robust.

🖼️ Visual Concept
- Original (640×480)
- clear, sharp details
- After reducing to 320×240 then upscaling
- blurry, less detail, like low resolution

🟦 Summary Table
Step	Action	            Result
1	    Read image size	    Know how big image is
2	    Downscale by scale	Loses detail
3	    Upscale back	    Looks low quality

🟢 Summary
Scale	    Size	Effect
1.0	        100%	No change
0.8	        80%	    Small blur
0.5	        50%	    Strong blur, low-res
0.3	        30%	    Very blurry
0.1	        10%	    Extremely blurry


✅ What is Gaussian Noise SIGMA?
📌 Sigma = intensity of the noise
- Sigma (σ) is how strong the noise is.
👉 In Gaussian distribution:
- mean = average (center)
- sigma = spread
- We use it like:
- noise = random values around 0
- but spread by sigma

add_gaussian_noise() adds random noise to an image to simulate:
- Poor lighting
- Cheap camera sensor
- Night vision noise
- Low-quality CCTV grain
This noise looks similar to “snowy dots” on the image

1. Generate random Gaussian noise
noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
👉 What this does:
- Creates a noise array the same size as the image
- Each pixel gets a random value
- Values follow a Gaussian (normal) distribution
- Gaussian distribution:
mean = 0 → noise is centered around 0
sigma = 25 → intensity of noise
- Higher sigma = stronger noise
- Lower sigma = smoother noise
Example:
A pixel value 128 may get noise like:
128 + (-10)
128 + (+3)
128 + (15)
128 + (-20)

2. Add noise to the image
noisy_img = img.astype(np.float32) + noise
- The noise values are added to each pixel
- Using float32 prevents overflow (255+something)
Example:
original pixel = 100
noise = +15
new pixel = 115

3. Clip pixel values to valid range
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
- Why clipping?
- Some noise values may push pixels < 0 or > 255
- Images must use 0–255 per channel
Example:
-10 → clipped to 0
300 → clipped to 255
- Then convert back to uint8 (standard image format).

🎨 Effect of sigma values
sigma = 5
- Small soft noise
- Barely visible
sigma = 15
- Moderate grain
- Looks like webcam in low light
sigma = 25 (default)
- Strong grain
- Looks like night mode or CCTV
sigma = 40+
- Extreme noise
- Image becomes very dirty

📸 Visual Concept (text version)
Original
😊 (clean)
σ = 10
🙂 (slightly grainy)
σ = 25
😐 (grainy, noisy)
σ = 50
😵 (very noisy)

🟢 Summary
Parameter	    Meaning
mean	        shifts noise brightness (usually keep 0)
sigma	        strength/intensity of noise
img.shape	    ensures noise matches image dimensions

This function is commonly used in training machine learning models to make them robust to real-world camera quality issues.

📊 VISUALIZING SIGMA 
- Sigma = 5 (low noise)
- Small dots, barely noticeable
- Sigma = 15 (medium noise)
- More dots, looks like cheap webcam
- Sigma = 25 (strong noise)
- Grainy night mode
- Sigma = 50 (very strong noise)
- Face almost destroyed by noise

📌 Example with numbers
- Say pixel value = 100
- Gaussian noise is random like:
Sigma	Example noise values
5	    -3, +2, +1, -4
15	    -12, +7, +15, -10
25	    -21, +20, +32, -27
50	    -40, +55, -60, +33
Then it adds to the pixel:
- Low sigma → small change
- High sigma → big change

🎉 Summary 
Technique	                Purpose	Visual Effect
reduce_resolution(scale)	low quality / low detail	blurry
gaussian_noise(sigma)	    bad lighting / sensor noise	grainy

Sigma = noise strength
Higher sigma → stronger noise.

🎯 KEY DIFFERENCE
Method	            What degrades image?	        Result
Reduce resolution	Removes detail → blur	        Blurry, smooth, pixelated image
Gaussian noise	    Adds random dots → grain	    Grainy, speckled, dirty image

Simple analogy:
- Reduce resolution = take a small picture and zoom it → blurry
- Gaussian noise = sprinkle sand on the picture → grainy

🧠 Why this matters for Face Tracking or Head Pose?
Reduce resolution tests:
- How well your model works when face is far / small / blurry
Gaussian noise tests:
- How well your model works in low light / bad camera / high ISO
They are completely different real-world conditions.

🎛️ Mean – Noise Shift (Bias)
- This moves the entire noise up or down.
- If mean = 0
    - Noise is centered around 0
    → Bright & dark pixels added evenly
- If mean > 0
    - Image becomes brighter overall (because noise tends upward)
Example:
mean = 20 → image becomes slightly brighter + grainy
If mean < 0
- Image becomes darker overall (noise tends downward)
Example:
mean = -20 → darker + grainy

🧪 Visual Example (easy to imagine)
✔ mean = 0, sigma = 25
Balanced grainy image
✔ mean = 20, sigma = 25
Image is grainy BUT slightly brighter
✔ mean = -20, sigma = 25
Image is grainy BUT slightly darker

❗ IMPORTANT NOTE
- For realistic camera noise:
mean = 0 is best
- Most real noise is balanced (no lighter or darker bias)
- Changing mean is only useful if you want to simulate:
Overexposed + noisy (mean > 0)
Underexposed + noisy (mean < 0)

🎯 Recommended values
Condition	            mean	sigma
Normal webcam noise	    0	    10–20
Low light noise	        0	    25–40
Very bad CCTV	        0	    50–70
Overexposed noisy cam	10–20	20–30
Underexposed noisy cam	-10–20	20–30


apply_combined_augmentation(img) applies 3 random image effects to make the image look:
- darker or brighter
- noisy
- low-resolution (sometimes)
This is useful when training AI because it teaches the model to handle real-world bad camera quality.

✅ 1. Random Gamma Correction
img = apply_gamma_correction(img, gamma=random.uniform(0.7, 1.3))
What it does:
- Makes the image randomly brighter or darker.
- gamma < 1 → brighter
- gamma > 1 → darker
random.uniform(0.7, 1.3) picks a random value between 0.7 and 1.3.
Example:
- gamma = 0.8 → image gets brighter
- gamma = 1.2 → image gets darker

✅ 2. Add Random Gaussian Noise
img = add_gaussian_noise(img, sigma=random.randint(10, 30))
What it does:
- Adds random grain/noise, like a bad webcam.
- random.randint(10, 30) picks a noise strength between 10 and 30.
lower sigma = little noise
higher sigma = strong grainy noise

✅ 3. Randomly reduce resolution (50% chance)
if random.random() > 0.5:
    img = reduce_resolution(img, scale=random.uniform(0.4, 0.7))
What this means:
- 50% chance → do nothing
- 50% chance → make the image blurry / low-res
random.random() gives a number between 0 and 1
- So if number > 0.5 → apply resolution reduction.
- scale=random.uniform(0.4, 0.7)
Means:
- Times where scale = 0.4 → very blurry
- Times where scale = 0.7 → slightly blurry
How it works:
- Shrinks the image (e.g., 640 → 300 px)
- Enlarges it back → causes blur / pixelation

It helps AI models become more robust to real-world camera problems.




