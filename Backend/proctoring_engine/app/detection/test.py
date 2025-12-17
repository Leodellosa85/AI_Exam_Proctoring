# Using 12 points landmark

# --- 12-POINT MAPPING DEFINITION ---
# MediaPipe indices (468) corresponding to 12 key points
MEDIAPIPE_TO_12_INDICES = [
    1, 152, 33, 263, 61, 291, 133, 362, 98, 327, 66, 296
]

# GT 68 indices that correspond to the 12 points above (0-based)
GT_12_INDICES = [
    30, 8, 36, 45, 48, 54, 39, 42, 31, 35, 21, 24
]
# ============================

# 3) Modular 12-point landmark detector

class LandmarkDetector:
    def __init__(self, model_name='mediapipe'):
        self.model_name = model_name
        if model_name == 'mediapipe':
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        else:
            self.face_mesh = None

    def detect_landmarks(self, image_bgr):
        """
        Return list of 12 (x,y) coordinates in pixels or None if no face detected.
        """
        if self.model_name == 'mediapipe' and self.face_mesh:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            if results and results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                # --- CHANGE 2: Use the 12-point MediaPipe indices ---
                indices = MEDIAPIPE_TO_12_INDICES

                h, w = image_bgr.shape[:2]
                return [(lm[i].x * w, lm[i].y * h) for i in indices]
        return None

    def close(self):
        if self.model_name == 'mediapipe' and self.face_mesh:
            pass

# 4) Head-pose computation (No change needed here as it handles N points)
def compute_head_pose(landmarks_2d, landmarks_3d, image_shape=None):
    """
    landmarks_2d: list of 12 (x,y) pixels
    landmarks_3d: list of 12 corresponding (X,Y,Z)
    Returns: pitch, yaw, roll in degrees
    """
    image_points = np.array(landmarks_2d, dtype=np.float64)
    model_points = np.array(landmarks_3d, dtype=np.float64)

    # Calculate Camera Matrix
    if image_shape is None:
        cx = cy = 320
        fx = fy = 640
    else:
        h, w = image_shape[:2]
        cx, cy = w/2, h/2
        fx = fy = max(h, w)

    camera_matrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    dist_coeffs = np.zeros((4,1))

    # SolvePnP
    try:
        success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)

        if not success:
            success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                return (np.nan, np.nan, np.nan)
    except Exception:
        return (np.nan, np.nan, np.nan)

    # Convert Rotation Vector to Euler Angles (RPY extraction)
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)

    # RPY calculation
    if sy < 1e-6:
        pitch = np.arctan2(-R[1,2], R[1,1])
        yaw = np.arctan2(-R[2,0], sy)
        roll = 0.0
    else:
        pitch = np.arctan2(R[2,1], R[2,2])
        yaw   = np.arctan2(-R[2,0], sy)
        roll  = np.arctan2(R[1,0], R[0,0])

    return tuple(np.degrees([pitch, yaw, roll]))

# 5) Load dataset

jpgs = sorted([f for f in os.listdir(DATASET_PATH) if f.lower().endswith('.jpg')])[:NUM_IMAGES]
print(f"Found {len(jpgs)} images to process.")

# --- CHANGE 3: Update Global GT_INDICES ---
# GT indices for 12 points (AFLW 68-point mapping)
GT_INDICES = GT_12_INDICES

detector = LandmarkDetector('mediapipe')

# --- Evaluation Metrics ---
pitch_errors, yaw_errors, roll_errors = [], [], []
per_point_errors = []
total_gt_faces = 0
true_positives = 0
false_negatives = 0
processed = 0
# --------------------------

for img_name in jpgs:
    img_path = os.path.join(DATASET_PATH, img_name)
    mat_path = img_path.replace('.jpg', '.mat')
    if not os.path.exists(mat_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue

    mat = scipy.io.loadmat(mat_path)
    pt3d = mat.get('pt3d_68')
    if pt3d is None:
        continue

    # Standardize pt3d format
    if pt3d.shape[0]==3 and pt3d.shape[1]==68:
        pt3d = pt3d.T
    elif pt3d.shape==(68,3):
        pt3d = pt3d.copy()
    else:
        continue

    # extract GT pose
    pose = mat.get('Pose_Para')
    if pose is None:
        continue

    # --- DETECTION RATE LOGIC START ---

    total_gt_faces += 1

    # detect landmarks (2D pixel coordinates) - now 12 points
    landmarks_2d = detector.detect_landmarks(image)

    if landmarks_2d is None:
        false_negatives += 1
        continue

    true_positives += 1

    # --- DETECTION RATE LOGIC END ---

    pose_arr = np.asarray(pose).squeeze()

    # Convert GT pose from radians to degrees
    gt_pitch = np.degrees(pose_arr[0])
    gt_yaw   = np.degrees(pose_arr[1])
    gt_roll  = np.degrees(pose_arr[2])

    # map GT 3D points for the 12 landmarks
    try:
        landmarks_3d = [pt3d[idx] for idx in GT_INDICES]
    except IndexError:
        continue

    # compute head pose
    pred_pitch, pred_yaw, pred_roll = compute_head_pose(landmarks_2d, landmarks_3d, image.shape)

    # Skip if PnP failed
    if np.isnan(pred_pitch):
        continue

    # errors (Absolute Error)
    pitch_errors.append(abs(pred_pitch - gt_pitch))
    yaw_errors.append(abs(pred_yaw - gt_yaw))
    roll_errors.append(abs(pred_roll - gt_roll))

    # per-point Euclidean distance errors (pixel space)
    current_per_point_error = np.mean([
        np.linalg.norm(np.array(landmarks_2d[i]) - np.array([landmarks_3d[i][0], landmarks_3d[i][1]]))
        for i in range(len(landmarks_2d))
    ])
    per_point_errors.append(current_per_point_error)

    processed += 1
    # --- CHANGE 4: Update Logging ---
    # print(f"[{processed}] {img_name}  err_pitch={pitch_errors[-1]:.2f}, err_yaw={yaw_errors[-1]:.2f}, err_roll={roll_errors[-1]:.2f}, mean_12pt_error={per_point_errors[-1]:.1f}")


detector.close()

if total_gt_faces > 0:
    detection_rate = true_positives / total_gt_faces
else:
    detection_rate = 0

print("\n======== Summary ========")
print(f"Total Ground Truth Faces Tested: {total_gt_faces}")
print(f"Faces Detected (TP): {true_positives}")
print(f"Faces Missed (FN): {false_negatives}")
print(f"DETECTION RATE (RECALL): {detection_rate:.4f}")
print("-" * 25)

if processed > 0:
    print(f"Pose Estimated Images: {processed}")
    print(f"Mean Pitch Error: {np.mean(pitch_errors):.2f}°")
    print(f"Mean Yaw Error:   {np.mean(yaw_errors):.2f}°")
    print(f"Mean Roll Error:  {np.mean(roll_errors):.2f}°")
    # --- CHANGE 4: Update Summary ---
    print(f"Mean 12-point Euclidean error: {np.mean(per_point_errors):.1f} px")
else:
    print("No images processed successfully for pose estimation.")