import {
  FaceLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

// ===================== CONFIG =====================
const scriptTag = document.getElementById('proctor-script');
const MODEL_PATH = scriptTag?.dataset.modelPath ?? '';
const WASM_PATH = scriptTag?.dataset.wasmPath ?? '';
const WS_URL_TEMPLATE = scriptTag?.dataset.wsUrl ?? 'ws://127.0.0.1:8001/ws/{sessionId}';

const TARGET_FPS = 8;
const FRAME_INTERVAL = 1000 / TARGET_FPS;

// ===================== STATE =====================
let faceLandmarker;
let ws = null;
let sessionId = null;
let lastFrameTime = 0;
let frameCount = 0;
let fps = 0;

let isCalibrating = true; // <--- New State for Startup
let lastViolationTrigger = 0; // <--- New Timer for Per-Second violations
let basePose = { yaw: 0, pitch: 0, roll: 0 }; 

let stats = {
  totalFrames: 0,
  facesDetected: 0,
  violations: 0,
  lastViolationTime: 0
};

// ===================== THRESHOLD CONFIG =====================
// Positive/Negative signs depend on your math logic.
// Based on standard Euler angles:
// Pitch: Positive = Down, Negative = Up
// Yaw:   Positive = Left, Negative = Right (or vice versa depending on mirror)

// --- 1. ANGLE LIMITS (Relative to Calibration) ---
const ANGLE_LIMITS = {
    PITCH_DOWN: 15,    // Looking Down limit
    PITCH_UP: -25,     // Looking Up limit
    YAW_SIDE: 30,      // Turning Left/Right limit (+/- 30)
    ROLL: 25           // Tilt limit
};

// --- 2. POSITION LIMITS (The "Safe Zone") ---
// We use the same box size as calibration (50% of screen center)
// If face center leaves this box -> Violation.
const SAFE_ZONE = {
    xMin: 0.25, // 25% from left
    xMax: 0.75, // 75% from left
    yMin: 0.15, // 15% from top (allow more headroom)
    yMax: 0.85  // 85% from top
};

// ===================== DOM =====================
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const yawEl = document.getElementById("yaw");
const pitchEl = document.getElementById("pitch");
const rollEl = document.getElementById("roll");
const fpsEl = document.getElementById("fps");
const framesEl = document.getElementById("frames");
const violationsEl = document.getElementById("violations");
const statusEl = document.getElementById("status");

const startBtn = document.getElementById("startBtn");
const endBtn = document.getElementById("endBtn");

const guideTextEl = document.getElementById("guide-text"); // 

// ===================== INIT =====================
async function initMediaPipe() {
  statusEl.textContent = "Initializing MediaPipe…";

  const vision = await FilesetResolver.forVisionTasks(
    WASM_PATH || 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_PATH,
      delegate: "CPU"
    },
    numFaces: 1,
    minFaceDetectionConfidence: 0.6,
    minTrackingConfidence: 0.4,
    outputFacialTransformationMatrixes: true,
    outputFaceBlendshapes: true,
    runningMode: "VIDEO" // ✅ Use VIDEO mode for continuous detection
  });

  statusEl.textContent = "MediaPipe ready";
}

// ===================== HEAD POSE =====================
// ===================== HEAD POSE (Matches Python Collab) =====================
function calculateHeadPose(matrix) {
  // The matrix 'm' is a 1D array of 16 elements (4x4 matrix flattened).
  // Indices map to the 3x3 rotation matrix like this:
  // 0  1  2
  // 4  5  6
  // 8  9  10

  const r00 = matrix[0];
  const r10 = matrix[4];
  const r11 = matrix[5];
  const r12 = matrix[6];
  const r20 = matrix[8];
  const r21 = matrix[9];
  const r22 = matrix[10];

  // Python: sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
  const sy = Math.sqrt(r00 * r00 + r10 * r10);

  let pitch, yaw, roll;
  const singular = sy < 1e-6; // Handle Gimbal Lock

  if (!singular) {
    // Python: pitch = np.arctan2(R[2,1], R[2,2]) (Rotation around X)
    pitch = Math.atan2(r21, r22);

    // Python: yaw = np.arctan2(-R[2,0], sy)      (Rotation around Y)
    yaw = Math.atan2(-r20, sy);

    // Python: roll = np.arctan2(R[1,0], R[0,0])  (Rotation around Z)
    roll = Math.atan2(r10, r00);
  } else {
    // Python: pitch = np.arctan2(-R[1,2], R[1,1])
    pitch = Math.atan2(-r12, r11);
    
    // Python: yaw = np.arctan2(-R[2,0], sy)
    yaw = Math.atan2(-r20, sy);
    
    // Python: roll = 0
    roll = 0;
  }

  // Convert Radians to Degrees
  return {
    pitch: -pitch * (180 / Math.PI),
    yaw: yaw * (180 / Math.PI),
    roll: roll * (180 / Math.PI)
  };
}

// ===================== VIOLATION LOGIC =====================
function detectViolations(pose, blendshapes, box) {
//   const v = [];

//   if (Math.abs(pose.yaw) > 30) v.push("Head turned");
//   if (Math.abs(pose.pitch) > 20) v.push("Looking away");
//   if (Math.abs(pose.roll) > 25) v.push("Head tilted");

//   if (blendshapes) {
//     const l = blendshapes.find(b => b.categoryName === 'eyeBlinkLeft')?.score || 0;
//     const r = blendshapes.find(b => b.categoryName === 'eyeBlinkRight')?.score || 0;
//     if (l > 0.7 && r > 0.7) v.push("Eyes closed");
//   }

//   return v;

  const v = [];

   // --- 1. ANGLE CHECKS (Relative to Calibration) ---
  if (pose.pitch > ANGLE_LIMITS.PITCH_DOWN) v.push("Looking down");
  if (pose.pitch < ANGLE_LIMITS.PITCH_UP)   v.push("Looking up");
  
  // Check Left/Right (Absolute value covers both sides)
  if (Math.abs(pose.yaw) > ANGLE_LIMITS.YAW_SIDE) v.push("Head turned");
  
  if (Math.abs(pose.roll) > ANGLE_LIMITS.ROLL) v.push("Head tilted");

  // --- 2. POSITION CHECKS (Stay in the center!) ---
  if (box) {
      const faceCX = (box.x + box.w / 2) / canvas.width; // 0.0 to 1.0
      const faceCY = (box.y + box.h / 2) / canvas.height; // 0.0 to 1.0

      if (faceCX < SAFE_ZONE.xMin) v.push("Too far Left");
      if (faceCX > SAFE_ZONE.xMax) v.push("Too far Right");
      if (faceCY < SAFE_ZONE.yMin) v.push("Too High");
      if (faceCY > SAFE_ZONE.yMax) v.push("Too Low");
  }

  // --- 3. EYES ---
  if (blendshapes) {
    const l = blendshapes.find(b => b.categoryName === 'eyeBlinkLeft')?.score || 0;
    const r = blendshapes.find(b => b.categoryName === 'eyeBlinkRight')?.score || 0;
    if (l > 0.7 && r > 0.7) v.push("Eyes closed");
  }

  return v;
}

// ===================== DRAW & BOX LOGIC =====================
function getBoundingBox(landmarks) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  
  for (const p of landmarks) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  
  return {
    x: minX * canvas.width,
    y: minY * canvas.height,
    w: (maxX - minX) * canvas.width,
    h: (maxY - minY) * canvas.height
  };
}

function drawBoundingBox(box, isViolation) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
 if(guideTextEl) guideTextEl.style.display = "none";

  // Draw Calibration Guide (if needed)
  if (isCalibrating) {
    ctx.strokeStyle = "#00FFFF"; // Cyan
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 5]); // Dashed line
    // // Draw a box in the center 50% of screen
    // const guideX = canvas.width * 0.25;
    // const guideY = canvas.height * 0.25;
    // const guideW = canvas.width * 0.5;
    // const guideH = canvas.height * 0.5;
    // ctx.strokeRect(guideX, guideY, guideW, guideH);
    // ctx.setLineDash([]); // Reset dash

     // Draw the "Ideal" Box
    const gX = canvas.width * SAFE_ZONE.xMin;
    const gY = canvas.height * SAFE_ZONE.yMin;
    const gW = canvas.width * (SAFE_ZONE.xMax - SAFE_ZONE.xMin);
    const gH = canvas.height * (SAFE_ZONE.yMax - SAFE_ZONE.yMin);
    ctx.strokeRect(gX, gY, gW, gH);
    ctx.setLineDash([]); 

    if(guideTextEl) {
        guideTextEl.style.display = "block";
        // Text updated in loop
    }
  }
    
    // ctx.fillStyle = "white";
    // ctx.font = "16px Arial";
    // ctx.fillText("Please fit face inside box", guideX, guideY - 10);

     // UPDATE HTML TEXT INSTEAD OF CANVAS TEXT
    // guideTextEl.style.display = "block";
//     // guideTextEl.textContent = "Please fit face inside box\nto start exam";
//   }

   // Draw Face Box
  if (box) {
    ctx.lineWidth = 3;
    ctx.strokeStyle = isCalibrating ? "yellow" : (isViolation ? "red" : "#00FF00");
    ctx.strokeRect(box.x, box.y, box.w, box.h);
  }
}


// ===================== DRAW =====================
// function drawMesh(results, violations) {
//   ctx.clearRect(0, 0, canvas.width, canvas.height);

//   if (!results.faceLandmarks?.[0]) return;

//   ctx.fillStyle = violations.length ? 'rgba(255,0,0,0.4)' : 'rgba(0,255,0,0.3)';
//   for (const p of results.faceLandmarks[0]) {
//     ctx.beginPath();
//     ctx.arc(p.x * canvas.width, p.y * canvas.height, 1, 0, Math.PI * 2);
//     ctx.fill();
//   }
// }

let lastVideoTime = -1;
let lastProcessTime = 0; // New variable for smooth throttling

// ===================== MAIN LOOP =====================
function processFrames() {

    // If video stopped or model missing, stop the loop safely
  if (video.ended) return; 

  // Always request the next frame to keep the loop alive
  requestAnimationFrame(processFrames);

  // If paused or model not ready, just skip this iteration (don't kill the loop)
  if (!faceLandmarker || video.paused) return;

  const now = performance.now();
   // THROTTLING: Only process if enough time has passed (Smooth 8 FPS)
  if (now - lastProcessTime < FRAME_INTERVAL) {
    return;
  }
  lastProcessTime = now; // Update time immediately after check

  // MEDIA PIPE STRICT CHECK: 
  // Only detect if video time has advanced
  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  // --- START DETECTION ---
  stats.totalFrames++;
  frameCount++;

  // Calculate Real FPS
  if (frameCount >= TARGET_FPS) {
    // Current time - time when we started this batch of frames
    fps = Math.round(1000 / ((now - lastFrameTime) / frameCount));
    frameCount = 0;
    lastFrameTime = now;
  }
  // Detect
  const results = faceLandmarker.detectForVideo(video, now);

  // ... (Rest of your drawing/logic remains the same) ...
  if (results?.facialTransformationMatrixes?.length > 0) {
      stats.facesDetected++;
    
      const landmarks = results.faceLandmarks[0];
      const matrix = Array.from(results.facialTransformationMatrixes[0].data);
      const pose = calculateHeadPose(matrix);
      const blend = results.faceBlendshapes?.[0]?.categories;
      const box = getBoundingBox(landmarks);

      // --- CALIBRATION CHECK ---
    if (isCalibrating) {

        statusEl.textContent = "⚠ Align face in box...";
        statusEl.className = "status warn";

         // Calculate Center
        const faceCX = (box.x + box.w / 2) / canvas.width;
        const faceCY = (box.y + box.h / 2) / canvas.height;

         // Check against Safe Zone
        const isCenteredX = faceCX > SAFE_ZONE.xMin && faceCX < SAFE_ZONE.xMax;
        const isCenteredY = faceCY > SAFE_ZONE.yMin && faceCY < SAFE_ZONE.yMax;
        const isCloseEnough = box.w > (canvas.width * 0.15); // Face must be reasonably sized

         if (guideTextEl) {
            if (!isCenteredX) guideTextEl.textContent = "Move Left / Right";
            else if (!isCenteredY) guideTextEl.textContent = "Move Up / Down";
            else if (!isCloseEnough) guideTextEl.textContent = "Move Closer";
            else guideTextEl.textContent = "Hold Still...";
        }

        if (isCenteredX && isCenteredY && isCloseEnough) {
            // CAPTURE BASE POSE
            basePose = { yaw: pose.yaw, pitch: pose.pitch, roll: pose.roll };
            isCalibrating = false;
            faceLandmarker.setOptions({ minFaceDetectionConfidence: 0.4 });
            statusEl.textContent = "✓ Exam Started";
            statusEl.className = "status ok";
        }
        drawBoundingBox(box, false);
        return; 

         // 1. Define the Guide Box (The Cyan Box)
        // const guideX = canvas.width * 0.25;      // Starts at 25% width
        // const guideY = canvas.height * 0.25;     // Starts at 25% height
        // const guideW = canvas.width * 0.5;       // 50% width
        // const guideH = canvas.height * 0.5;      // 50% height

        // 2. Get Face Center Point
        // const faceCX = box.x + (box.w / 2);
        // const faceCY = box.y + (box.h / 2);

        // 3. Strict Checks
        
        // Is Face Centered Horizontally? (Must be inside the guide X range)
        // const isCenteredX = faceCX > guideX && faceCX < (guideX + guideW);

        // // Is Face Centered Vertically? (Must be inside the guide Y range)
        // const isCenteredY = faceCY > guideY && faceCY < (guideY + guideH);

        // // Is Face Big Enough? (Must fill at least 40% of the guide box width)
        // // This prevents people from calibrating while standing 10 feet away.
        // const isCloseEnough = box.w > (guideW * 0.4);

        //  if (isCenteredX && isCenteredY && isCloseEnough) {
        //     // SUCCESS: Store Zero Point
        //     basePose = { 
        //         yaw: pose.yaw, 
        //         pitch: pose.pitch, 
        //         roll: pose.roll 
        //     };

        //     isCalibrating = false;
        //     faceLandmarker.setOptions({ minFaceDetectionConfidence: 0.4 });
            
        //     statusEl.textContent = "✓ Exam Started";
        //     statusEl.className = "status ok";
        // } else {
        //     // Provide Feedback to User
        //     if (!isCenteredX) statusEl.textContent = "Move Left/Right";
        //     else if (!isCenteredY) statusEl.textContent = "Move Up/Down";
        //     else if (!isCloseEnough) statusEl.textContent = "Move Closer";
        //     statusEl.className = "status warn";
        // }


        // // Draw the visual boxes
        // drawBoundingBox(box, false);
        // return; 
    }

    // --- EXAM MODE (RELATIVE POSE) ---
    // Calculate pose relative to the calibrated "Zero" point
    const relativePose = {
        yaw: pose.yaw - basePose.yaw,
        pitch: pose.pitch - basePose.pitch,
        roll: pose.roll - basePose.roll
    };

    //   const violations = detectViolations(pose, blend);
      // 2. Detect Violations (Pass Box for position check)
    const violations = detectViolations(relativePose, blend, box);

      yawEl.textContent = relativePose.yaw.toFixed(1) + "°";
      pitchEl.textContent = relativePose.pitch.toFixed(1) + "°";
      rollEl.textContent = relativePose.roll.toFixed(1) + "°";
     
    //   yawEl.textContent = pose.yaw.toFixed(1) + "°";
    //   pitchEl.textContent = pose.pitch.toFixed(1) + "°";
    //   rollEl.textContent = pose.roll.toFixed(1) + "°";

      fpsEl.textContent = fps;
      framesEl.textContent = stats.totalFrames;
      violationsEl.textContent = stats.violations;
      // --- VIOLATION LOGIC (PER SECOND) ---
      if (violations.length) {
        // statusEl.textContent = "⚠ Suspicious behavior";
        statusEl.textContent = violations[0];
        statusEl.className = "status warn";
        // Only increment violation if 1 second has passed since last one
      const timestamp = Date.now();
      if (timestamp - lastViolationTrigger >= 1000) {
        stats.violations++;
        lastViolationTrigger = timestamp;
      }
    } else {
      statusEl.textContent = "✓ Normal behavior";
      statusEl.className = "status ok";
    }
    //   drawMesh(results, detectViolations(pose, results.faceBlendshapes?.[0]?.categories));
    drawBoundingBox(box, violations.length > 0);
      if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: "metadata",
        timestamp: Date.now(),
        pose,
        violations
      }));
    }
  } else {
      yawEl.textContent = "--";
      statusEl.textContent = isCalibrating ? "⚠ Show Face" : "⚠ Face Lost / Blocked";
    //   yawEl.textContent = pitchEl.textContent = rollEl.textContent = "--";
    //   statusEl.textContent = isCalibrating ? "⚠ Show Face" : "⚠ No face detected";
      statusEl.className = "status err";
      // Clear whole canvas but keep guide if calibrating
     drawBoundingBox(null, false);
    //   ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas if no face
  }
}

// ===================== WS =====================
function connectWS() {
  const url = WS_URL_TEMPLATE.replace('{sessionId}', sessionId);
  ws = new WebSocket(url);
}

// ===================== UI =====================
startBtn.onclick = async () => {
  startBtn.disabled = true;
  statusEl.textContent = "Loading Model...";

   try {
    // 1. Initialize AI first
    await initMediaPipe();

    // 2. Get Camera
    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: { width: 640, height: 480 } 
    });
    video.srcObject = stream;

    // 3. Initialize Session
    sessionId = crypto.randomUUID();
    connectWS();

    video.onloadedmetadata = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // 3. Play and Start Loop
      video.play().then(() => {
        endBtn.disabled = false;
        statusEl.textContent = "Calibrating...";
        isCalibrating = true; // Reset state on start
        stats = { totalFrames: 0, facesDetected: 0, violations: 0 };
        violationsEl.textContent = "0";
        framesEl.textContent = "0";
        processFrames();
      });
    };

  } catch (err) {
    console.error(err);
    statusEl.textContent = "Error: " + err.message;
    statusEl.className = "status err";
    startBtn.disabled = false;
  }
};


endBtn.onclick = () => {
  // 1. Stop AI & Camera
  ws?.close();
  video.srcObject?.getTracks().forEach(t => t.stop());
  
  // 2. Reset Buttons
  startBtn.disabled = false;
  endBtn.disabled = true;
  
  // 3. Reset Status Text
  statusEl.textContent = "Session ended";
  statusEl.className = "status";
  yawEl.textContent = "--";
  pitchEl.textContent = "--";
  rollEl.textContent = "--";

  // 4. Reset Stats Visuals
  fps = 0;
  fpsEl.textContent = "0";
  // We keep the final count visible until they click start again,
  // or you can set them to 0 here immediately:
  stats.frameCount = 0;
  stats.violations = 0;
  framesEl.textContent = "0";
  violationsEl.textContent = "0";
  
  // 5. Clear Canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if(guideTextEl) guideTextEl.style.display = "none";
};
