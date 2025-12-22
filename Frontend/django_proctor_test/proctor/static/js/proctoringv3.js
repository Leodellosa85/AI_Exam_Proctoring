import {
  FaceLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

// ===================== CONFIG =====================
const scriptTag = document.getElementById('proctor-script');
const MODEL_PATH = scriptTag?.dataset.modelPath ?? '';
const WASM_PATH = scriptTag?.dataset.wasmPath ?? '';
// Note: Ensure your Django server is actually running on port 8001
const WS_URL_TEMPLATE = scriptTag?.dataset.wsUrl ?? 'ws://127.0.0.1:8002/ws/{sessionId}';

const TARGET_FPS = 8;
const FRAME_INTERVAL = 1000 / TARGET_FPS;

// --- 1. ANGLE LIMITS ---
const ANGLE_LIMITS = {
    PITCH_DOWN: 15,     
    PITCH_UP: -25,      
    YAW_LEFT: 30,       
    YAW_RIGHT: -30,     
    ROLL_LEFT: 25,      
    ROLL_RIGHT: -25     
};

// --- 2. POSITION LIMITS ---
const SAFE_ZONE = {
    xMin: 0.2, 
    xMax: 0.8, 
    yMin: 0.1, 
    yMax: 0.9  
};

// ===================== DOM =====================
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const guideTextEl = document.getElementById("guide-text");

const yawEl = document.getElementById("yaw");
const pitchEl = document.getElementById("pitch");
const rollEl = document.getElementById("roll");
const fpsEl = document.getElementById("fps");
const framesEl = document.getElementById("frames");
const violationsEl = document.getElementById("violations");
const statusEl = document.getElementById("status");

const startBtn = document.getElementById("startBtn");
const endBtn = document.getElementById("endBtn");

// ===================== STATE =====================
let faceLandmarker;
let ws = null;
let sessionId = null;
let lastFrameTime = 0;
let frameCount = 0;
let fps = 0;

let isCalibrating = true; 
let basePose = { yaw: 0, pitch: 0, roll: 0 }; 
let lastViolationTrigger = 0;

let faceLostStartTime = null; 

let stats = {
  totalFrames: 0,
  facesDetected: 0,
  violations: 0
};

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
    minTrackingConfidence: 0.5,
    outputFacialTransformationMatrixes: true,
    // ✅ DISABLED: Correctly fixes the "CPU only ops" warning
    outputFaceBlendshapes: false, 
    runningMode: "VIDEO"
  });

  statusEl.textContent = "MediaPipe ready";
}

// ===================== HEAD POSE =====================
function calculateHeadPose(matrix) {
  const r00 = matrix[0]; const r10 = matrix[4]; const r11 = matrix[5];
  const r12 = matrix[6]; const r20 = matrix[8]; const r21 = matrix[9];
  const r22 = matrix[10];

  const sy = Math.sqrt(r00 * r00 + r10 * r10);
  let pitch, yaw, roll;
  const singular = sy < 1e-6; 

  if (!singular) {
    pitch = Math.atan2(r21, r22);
    yaw = Math.atan2(-r20, sy);
    roll = Math.atan2(r10, r00);
  } else {
    pitch = Math.atan2(-r12, r11);
    yaw = Math.atan2(-r20, sy);
    roll = 0;
  }

  return {
    pitch: -pitch * (180 / Math.PI), 
    yaw: -yaw * (180 / Math.PI),     
    roll: roll * (180 / Math.PI)
  };
}

// ===================== VIOLATION LOGIC =====================
function detectViolations(pose, box) {
  const v = [];

  // --- 1. PITCH ---
  if (pose.pitch > ANGLE_LIMITS.PITCH_DOWN) v.push("Looking down");
  if (pose.pitch < ANGLE_LIMITS.PITCH_UP)   v.push("Looking up");
  
  // --- 2. YAW ---
  if (pose.yaw > ANGLE_LIMITS.YAW_LEFT)     v.push("Head turned Left");
  if (pose.yaw < ANGLE_LIMITS.YAW_RIGHT)    v.push("Head turned Right");
  
  // --- 3. ROLL ---
  if (pose.roll > ANGLE_LIMITS.ROLL_LEFT)   v.push("Head tilted Left");
  if (pose.roll < ANGLE_LIMITS.ROLL_RIGHT)  v.push("Head tilted Right");

  // --- 4. POSITION ---
  if (box) {
      const faceCX = (box.x + box.w / 2) / canvas.width; 
      const faceCY = (box.y + box.h / 2) / canvas.height; 

      if (faceCX < SAFE_ZONE.xMin) v.push("Too far Left");
      if (faceCX > SAFE_ZONE.xMax) v.push("Too far Right");
      if (faceCY < SAFE_ZONE.yMin) v.push("Too High");
      if (faceCY > SAFE_ZONE.yMax) v.push("Too Low");
  }

  return v;
}

// ===================== DRAWING =====================
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

  if (isCalibrating) {
    ctx.strokeStyle = "#00FFFF";
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 5]); 
    
    const gX = canvas.width * SAFE_ZONE.xMin;
    const gY = canvas.height * SAFE_ZONE.yMin;
    const gW = canvas.width * (SAFE_ZONE.xMax - SAFE_ZONE.xMin);
    const gH = canvas.height * (SAFE_ZONE.yMax - SAFE_ZONE.yMin);
    ctx.strokeRect(gX, gY, gW, gH);
    ctx.setLineDash([]); 

    if(guideTextEl) guideTextEl.style.display = "block";
  }

  if (box) {
    ctx.lineWidth = 3;
    ctx.strokeStyle = isCalibrating ? "yellow" : (isViolation ? "red" : "#00FF00");
    ctx.strokeRect(box.x, box.y, box.w, box.h);
  }
}

// ===================== MAIN LOOP =====================
let lastVideoTime = -1;
let lastProcessTime = 0;
const FACE_LOST_THRESHOLD = 3000;       // 3 Seconds (Warning/Start Logging)
const AUTO_TERMINATE_THRESHOLD = 30000; // 30 Seconds (End Exam)

function processFrames() {
  if (video.ended) return; 
  requestAnimationFrame(processFrames);

  if (!faceLandmarker || video.paused) return;

  const now = performance.now();
  if (now - lastProcessTime < FRAME_INTERVAL) return;
  lastProcessTime = now;

  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  stats.totalFrames++;
  frameCount++;

  if (frameCount >= TARGET_FPS) {
    fps = Math.round(1000 / ((now - lastFrameTime) / frameCount));
    frameCount = 0;
    lastFrameTime = now;
  }

  const results = faceLandmarker.detectForVideo(video, now);

  // =========================================================
  // CASE A: FACE DETECTED
  // =========================================================
  if (results?.facialTransformationMatrixes?.length > 0) {
      // 1. RESET FACE LOSS TIMER (Face is back!)
      faceLostStartTime = null;

      stats.facesDetected++;
    
      const landmarks = results.faceLandmarks[0];
      const matrix = Array.from(results.facialTransformationMatrixes[0].data);
      const pose = calculateHeadPose(matrix);
      const box = getBoundingBox(landmarks);

    // --- CALIBRATION ---
    if (isCalibrating) {
        statusEl.textContent = "⚠ Align face in box...";
        statusEl.className = "status warn";
        
        const faceCX = (box.x + box.w / 2) / canvas.width;
        const faceCY = (box.y + box.h / 2) / canvas.height;

        const isCenteredX = faceCX > SAFE_ZONE.xMin && faceCX < SAFE_ZONE.xMax;
        const isCenteredY = faceCY > SAFE_ZONE.yMin && faceCY < SAFE_ZONE.yMax;
        const isCloseEnough = box.w > (canvas.width * 0.15); 

        if (guideTextEl) {
            if (!isCenteredX) guideTextEl.textContent = "Move Left / Right";
            else if (!isCenteredY) guideTextEl.textContent = "Move Up / Down";
            else if (!isCloseEnough) guideTextEl.textContent = "Move Closer";
            else guideTextEl.textContent = "Hold Still...";
        }

        if (isCenteredX && isCenteredY && isCloseEnough) {
            basePose = { yaw: pose.yaw, pitch: pose.pitch, roll: pose.roll };
            isCalibrating = false;
            faceLandmarker.setOptions({ minFaceDetectionConfidence: 0.4 });
            statusEl.textContent = "✓ Exam Started";
            statusEl.className = "status ok";
        }
        drawBoundingBox(box, false);
        return; 
    }

    // --- EXAM MODE ---
    const relativePose = {
        yaw: pose.yaw - basePose.yaw,
        pitch: pose.pitch - basePose.pitch,
        roll: pose.roll - basePose.roll
    };

    const violations = detectViolations(relativePose, box);
     
    yawEl.textContent = relativePose.yaw.toFixed(1) + "°";
    pitchEl.textContent = relativePose.pitch.toFixed(1) + "°";
    rollEl.textContent = relativePose.roll.toFixed(1) + "°";

    fpsEl.textContent = fps;
    framesEl.textContent = stats.totalFrames;
    violationsEl.textContent = stats.violations;

    // Prepare Payload
    let wsPayload = {
        type: "metadata",
        timestamp: Date.now(),
        pose: relativePose,
        violations: violations,
        image: null
    };

    if (violations.length) {
        statusEl.textContent = violations[0]; 
        statusEl.className = "status warn";
        
        const timestamp = Date.now();
        
        // Rate Limit: 1 violation per second
        if (timestamp - lastViolationTrigger >= 1000) {
            stats.violations++;
            lastViolationTrigger = timestamp;

            // Capture Image
            const captureCanvas = document.createElement('canvas');
            captureCanvas.width = video.videoWidth;
            captureCanvas.height = video.videoHeight;
            const capCtx = captureCanvas.getContext('2d');
            capCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
            
            if(box) {
                capCtx.lineWidth = 5;
                capCtx.strokeStyle = "red";
                capCtx.strokeRect(box.x, box.y, box.w, box.h);
            }

            wsPayload.image = captureCanvas.toDataURL('image/jpeg', 0.5);
        }
    } else {
        statusEl.textContent = "✓ Normal behavior";
        statusEl.className = "status ok";
    }

    drawBoundingBox(box, violations.length > 0);

    if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(wsPayload));
    }

  } else {
    // =========================================================
    // CASE B: FACE NOT DETECTED (Logic Updated)
    // =========================================================
    
    yawEl.textContent = "--";
    drawBoundingBox(null, false);

    // 1. If Calibrating, show error immediately (don't wait 3 seconds)
    if (isCalibrating) {
        statusEl.textContent = "⚠ Show Face";
        statusEl.className = "status err";
        return;
    }

    // 2. Exam Mode: Handle 3 Second Delay
    
    // Start the timer if it hasn't started yet
    if (faceLostStartTime === null) {
        faceLostStartTime = now;
    }

    // Calculate how long face has been missing
    const timeMissing = now - faceLostStartTime;

    if (timeMissing > AUTO_TERMINATE_THRESHOLD) {
        // A. Update UI
        statusEl.textContent = "❌ EXAM TERMINATED (Abandonment)";
        statusEl.className = "status err";

        // B. Send Final Log to Server
        if (ws?.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: "metadata",
                timestamp: Date.now(),
                pose: { yaw: 0, pitch: 0, roll: 0 },
                violations: ["Session Terminated: Face Lost > 30s"],
                image: null // No need for image, the previous 27 were enough
            }));
            ws.close(); // Cut connection
        }

        // C. Stop Camera & Processing
        video.srcObject?.getTracks().forEach(t => t.stop());
        startBtn.disabled = false;
        endBtn.disabled = true;
        
        // D. Stop Loop
        return; // This 'return' stops requestAnimationFrame from running again
    }
    // --- CHECK 2: LOG VIOLATION (3 Seconds) ---
    if (timeMissing > FACE_LOST_THRESHOLD) {
        statusEl.textContent = `⚠ Face Lost (${Math.round(timeMissing/1000)}s)`;
        statusEl.className = "status err";

        const timestamp = Date.now();
        if (timestamp - lastViolationTrigger >= 1000) {
            stats.violations++;
            lastViolationTrigger = timestamp;

            const captureCanvas = document.createElement('canvas');
            captureCanvas.width = video.videoWidth;
            captureCanvas.height = video.videoHeight;
            const capCtx = captureCanvas.getContext('2d');
            capCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

            if (ws?.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: "metadata",
                    timestamp: timestamp,
                    pose: { yaw: 0, pitch: 0, roll: 0 },
                    violations: ["Face Lost"],
                    image: captureCanvas.toDataURL('image/jpeg', 0.5)
                }));
            }
        }
    } else {
        // --- WARNING PHASE (Less than 3 seconds) ---
        // Show Yellow warning, but do NOT record violation yet
        statusEl.textContent = "⚠ Re-acquiring face...";
        statusEl.className = "status warn"; // Yellow
    }
  }
}


// ===================== WS & UI =====================
function connectWS() {
  const url = WS_URL_TEMPLATE.replace('{sessionId}', sessionId);
  ws = new WebSocket(url);
  
  ws.onerror = (err) => {
    console.error("WebSocket Error:", err);
    statusEl.textContent = "WS Error - Is Django running on port 8001?";
    statusEl.className = "status err";
  };
}

startBtn.onclick = async () => {
  startBtn.disabled = true;
  statusEl.textContent = "Loading Model...";

   try {
    await initMediaPipe();

    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: { width: 640, height: 480 } 
    });
    video.srcObject = stream;

    sessionId = crypto.randomUUID();
    connectWS();

    video.onloadedmetadata = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      video.play().then(() => {
        endBtn.disabled = false;
        statusEl.textContent = "Calibrating...";
        
        isCalibrating = true;
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
  ws?.close();
  video.srcObject?.getTracks().forEach(t => t.stop());
  
  startBtn.disabled = false;
  endBtn.disabled = true;
  statusEl.textContent = "Session ended";
  statusEl.className = "status";
  
  yawEl.textContent = "--"; pitchEl.textContent = "--"; rollEl.textContent = "--";
  fps = 0; fpsEl.textContent = "0";
  framesEl.textContent = "0"; violationsEl.textContent = "0";
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if(guideTextEl) guideTextEl.style.display = "none";
};