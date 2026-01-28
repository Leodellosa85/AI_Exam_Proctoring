import {
  FaceLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

// ===================== CONFIG =====================
const scriptTag = document.getElementById("proctor-script");
const MODEL_PATH = scriptTag?.dataset.modelPath ?? "";
const WASM_PATH = scriptTag?.dataset.wasmPath ?? "";
// Django (logging / violations)
const WS_LOG_URL = "ws://127.0.0.1:8002/ws/v1/ws/{sessionId}";

// FastAPI (liveness)
const WS_LIVENESS_URL = "ws://127.0.0.1:8001/ws2/{sessionId}";
// const WS_URL_TEMPLATE =
//   scriptTag?.dataset.wsUrl ?? "ws://127.0.0.1:8002/ws/{sessionId}";

let wsLog = null;
let wsLiveness = null;


const TARGET_FPS = 8;
const FRAME_INTERVAL = 1000 / TARGET_FPS;

const ANGLE_LIMITS = {
  PITCH_DOWN: 15,
  PITCH_UP: -25,
  YAW_LEFT: 30,
  YAW_RIGHT: -30,
  ROLL_LEFT: 25,      
  ROLL_RIGHT: -25    
};

const SAFE_ZONE = { xMin: 0.2, xMax: 0.8, yMin: 0.1, yMax: 0.9 };

const TIER = {
  FLAG: 3000,
  BLUR: 5000,
  TERMINATE: 15000,
  CUMULATIVE: 60000,
  STABILITY: 3000
};

const SPOOF_POLICY = {
  SUSPICIOUS_LOCK_MS: 3000,   // lock if suspicious lasts 3s
  FAKE_TERMINATE_MS: 1500     // terminate if fake lasts 1.5s
};

// ===================== STATE =====================
let faceLandmarker, ws, sessionId;

const MODE = {
  CALIBRATION: "calibration",
  MONITORING: "monitoring",
  LOCKED: "locked",
  TERMINATED: "terminated"
};

let currentMode = MODE.CALIBRATION;

let lastProcessTime = 0;
let frameCount = 0;
let fps = 0;
let lastFpsTime = 0;

let isCalibrating = true;
let isLocked = false;
let isTerminated = false;

let basePose = { yaw: 0, pitch: 0, roll: 0 };

let lastViolationLogTime = 0;
let missingSince = null;
let totalMissingMs = 0;
let stabilityMs = 0;
let lastViolationTrigger = 0;
let stats = { violations: 0 };
let calibrationStartTime = null;

const LIVENESS_WINDOW = 16;
let livenessBuffer = [];
let lastPoseForMotion = null;

const SPOOF_SEND_INTERVAL = 10;   // every 10 frames (~1.25s @ 8fps)
const CROP_SCALE = 1.6;
// const CROP_SCALE = 2.7;
let spoofFrameCounter = 0;

let spoofStatus = "real";
let spoofSince = null;

let useBackendLiveness = true;   // toggle flag




// ===================== ELEMENTS =====================
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const appRoot = document.getElementById("app-root");
const guideTextEl = document.getElementById("guide-text");
const statusEl = document.getElementById("status");

const yawEl = document.getElementById("yaw");
const pitchEl = document.getElementById("pitch");
const rollEl = document.getElementById("roll");
const totalLostEl = document.getElementById("total-lost");
const violationsEl = document.getElementById("violations");
const fpsEl = document.getElementById("fps");

const startBtn = document.getElementById("startBtn");
const endBtn = document.getElementById("endBtn");

const toggleLivenessBtn = document.getElementById("toggleLivenessBtn");
const spoofScoreEl = document.getElementById("spoof-score");
const livenessStatusEl = document.getElementById("liveness-status");



// ===================== INIT =====================
async function initMediaPipe() {
  statusEl.textContent = "Loading AI Model…";
  const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_PATH, delegate: "CPU" },
    numFaces: 1,
    minFaceDetectionConfidence: 0.6,
    minTrackingConfidence: 0.5,
    outputFacialTransformationMatrixes: true,
    runningMode: "VIDEO"
  });

  statusEl.textContent = "AI ready";
}

// ===================== MAIN LOOP =====================
function processFrames() {
//   if (video.ended || isTerminated) return;
  if (currentMode === MODE.TERMINATED || video.ended) return;
  requestAnimationFrame(processFrames);

  const now = performance.now();
  if (now - lastProcessTime < FRAME_INTERVAL) return;
  lastProcessTime = now;

  frameCount++;
  if (now - lastFpsTime >= 1000) {
    fps = frameCount;
    frameCount = 0;
    lastFpsTime = now;
    fpsEl.textContent = fps;
  }

  const results = faceLandmarker.detectForVideo(video, now);
  const faceDetected =
    results?.facialTransformationMatrixes?.length &&
    results?.faceLandmarks?.length;

  switch (currentMode) {
    case MODE.CALIBRATION:
      handleCalibration(faceDetected, results, now);
      break;

    case MODE.MONITORING:
      handleMonitoring(faceDetected, results, now);
      break;

    case MODE.LOCKED:
      handleLocked(faceDetected, results, now);
      break;
  }
}

function handleCalibration(faceDetected, results, now) {
  if (!faceDetected) {
    statusEl.textContent = "⚠ SHOW YOUR FACE";
    statusEl.className = "status warn";
    drawBoundingBox(null, false);
    return;
  }

  const landmarks = results.faceLandmarks[0];
  const matrix = Array.from(results.facialTransformationMatrixes[0].data);
  const pose = calculateHeadPose(matrix);
  const box = getBoundingBox(landmarks);

  runCalibration(now, pose, box);
  drawBoundingBox(box, false);
}

function handleMonitoring(faceDetected, results, now) {
  if (!faceDetected) {
    handleFaceMissing(now);
    return;
  }

  missingSince = null;
  stabilityMs = 0;

  const landmarks = results.faceLandmarks[0];
  const matrix = Array.from(results.facialTransformationMatrixes[0].data);
  const pose = calculateHeadPose(matrix);
  const box = getBoundingBox(landmarks);

  // ---- Send face crop for anti-spoofing ----
  spoofFrameCounter++;

  if (spoofFrameCounter % SPOOF_SEND_INTERVAL === 0) {
    sendFaceCrop(box);
  }


  const relativePose = {
    yaw: pose.yaw - basePose.yaw,
    pitch: pose.pitch - basePose.pitch,
    roll: pose.roll - basePose.roll
  };

  const violations = detectViolations(relativePose, box);

  updateDisplayMetrics(relativePose);
  drawBoundingBox(box, violations.length > 0);

  syncMonitoringData(relativePose, violations, box);
}

function handleFaceMissing(now) {
  if (!missingSince) missingSince = now;

  const missingMs = now - missingSince;
  addCumulativeMissingTime();
  stabilityMs = 0;

  yawEl.textContent = "--";
  pitchEl.textContent = "--";
  rollEl.textContent = "--";

  statusEl.textContent = "sFACE MISSING";
  statusEl.className = "status warn";
  drawBoundingBox(null, false);

  if (missingMs >= TIER.BLUR) {
    lockSession();
  }

  if (missingMs >= TIER.TERMINATE) {
    terminateExam("Face missing too long");
  }

  if (totalMissingMs >= TIER.CUMULATIVE) {
    terminateExam("Total off-screen time exceeded");
  }

  syncMonitoringData(null, ["Face Missing"], null);
}

function handleLocked(faceDetected, results, now) {
  statusEl.textContent = "SESSION LOCKED";
  statusEl.className = "status err";

  if (!faceDetected) {
    stabilityMs = 0;
    addCumulativeMissingTime();
    return;
  }

  stabilityMs += FRAME_INTERVAL;
  addCumulativeMissingTime();
  const remaining = Math.max(0, TIER.STABILITY - stabilityMs);

  guideTextEl.style.display = "block";
  guideTextEl.textContent = `HOLD STILL ${Math.ceil(remaining / 1000)}s`;

  if (stabilityMs >= TIER.STABILITY) {
    unlockSession();
  }
}

function lockSession() {
  currentMode = MODE.LOCKED;
  stabilityMs = 0;
  appRoot.classList.add("exam-lock");
}

function unlockSession() {
  currentMode = MODE.MONITORING;
  missingSince = null;
  stabilityMs = 0;

  guideTextEl.style.display = "none";
  appRoot.classList.remove("exam-lock");
}

function terminateExam(reason) {
  currentMode = MODE.TERMINATED;

  statusEl.textContent = "❌ TERMINATED";
  statusEl.className = "status err";

  guideTextEl.style.display = "block";
  guideTextEl.innerHTML = `EXAM ENDED<br><small>${reason}</small>`;
  appRoot.classList.add("exam-lock");

  video.srcObject?.getTracks().forEach(t => t.stop());
}

// ===================== CALIBRATION =====================
function runCalibration(now, pose, box) {
  if (!box) return;
  const faceCX = (box.x + box.w / 2) / canvas.width;
  const faceCY = (box.y + box.h / 2) / canvas.height;

  const violations = [];

  if (faceCX < SAFE_ZONE.xMin) violations.push("Too far Right");
  if (faceCX > SAFE_ZONE.xMax) violations.push("Too far Left");
  if (faceCY < SAFE_ZONE.yMin) violations.push("Too High");
  if (faceCY > SAFE_ZONE.yMax) violations.push("Too Low");

  if (violations.length > 0) {
    calibrationStartTime = null;

    guideTextEl.style.display = "block";
    guideTextEl.textContent = violations.join("\n");

    statusEl.textContent = "⚠ POSITION FACE";
    statusEl.className = "status warn";
    return;
  }

  const centered =
    faceCX > SAFE_ZONE.xMin &&
    faceCX < SAFE_ZONE.xMax &&
    faceCY > SAFE_ZONE.yMin &&
    faceCY < SAFE_ZONE.yMax;

  if (!centered) {
    calibrationStartTime = null;
    guideTextEl.style.display = "block";
    guideTextEl.textContent = "CENTER YOUR FACE";
    statusEl.textContent = "⚠ POSITION FACE";
    statusEl.className = "status warn";
    return;
  }

  if (!calibrationStartTime) calibrationStartTime = now;
  const elapsed = now - calibrationStartTime;
  const remaining = Math.ceil((TIER.STABILITY - elapsed) / 1000);

  guideTextEl.textContent = `HOLD STILL... ${remaining}s`;

  if (elapsed >= TIER.STABILITY) {
    basePose = pose;
    isCalibrating = false;
    guideTextEl.style.display = "none";
    statusEl.textContent = "✓ STARTED";
    statusEl.className = "status ok";
    currentMode = MODE.MONITORING;
  }
}

// ===================== UTILS =====================
function calculateHeadPose(m) {
  const sy = Math.sqrt(m[0] * m[0] + m[4] * m[4]);
  const p = Math.atan2(m[9], m[10]);
  const y = Math.atan2(-m[8], sy);
  const r = Math.atan2(m[4], m[0]);

  return {
    pitch: -p * 57.3,
    yaw: -y * 57.3,
    roll: r * 57.3
  };
}

function getBoundingBox(landmarks) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of landmarks) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }
  return {
    x: minX * canvas.width,
    y: minY * canvas.height,
    w: (maxX - minX) * canvas.width,
    h: (maxY - minY) * canvas.height
  };
}

function detectViolations(p,box) {
  const v = [];
  if (p.pitch > ANGLE_LIMITS.PITCH_DOWN) v.push("Looking Down");
  if (p.pitch < ANGLE_LIMITS.PITCH_UP) v.push("Looking Up");
  if (p.yaw > ANGLE_LIMITS.YAW_LEFT) v.push("Looking Left");
  if (p.yaw < ANGLE_LIMITS.YAW_RIGHT) v.push("Looking Right");
  if (p.roll > ANGLE_LIMITS.ROLL_LEFT)   v.push("Head tilted Left");
  if (p.roll < ANGLE_LIMITS.ROLL_RIGHT)  v.push("Head tilted Right");

  if (box) {
    const faceCX = (box.x + box.w / 2) / canvas.width;
    const faceCY = (box.y + box.h / 2) / canvas.height;

    if (faceCX < SAFE_ZONE.xMin) v.push("Too far Right");
    if (faceCX > SAFE_ZONE.xMax) v.push("Too far Left");
    if (faceCY < SAFE_ZONE.yMin) v.push("Too High");
    if (faceCY > SAFE_ZONE.yMax) v.push("Too Low");
  }

  return v;
}

function drawBoundingBox(box, isViolation) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if(guideTextEl) guideTextEl.style.display = "none";
  if (!box) return;

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

function updateDisplayMetrics(p) {
  yawEl.textContent = `${p.yaw.toFixed(1)}°`;
  pitchEl.textContent = `${p.pitch.toFixed(1)}°`;
  rollEl.textContent = `${p.roll.toFixed(1)}°`;
}

function buildBasePayload(relativePose, violations) {
  return {
    type: "metadata",
    timestamp: Math.round(performance.now()),
    pose: relativePose || { yaw: 0, pitch: 0, roll: 0 },
    violations: violations || [],
    image: null
  };
}

function captureViolationImage(video, box) {
  const captureCanvas = document.createElement("canvas");
  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;

  const ctx = captureCanvas.getContext("2d");
  ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

  if (box) {
    ctx.lineWidth = 5;
    ctx.strokeStyle = "red";
    ctx.strokeRect(box.x, box.y, box.w, box.h);
  }

  return captureCanvas.toDataURL("image/jpeg", 0.5);
}

function addCumulativeMissingTime() {
  totalMissingMs += FRAME_INTERVAL;
  totalLostEl.textContent = `${Math.floor(totalMissingMs / 1000)}s`;

  if (totalMissingMs >= TIER.CUMULATIVE) {
    terminateExam("Total off-screen / locked time exceeded");
  }
}


function syncMonitoringData(relativePose, violations, box) {
  const now = performance.now(); // Use performance.now for consistency
  const payload = buildBasePayload(relativePose, violations);

  if (violations.length > 0) {
    // 1. Update status text every frame so UI is responsive
    statusEl.textContent = violations[0];
    statusEl.className = "status warn";

    // 2. Throttle Image Capture and Counter to 1 second
    if (now - lastViolationTrigger >= 1000) {
      lastViolationTrigger = now;
      stats.violations++;
      violationsEl.textContent = stats.violations; // Update UI counter

      payload.image = captureViolationImage(video, box);
      if (wsLog?.readyState === WebSocket.OPEN) {
        wsLog.send(JSON.stringify(payload));
      }
    }
  } else {
    statusEl.textContent = "✓ SECURE";
    statusEl.className = "status ok";
     if (now - lastViolationLogTime >= 1000) {
      lastViolationLogTime = now;

      if (wsLog?.readyState === WebSocket.OPEN) {
        wsLog.send(JSON.stringify(payload));
      }
    }
  }
}


// ===================== EVENTS =====================
startBtn.onclick = async () => {
  startBtn.disabled = true;
  await initMediaPipe();

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 }
  });

  console.log("✅ Camera access granted");
  console.log(WS_LIVENESS_URL)

  video.srcObject = stream;
  sessionId = crypto.randomUUID();

  wsLog = new WebSocket(WS_LOG_URL.replace("{sessionId}", sessionId));
  wsLiveness = new WebSocket(WS_LIVENESS_URL.replace("{sessionId}", sessionId));  
//   ws = new WebSocket(WS_URL_TEMPLATE.replace("{sessionId}", sessionId));
  wsLiveness.onopen = () => console.log("✅ Connected to liveness backend");
  wsLiveness.onerror = (e) => console.error("❌ WebSocket error", e);
  wsLiveness.onclose = () => console.warn("⚠ WebSocket closed");
  wsLiveness.onmessage = (e) => console.log("📩 Backend:", e.data);

  wsLiveness.onmessage = (event) => {
  const data = JSON.parse(event.data);

//   if (data.liveness) {
//     handleLivenessResult(data);
//   }
  if ("spoof_score" in data) {
    handleLivenessResult(data);
  }
};


  video.onloadedmetadata = () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    video.play();

    isCalibrating = true;
    isLocked = false;
    isTerminated = false;
    totalMissingMs = 0;
    stats.violations = 0;
    violationsEl.textContent = "0";
    lastViolationTrigger = 0; 

    processFrames();
    endBtn.disabled = false;
  };
};

endBtn.onclick = () => {
  terminateExam("User ended session");
  ws?.close();
};


function handleLivenessResult(data) {
  if (!data || !data.liveness) return;
  console.log("Liveness result:", data);

   // ---- Always show score ----
  if (typeof data.spoof_score === "number") {
    spoofScoreEl.textContent = data.spoof_score.toFixed(3);
  }

  livenessStatusEl.textContent = data.liveness.toUpperCase();

  if (!useBackendLiveness) {
    return; // UI only, no lock, no terminate, no punishment
  }


  const now = performance.now();
  const newStatus = data.liveness;

  // Reset timer if status changed
  if (newStatus !== spoofStatus) {
    spoofStatus = newStatus;
    spoofSince = now;
  }

  const duration = spoofSince ? now - spoofSince : 0;


  if (newStatus === "suspicious") {
    // Lock if sustained
    if (duration >= SPOOF_POLICY.SUSPICIOUS_LOCK_MS &&
        currentMode === MODE.MONITORING) {
      lockSession();
    }
    return;
  }

  if (newStatus === "fake") {
    // Immediate cumulative counting
    addCumulativeMissingTime();

    if (duration >= SPOOF_POLICY.FAKE_TERMINATE_MS &&
        currentMode !== MODE.TERMINATED) {
      terminateExam("Spoofing / Fake face detected");
    }
  }
}



function cropFaceScaled(video, box, scale = CROP_SCALE) {
  if (!box) return null;

  const vw = video.videoWidth;
  const vh = video.videoHeight;

  const cx = box.x + box.w / 2;
  const cy = box.y + box.h / 2;
  const size = Math.max(box.w, box.h) * scale;

  // Define the crop area (might go out of bounds)
  const x = cx - size / 2;
  const y = cy - size / 2;

  // Always create a square canvas
  const c = document.createElement("canvas");
  c.width = 120;  // Fixed size for transmission (saves bandwidth)
  c.height = 120;
  const ctx = c.getContext("2d");

  // This drawImage version handles out-of-bounds by not stretching
  // It maps the video region (x, y, size, size) to the canvas (0, 0, 120, 120)
  ctx.drawImage(video, x, y, size, size, 0, 0, 120, 120);

  // Use 0.8 quality to keep the file small but keep texture details
//   return c.toDataURL("image/jpeg", 0.8);
  return new Promise(resolve => {
    c.toBlob(blob => resolve(blob), "image/jpeg", 0.9);
  });
}


async function sendFaceCrop(box) {
  if (!wsLiveness || wsLiveness.readyState !== WebSocket.OPEN) return;

  const blob = await cropFaceScaled(video, box, CROP_SCALE);
  if (!blob) return;

  wsLiveness.send(JSON.stringify({
    type: "face_crop",
    encoding: "binary",
    mime: "image/jpeg",
    size: blob.size
  }));

  const buffer = await blob.arrayBuffer();
  wsLiveness.send(buffer);
}


toggleLivenessBtn.onclick = () => {
  useBackendLiveness = !useBackendLiveness;

  toggleLivenessBtn.textContent =
    `Liveness: ${useBackendLiveness ? "ON" : "OFF"}`;

  toggleLivenessBtn.style.background =
    useBackendLiveness ? "#22c55e" : "#6b7280";
};

