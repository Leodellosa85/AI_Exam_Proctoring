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
const WS_LIVENESS_URL = "ws://127.0.0.1:8001/ws/{sessionId}";
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
  YAW_RIGHT: -30
};

const SAFE_ZONE = { xMin: 0.2, xMax: 0.8, yMin: 0.1, yMax: 0.9 };

const TIER = {
  FLAG: 3000,
  BLUR: 5000,
  TERMINATE: 15000,
  CUMULATIVE: 60000,
  STABILITY: 3000
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

//   if (faceDetected) {
//     missingSince = null;
//     handleFacePresent(results, now);
//   } else {
//     if (!missingSince) missingSince = now;
//     handleFaceMissing(now);
//   }
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

  const relativePose = {
    yaw: pose.yaw - basePose.yaw,
    pitch: pose.pitch - basePose.pitch,
    roll: pose.roll - basePose.roll
  };
  sendLivenessFeatures(relativePose);

  const violations = detectViolations(relativePose);
  updateDisplayMetrics(relativePose);
  drawBoundingBox(box, violations.length > 0);

  syncMonitoringData(relativePose, violations, box);
}

function handleFaceMissing(now) {
  if (!missingSince) missingSince = now;

  const missingMs = now - missingSince;
  totalMissingMs += FRAME_INTERVAL;
  stabilityMs = 0;

  yawEl.textContent = "--";
  pitchEl.textContent = "--";
  rollEl.textContent = "--";
  totalLostEl.textContent = `${Math.floor(totalMissingMs / 1000)}s`;

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
    return;
  }

  stabilityMs += FRAME_INTERVAL;
  const remaining = Math.max(0, TIER.STABILITY - stabilityMs);

  guideTextEl.style.display = "block";
  guideTextEl.textContent = `HOLD STILL ${Math.ceil(remaining / 1000)}s`;

  if (stabilityMs >= TIER.STABILITY) {
    unlockSession();
  }
}


// ===================== FACE PRESENT =====================
function handleFacePresent(results, now) {
  const landmarks = results.faceLandmarks[0];
  const matrix = Array.from(
    results.facialTransformationMatrixes[0].data
  );

  const pose = calculateHeadPose(matrix);
  const box = getBoundingBox(landmarks);

  // -------- LOCKED STATE --------
  if (isLocked) {
    stabilityMs += FRAME_INTERVAL;
    const remaining = Math.max(0, TIER.STABILITY - stabilityMs);

    statusEl.textContent = `⚠ STABILIZING... ${(remaining / 1000).toFixed(1)}s`;
    statusEl.className = "status warn";

    guideTextEl.style.display = "block";
    guideTextEl.textContent = "HOLD STILL TO UNLOCK";
    appRoot.classList.add("exam-lock");

    drawBoundingBox(box, false);

    if (stabilityMs >= TIER.STABILITY) unlockSession();
    return;
  }

  // -------- CALIBRATION --------
  if (isCalibrating) {
    runCalibration(now, pose, box);
    drawBoundingBox(box, false);
    return;
  }

  // -------- NORMAL MONITORING --------
  stabilityMs = 0;

  const relativePose = {
    yaw: pose.yaw - basePose.yaw,
    pitch: pose.pitch - basePose.pitch,
    roll: pose.roll - basePose.roll
  };

  


  updateDisplayMetrics(relativePose);
  const violations = detectViolations(relativePose);

  if (violations.length) {
    statusEl.textContent = violations[0];
    statusEl.className = "status warn";
    drawBoundingBox(box, true);
  } else {
    statusEl.textContent = "✓ SECURE";
    statusEl.className = "status ok";
    drawBoundingBox(box, false);
  }

  syncMonitoringData(relativePose, violations, box);
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
// ===================== STATE HELPERS =====================
// function lockSession() {
//   isLocked = true;
//   stabilityMs = 0;
//   appRoot.classList.add("exam-lock");
//   guideTextEl.style.display = "block";
//   guideTextEl.textContent = "FACE LOST\nEXAM BLURRED";
// }

// function unlockSession() {
//   isLocked = false;
//   stabilityMs = 0;
//   guideTextEl.style.display = "none";
//   appRoot.classList.remove("exam-lock");

//   statusEl.textContent = "✓ SECURE";
//   statusEl.className = "status ok";
// }

// function terminateExam(reason) {
//   isTerminated = true;
//   statusEl.textContent = "❌ TERMINATED";
//   statusEl.className = "status err";

//   guideTextEl.style.display = "block";
//   guideTextEl.innerHTML = `EXAM ENDED<br><small>${reason}</small>`;
//   appRoot.classList.add("exam-lock");

//   video.srcObject?.getTracks().forEach(t => t.stop());
// }

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

function detectViolations(p) {
  const v = [];
  if (p.pitch > ANGLE_LIMITS.PITCH_DOWN) v.push("Looking Down");
  if (p.pitch < ANGLE_LIMITS.PITCH_UP) v.push("Looking Up");
  if (p.yaw > ANGLE_LIMITS.YAW_LEFT) v.push("Looking Left");
  if (p.yaw < ANGLE_LIMITS.YAW_RIGHT) v.push("Looking Right");
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
      if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(payload));
      }
    }
  } else {
    statusEl.textContent = "✓ SECURE";
    statusEl.className = "status ok";
     if (now - lastViolationLogTime >= 1000) {
      lastViolationLogTime = now;

      if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(payload));
      }
    }
  }

  // 3. Send to WebSocket (Metadata sent every frame, image only when captured)
//   if (ws?.readyState === WebSocket.OPEN) {
//     ws.send(JSON.stringify(payload));
//   }
}


// ===================== EVENTS =====================
startBtn.onclick = async () => {
  startBtn.disabled = true;
  await initMediaPipe();

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 }
  });

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

  if (data.liveness) {
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

function computeMotion(curr, prev) {
  if (!prev) return 0;

  const dy = curr.yaw - prev.yaw;
  const dp = curr.pitch - prev.pitch;
  const dr = curr.roll - prev.roll;

  return Math.sqrt(dy*dy + dp*dp + dr*dr);
}

function sendLivenessFeatures(pose) {
  if (!pose) return;
  const rawmotion = computeMotion(pose, lastPoseForMotion);
  lastPoseForMotion = pose;

  const motion = Math.min(rawmotion / 10.0, 3.0);

  const features = [
    pose.yaw,
    pose.pitch,
    pose.roll,
    motion
  ];

  livenessBuffer.push(features);

  if (livenessBuffer.length > LIVENESS_WINDOW) {
    livenessBuffer.shift();
  }

  if (livenessBuffer.length === LIVENESS_WINDOW) {
    if (wsLiveness?.readyState === WebSocket.OPEN) {
    wsLiveness.send(JSON.stringify({
      type: "liveness_features",
      features
    }));
  }
  }
}

function handleLivenessResult(data) {
  if (data.liveness === "fake") {
    statusEl.textContent = "⚠ SPOOF DETECTED";
    statusEl.className = "status err";
  }
  else if (data.liveness === "suspicious") {
    statusEl.textContent = "⚠ LIVENESS UNCERTAIN";
    statusEl.className = "status warn";
  }
  console.log("Liveness result:", data);
}

// s
