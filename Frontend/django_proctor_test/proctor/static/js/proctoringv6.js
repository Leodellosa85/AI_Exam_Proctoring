import {
  FaceLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

// ===================== CONFIG =====================
const scriptTag = document.getElementById("proctor-script");
const MODEL_PATH = scriptTag?.dataset.modelPath ?? "";
const WASM_PATH = scriptTag?.dataset.wasmPath ?? "";

// Endpoints
const WS_LOG_URL = "ws://127.0.0.1:8002/ws/v1/ws/{sessionId}";
const WS_LIVENESS_URL = "ws://127.0.0.1:8001/ws2/{sessionId}";

const TARGET_FPS = 8;
const FRAME_INTERVAL = 1000 / TARGET_FPS;

const ANGLE_LIMITS = {
  PITCH_DOWN: 15, PITCH_UP: -25,
  YAW_LEFT: 30, YAW_RIGHT: -30,
  ROLL_LEFT: 25, ROLL_RIGHT: -25    
};

const SAFE_ZONE = { xMin: 0.2, xMax: 0.8, yMin: 0.1, yMax: 0.9 };

const TIER = {
  FLAG: 3000, BLUR: 5000, TERMINATE: 15000,
  CUMULATIVE: 30000, STABILITY: 3000
};

const SPOOF_POLICY = {
  SUSPICIOUS_LOCK_MS: 3000,
  FAKE_TERMINATE_MS: 1500     
};

// --- Active Liveness Config ---
const CHALLENGE_CONFIG = {
  TRIGGER_THRESHOLD: 0.9,  // Trigger if spoof_score < 0.9
  DURATION_FRAMES: 100,     // ~7.5 seconds at 8fps
  CORRELATION_PASS: 0.5,   // Pass if correlation > 0.6
  DOT_RADIUS: 12,
  COOLDOWN_MS: 20000,      // 20s between challenges
  LERP_SPEED: 0.25 
};

// ===================== STATE =====================
const MODE = {
  CALIBRATION: "calibration",
  MONITORING: "monitoring",
  LOCKED: "locked",
  TERMINATED: "terminated"
};

let currentMode = MODE.CALIBRATION;
let faceLandmarker, wsLog, wsLiveness, sessionId;
let lastProcessTime = 0;
let frameCount = 0;
let fps = 0;
let lastFpsTime = 0;

let isCalibrating = true;
let basePose = { yaw: 0, pitch: 0, roll: 0 };
let missingSince = null;
let totalMissingMs = 0;
let stabilityMs = 0;
let lastViolationTrigger = 0;
let stats = { violations: 0 };
let calibrationStartTime = null;

// Anti-Spoofing State
let spoofFrameCounter = 0;
const SPOOF_SEND_INTERVAL = 10;
let spoofStatus = "real";
let spoofSince = null;
let useBackendLiveness = true;
const CROP_SCALE = 1.2;

// Active Liveness (Challenge) State
let challengeActive = false;
let challengeFrameCount = 0;
let lastChallengeTime = 0;
let dotPos = { x: 0.5, y: 0.5 };
let dotTarget = { x: 0.5, y: 0.5 };
let dotHistory = []; 
let gazeHistory = []; 

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
  if (currentMode === MODE.TERMINATED || video.ended) return;
  requestAnimationFrame(processFrames);

  const now = performance.now();
  if (now - lastProcessTime < FRAME_INTERVAL) return;
  lastProcessTime = now;

  // FPS Counter
  frameCount++;
  if (now - lastFpsTime >= 1000) {
    fps = frameCount;
    frameCount = 0;
    lastFpsTime = now;
    fpsEl.textContent = fps;
  }

  const results = faceLandmarker.detectForVideo(video, now);
  const faceDetected = results?.facialTransformationMatrixes?.length > 0 && results?.faceLandmarks?.length > 0;

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

// ===================== CORE HANDLERS =====================

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

  // 1. Passive Liveness (Backend Crop)
  spoofFrameCounter++;
  if (spoofFrameCounter % SPOOF_SEND_INTERVAL === 0) {
    sendFaceCrop(box);
  }

  // 2. Active Liveness (Challenge Logic)
  if (challengeActive) {
    runActiveChallengeLogic(landmarks);
  }

  // 3. Pose & Violation logic
  const relativePose = {
    yaw: pose.yaw - basePose.yaw,
    pitch: pose.pitch - basePose.pitch,
    roll: pose.roll - basePose.roll
  };

  const violations = detectViolations(relativePose, box);
  updateDisplayMetrics(relativePose);
  
  // 4. UI Drawing
  drawBoundingBox(box, violations.length > 0);
  if (challengeActive) drawChallengeDot();

  syncMonitoringData(relativePose, violations, box);
}

function handleFaceMissing(now) {
  if (!missingSince) missingSince = now;
  const missingMs = now - missingSince;
  addCumulativeMissingTime();

  yawEl.textContent = "--"; pitchEl.textContent = "--"; rollEl.textContent = "--";
  drawBoundingBox(null, false);

  if (missingMs < TIER.FLAG) {
    statusEl.textContent = "FACE LOST - RETURN";
    statusEl.className = "status warn";
  } else {
    statusEl.textContent = "VIOLATION: FACE MISSING";
    statusEl.className = "status err";
    syncMonitoringData(null, ["Face Missing"], null);
  }

  if (missingMs >= TIER.BLUR) lockSession();
}

function handleLocked(faceDetected, results, now) {
  statusEl.textContent = "SESSION LOCKED";
  statusEl.className = "status err";

  if (!faceDetected) {
    stabilityMs = 0;
    if (now - missingSince >= TIER.TERMINATE) terminateExam("Abandoned Session");
    return;
  }

  stabilityMs += FRAME_INTERVAL;
  const remaining = Math.max(0, TIER.STABILITY - stabilityMs);
  guideTextEl.style.display = "block";
  guideTextEl.textContent = `HOLD STILL ${Math.ceil(remaining / 1000)}s`;

  if (stabilityMs >= TIER.STABILITY) unlockSession();
}

// ===================== ACTIVE LIVENESS LOGIC =====================

function startActiveChallenge() {
  if (challengeActive) return;
  console.log("🚀 Liveness score low. Triggering Active Challenge...");
  challengeActive = true;
  challengeFrameCount = 0;
  dotHistory = [];
  gazeHistory = [];
  
  guideTextEl.style.display = "block";
  guideTextEl.innerHTML = "SECURITY CHECK:<br>FOLLOW THE MOVING DOT WITH YOUR EYES";
  statusEl.textContent = "⚠ VERIFYING LIVENESS";
  statusEl.className = "status warn";
}

function runActiveChallengeLogic(landmarks) {
  // 1. Move dot to a distant target
  if (challengeFrameCount % 25 === 0) {
    let newX, newY, dist;
    do {
      newX = 0.05 + Math.random() * 0.9; // Expanded range to 5% - 95%
      newY = 0.1 + Math.random() * 0.8;
      // Calculate distance from current target
      dist = Math.sqrt(Math.pow(newX - dotTarget.x, 2) + Math.pow(newY - dotTarget.y, 2));
    } while (dist < 0.4); // Force the dot to move at least 40% of the screen distance

    dotTarget = { x: newX, y: newY };
  }

  // 2. Smooth movement
  dotPos.x += (dotTarget.x - dotPos.x) * CHALLENGE_CONFIG.LERP_SPEED;
  dotPos.y += (dotTarget.y - dotPos.y) * CHALLENGE_CONFIG.LERP_SPEED;

  // 3. Iris and Eye Corners
  const iris = landmarks[468];
  const inner = landmarks[133];
  const outer = landmarks[33];
  
  const eyeCX = (inner.x + outer.x) / 2;
  const eyeCY = (inner.y + outer.y) / 2;

  // 4. Save History
  dotHistory.push({ x: dotPos.x, y: dotPos.y });
  gazeHistory.push({ x: iris.x - eyeCX, y: iris.y - eyeCY });

  challengeFrameCount++;
  if (challengeFrameCount >= CHALLENGE_CONFIG.DURATION_FRAMES) {
    finishChallenge();
  }
}

function finishChallenge() {
  challengeActive = false;
  lastChallengeTime = performance.now();
  guideTextEl.style.display = "none";

  // HUMAN LATENCY COMPENSATION:
  // Humans take about 150-250ms to react. At 8fps, that's ~2 frames.
  // We shift the arrays so the eye movement matches the dot movement that caused it.
  const shift = 2; 
  const shiftedDot = dotHistory.slice(0, dotHistory.length - shift);
  const shiftedGaze = gazeHistory.slice(shift);

  const dx = shiftedDot.map(p => p.x);
  const gx = shiftedGaze.map(p => p.x);
  const dy = shiftedDot.map(p => p.y);
  const gy = shiftedGaze.map(p => p.y);

  const corrX = calculatePearson(dx, gx);
  const corrY = calculatePearson(dy, gy);
  
  // Use Max correlation or Average. Usually, horizontal (X) is more reliable.
  const finalCorr = (corrX + corrY) / 2;

  console.log(`Challenge Result - X: ${corrX.toFixed(2)}, Y: ${corrY.toFixed(2)}`);

  if (finalCorr > CHALLENGE_CONFIG.CORRELATION_PASS) {
    statusEl.textContent = "✓ LIVENESS VERIFIED: " + finalCorr.toFixed(2);
    statusEl.className = "status ok";
    console.log("PASS: Eye movement correlates with dot.");
  } else {
    statusEl.textContent = "LIVENESS FAILED: " + finalCorr.toFixed(2);
    statusEl.className = "status err";
    syncMonitoringData(null, ["Active Liveness Failed (Low Correlation)"], null);
  }
}

function calculatePearson(x, y) {
  const n = x.length;
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((a, v, i) => a + v * y[i], 0);
  const sumX2 = x.reduce((a, v) => a + v * v, 0);
  const sumY2 = y.reduce((a, v) => a + v * v, 0);
  const num = n * sumXY - sumX * sumY;
  const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
  return den === 0 ? 0 : num / den;
}

function drawChallengeDot() {
  const x = dotPos.x * canvas.width;
  const y = dotPos.y * canvas.height;
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, CHALLENGE_CONFIG.DOT_RADIUS, 0, Math.PI * 2);
  ctx.fillStyle = "red";
  ctx.shadowBlur = 15;
  ctx.shadowColor = "red";
  ctx.fill();
  ctx.restore();
}

// ===================== LIVENESS BACKEND INTEGRATION =====================

function handleLivenessResult(data) {
  if (!data) return;

  if (typeof data.spoof_score === "number") {
    spoofScoreEl.textContent = data.spoof_score.toFixed(3);
    
    // TRIGGER ACTIVE CHALLENGE IF SCORE IS LOW
    const now = performance.now();
    if (data.spoof_score < CHALLENGE_CONFIG.TRIGGER_THRESHOLD && 
        !challengeActive && 
        (now - lastChallengeTime > CHALLENGE_CONFIG.COOLDOWN_MS)) {
      startActiveChallenge();
    }
  }

  if (data.liveness) {
    livenessStatusEl.textContent = data.liveness.toUpperCase();
    const now = performance.now();
    if (data.liveness !== spoofStatus) {
      spoofStatus = data.liveness;
      spoofSince = now;
    }
    
    if (useBackendLiveness) {
      const duration = now - spoofSince;
      if (data.liveness === "fake" && duration >= SPOOF_POLICY.FAKE_TERMINATE_MS) {
        terminateExam("Spoofing detected");
      }
    }
  }
}

// ===================== SESSION UTILS =====================

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

  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
  wsLog?.close();
  wsLiveness?.close();
}

// ===================== CALCULATIONS =====================

function calculateHeadPose(m) {
  const sy = Math.sqrt(m[0] * m[0] + m[4] * m[4]);
  const p = Math.atan2(m[9], m[10]);
  const y = Math.atan2(-m[8], sy);
  const r = Math.atan2(m[4], m[0]);
  return { pitch: -p * 57.3, yaw: -y * 57.3, roll: r * 57.3 };
}

function getBoundingBox(landmarks) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of landmarks) {
    minX = Math.min(minX, p.x); minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x); maxY = Math.max(maxY, p.y);
  }
  return {
    x: minX * canvas.width, y: minY * canvas.height,
    w: (maxX - minX) * canvas.width, h: (maxY - minY) * canvas.height
  };
}

function detectViolations(p, box) {
  const v = [];
  if (p.pitch > ANGLE_LIMITS.PITCH_DOWN) v.push("Looking Down");
  if (p.pitch < ANGLE_LIMITS.PITCH_UP) v.push("Looking Up");
  if (p.yaw > ANGLE_LIMITS.YAW_LEFT) v.push("Looking Left");
  if (p.yaw < ANGLE_LIMITS.YAW_RIGHT) v.push("Looking Right");
  if (box) {
    const cx = (box.x + box.w / 2) / canvas.width;
    const cy = (box.y + box.h / 2) / canvas.height;
    if (cx < SAFE_ZONE.xMin || cx > SAFE_ZONE.xMax) v.push("Face off-center");
  }
  return v;
}

function drawBoundingBox(box, isViolation) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!box) return;
  ctx.lineWidth = 3;
  ctx.strokeStyle = isCalibrating ? "cyan" : (isViolation ? "red" : "#00FF00");
  ctx.strokeRect(box.x, box.y, box.w, box.h);
}

function addCumulativeMissingTime() {
  totalMissingMs += FRAME_INTERVAL;
  totalLostEl.textContent = `${Math.floor(totalMissingMs / 1000)}s`;
  if (totalMissingMs >= TIER.CUMULATIVE) terminateExam("Off-screen time exceeded");
}

function syncMonitoringData(relativePose, violations, box) {
  const now = performance.now();
  const payload = {
    type: "metadata",
    timestamp: Math.round(now),
    pose: relativePose || { yaw: 0, pitch: 0, roll: 0 },
    violations: violations || [],
    image: null
  };

  if (violations.length > 0) {
    if (now - lastViolationTrigger >= 1000) {
      lastViolationTrigger = now;
      stats.violations++;
      violationsEl.textContent = stats.violations;
      payload.image = captureViolationImage(video, box);
      if (wsLog?.readyState === WebSocket.OPEN) wsLog.send(JSON.stringify(payload));
    }
  } else if (now - lastViolationLogTime >= 1000) {
    lastViolationLogTime = now;
    if (wsLog?.readyState === WebSocket.OPEN) wsLog.send(JSON.stringify(payload));
  }
}

function captureViolationImage(video, box) {
  const c = document.createElement("canvas");
  c.width = video.videoWidth; c.height = video.videoHeight;
  const ctx = c.getContext("2d");
  ctx.drawImage(video, 0, 0);
  if (box) { ctx.strokeStyle = "red"; ctx.lineWidth = 5; ctx.strokeRect(box.x, box.y, box.w, box.h); }
  return c.toDataURL("image/jpeg", 0.5);
}

async function sendFaceCrop(box) {
  if (!wsLiveness || wsLiveness.readyState !== WebSocket.OPEN || !box) return;
  const vw = video.videoWidth, vh = video.videoHeight;
  const size = Math.max(box.w, box.h) * CROP_SCALE;
  const x = (box.x + box.w / 2) - size / 2;
  const y = (box.y + box.h / 2) - size / 2;

  const c = document.createElement("canvas");
  c.width = 128; c.height = 128;
  c.getContext("2d").drawImage(video, x, y, size, size, 0, 0, 128, 128);
  
  c.toBlob(async (blob) => {
    wsLiveness.send(JSON.stringify({ type: "face_crop", size: blob.size }));
    wsLiveness.send(await blob.arrayBuffer());
  }, "image/jpeg", 0.9);
}

function updateDisplayMetrics(p) {
  yawEl.textContent = `${p.yaw.toFixed(1)}°`;
  pitchEl.textContent = `${p.pitch.toFixed(1)}°`;
  rollEl.textContent = `${p.roll.toFixed(1)}°`;
}

function runCalibration(now, pose, box) {
  if (!calibrationStartTime) calibrationStartTime = now;
  const elapsed = now - calibrationStartTime;
  const remaining = Math.ceil((TIER.STABILITY - elapsed) / 1000);
  guideTextEl.style.display = "block";
  guideTextEl.textContent = `CALIBRATING... HOLD STILL ${remaining}s`;

  if (elapsed >= TIER.STABILITY) {
    basePose = pose;
    isCalibrating = false;
    currentMode = MODE.MONITORING;
    guideTextEl.style.display = "none";
  }
}

// ===================== EVENTS =====================
startBtn.onclick = async () => {
  startBtn.disabled = true;
  await initMediaPipe();
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  video.srcObject = stream;
  sessionId = crypto.randomUUID();

  wsLog = new WebSocket(WS_LOG_URL.replace("{sessionId}", sessionId));
  wsLiveness = new WebSocket(WS_LIVENESS_URL.replace("{sessionId}", sessionId));

  wsLiveness.onmessage = (e) => handleLivenessResult(JSON.parse(e.data));

  video.onloadedmetadata = () => {
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    video.play();
    processFrames();
    endBtn.disabled = false;
  };
};

endBtn.onclick = () => terminateExam("User Ended");

toggleLivenessBtn.onclick = () => {
  useBackendLiveness = !useBackendLiveness;
  toggleLivenessBtn.textContent = `Liveness: ${useBackendLiveness ? "ON" : "OFF"}`;
  toggleLivenessBtn.style.background = useBackendLiveness ? "#22c55e" : "#6b7280";
};