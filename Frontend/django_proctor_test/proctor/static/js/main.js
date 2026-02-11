/**
 * ----------------------------------------------------
 * Application entry point and orchestration layer.
 *
 * Responsibilities:
 * - Wire UI elements to application logic
 * - Initialize MediaPipe face detection
 * - Drive the frame processing loop
 * - Coordinate session state transitions (calibration, monitoring, lock)
 * - Delegate liveness detection to backend via LivenessClient
 *
 * This file intentionally contains NO:
 * - Model preprocessing logic
 * - WebSocket protocol details
 * - Business rules for violations or penalties
 *
 * Those concerns are handled by dedicated modules.
 * ----------------------------------------------------
 */

import { MODE, FRAME_INTERVAL, SPOOF_SEND_INTERVAL, TIER } from "./config.js";

import { initFaceLandmarker, detectFace } from "./mediapipe/faceLandmarker.js";
import { calculateHeadPose, getBoundingBox } from "./mediapipe/pose.js";

import { drawBoundingBox } from "./ui/overlay.js";
import {
  setStatus,
  updateDisplayMetrics,
  updateTotalLostUI,
} from "./ui/status.js";

import { sessionState, resetSessionState } from "./state/sessionState.js";
import { LivenessClient } from "./liveness/livenessClient.js";

import {
  generateMiniCrops,
  generateFaceBagnetCrop,
} from "./liveness/cropper.js";

import { detectViolations } from "./logic/violations.js";
import { runCalibration } from "./logic/calibration.js";
import { handleFaceMissing } from "./logic/timers.js";
import { sendMonitoringData } from "./logic/logging.js";
import { handleLivenessResult } from "./logic/livenessPolicy.js";
import {
  lockSession,
  unlockSession,
  terminateSession,
} from "./state/sessionActions.js";

import { WS_LOG_URL, WS_LIVENESS_URL } from "./config.js";
import { captureViolationImage } from "./media/capture.js";

/* ===================== DOM ELEMENTS ===================== */
const appRoot = document.getElementById("app-root");
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const statusEl = document.getElementById("status");
const guideTextEl = document.getElementById("guide-text");
const fpsEl = document.getElementById("fps");

const yawEl = document.getElementById("yaw");
const pitchEl = document.getElementById("pitch");
const rollEl = document.getElementById("roll");
const totalLostEl = document.getElementById("total-lost");
const violationsEl = document.getElementById("violations");

const spoofScoreEl = document.getElementById("spoof-score");
const livenessStatusEl = document.getElementById("liveness-status");

const startBtn = document.getElementById("startBtn");
const endBtn = document.getElementById("endBtn");

const scriptTag = document.getElementById("proctor-script");

/* ===================== RUNTIME OBJECTS ===================== */
let faceLandmarker = null;
let livenessClient = null;

let useBackendLiveness = true;

/* ===================== SESSION SIDE EFFECTS ===================== */
function applySessionEffects(reason = "") {
  switch (sessionState.mode) {
    case MODE.LOCKED:
      appRoot.classList.add("exam-lock");
      guideTextEl.style.display = "block";
      guideTextEl.textContent = "SESSION LOCKED";
      endBtn.disabled = false;
      break;

    case MODE.MONITORING:
      appRoot.classList.remove("exam-lock");
      guideTextEl.style.display = "none";
      setStatus(statusEl, "MONITORING", "ok");
      break;

    case MODE.TERMINATED:
      appRoot.classList.add("exam-lock");
      setStatus(statusEl, "TERMINATED", "err");
      guideTextEl.style.display = "block";
      const finalReason =
        reason || sessionState.terminationReason || "Session Ended";
      guideTextEl.innerHTML = `EXAM TERMINATED<br><small style="color: #fca5a5;">${finalReason}</small>`;

      endBtn.disabled = true;
      startBtn.disabled = false;

      if (video.srcObject) {
        video.srcObject.getTracks().forEach((t) => t.stop());
        video.srcObject = null;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      break;
  }
}

/* ===================== MAIN LOOP ===================== */
function processFrames() {
  if (sessionState.mode === MODE.TERMINATED || video.ended) {
    applySessionEffects(sessionState.terminationReason);
    return;
  }
  requestAnimationFrame(processFrames);

  const now = performance.now();
  if (now - sessionState.lastProcessTime < FRAME_INTERVAL) return;
  sessionState.lastProcessTime = now;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  applySessionEffects();

  // FPS counter
  sessionState.frameCount++;
  if (now - sessionState.lastFpsTime >= 1000) {
    fpsEl.textContent = sessionState.frameCount;
    sessionState.frameCount = 0;
    sessionState.lastFpsTime = now;
  }

  const results = detectFace(faceLandmarker, video, now);
  const faceDetected =
    results?.facialTransformationMatrixes?.length &&
    results?.faceLandmarks?.length;

  switch (sessionState.mode) {
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

/* ===================== HELPERS ===================== */
function scaleBoxToCanvas(normBox) {
  return {
    x: normBox.x * canvas.width,
    y: normBox.y * canvas.height,
    w: normBox.w * canvas.width,
    h: normBox.h * canvas.height,
  };
}

/* ===================== CALIBRATION ===================== */
function handleCalibration(faceDetected, results, now) {
  guideTextEl.style.display = "block";
  if (!faceDetected) {
    guideTextEl.textContent = "CENTER YOUR FACE";
    setStatus(statusEl, "SHOW YOUR FACE", "warn");
    drawBoundingBox(ctx, canvas, null, false, true);
    return;
  }

  const landmarks = results.faceLandmarks[0];
  const matrix = Array.from(results.facialTransformationMatrixes[0].data);
  const pose = calculateHeadPose(matrix);
  const box = scaleBoxToCanvas(getBoundingBox(landmarks));

  const result = runCalibration({
    now,
    pose,
    box,
    canvas,
    calibrationStartTime: sessionState.calibrationStartTime,
  });

  switch (result.state) {
    case "INVALID_POSITION":
      sessionState.calibrationStartTime = null;
      guideTextEl.textContent = "CENTER YOUR FACE";
      setStatus(statusEl, result.violations.join("\n"), "warn");
      break;

    case "HOLDING":
      sessionState.calibrationStartTime = result.calibrationStartTime;
      guideTextEl.textContent = `HOLD STILL...`; // Countdown text
      setStatus(
        statusEl,
        `CALIBRATING... ${Math.ceil(result.remainingMs / 1000)}s`,
        "warn"
      );
      break;

    case "CALIBRATED":
      sessionState.basePose = result.basePose;
      sessionState.mode = MODE.MONITORING;
      sessionState.calibrationStartTime = null;
      setStatus(statusEl, "STARTED", "ok");
      guideTextEl.style.display = "none";
      break;
  }

  drawBoundingBox(ctx, canvas, box, false, true);
}

/* ===================== MONITORING ===================== */
function handleMonitoring(faceDetected, results, now) {
  if (!faceDetected) {
    const { shouldLog } = handleFaceMissing(now);

    updateTotalLostUI(sessionState.totalMissingMs, totalLostEl);
    setStatus(statusEl, "FACE LOST - PLEASE RETURN", "warn");

    updateDisplayMetrics(
      { yaw: 0, pitch: 0, roll: 0 },
      { yaw: yawEl, pitch: pitchEl, roll: rollEl }
    );

    drawBoundingBox(ctx, canvas, null, false, false);

    if (shouldLog) {
      sessionState.stats.violations++;
      violationsEl.textContent = sessionState.stats.violations;
      const missingImage = captureViolationImage(video, null);
      sendMonitoringData(null, ["Face Missing"], missingImage);
    }
    return;
  }

  sessionState.lastMissingTick = null;
  sessionState.missingSince = null;
  sessionState.stabilityMs = 0;

  const landmarks = results.faceLandmarks[0];
  const matrix = Array.from(results.facialTransformationMatrixes[0].data);
  const pose = calculateHeadPose(matrix);
  const box = scaleBoxToCanvas(getBoundingBox(landmarks));

  const relativePose = {
    yaw: -(pose.yaw - sessionState.basePose.yaw),
    pitch: pose.pitch - sessionState.basePose.pitch,
    roll: pose.roll - sessionState.basePose.roll,
  };

  updateDisplayMetrics(relativePose, {
    yaw: yawEl,
    pitch: pitchEl,
    roll: rollEl,
  });

  const violations = detectViolations(relativePose, box);
  if (violations.length > 0) {
    if (now - sessionState.lastViolationLogTime >= 1000) {
      sessionState.lastViolationLogTime = now;

      const violationImage = captureViolationImage(video, box);

      sessionState.stats.violations++;
      violationsEl.textContent = sessionState.stats.violations;

      sendMonitoringData(relativePose, violations, violationImage);
    }
  }
  drawBoundingBox(ctx, canvas, box, violations.length > 0, false);

  sessionState.spoofFrameCounter++;
  if (sessionState.spoofFrameCounter % SPOOF_SEND_INTERVAL === 0) {
    sendFaceCrop(box);
  }
}

/* ===================== LOCKED ===================== */
function handleLocked(faceDetected, results, now) {
  setStatus(statusEl, "SESSION LOCKED", "err");

  if (!faceDetected) {
    const { shouldLog } = handleFaceMissing(now);

    if (shouldLog) {
      sessionState.stats.violations++;
      violationsEl.textContent = sessionState.stats.violations;

      const image = captureViolationImage(video, null);
      sendMonitoringData(null, ["Face Missing (Locked)"], image);
    }

    sessionState.stabilityMs = 0;
    updateTotalLostUI(sessionState.totalMissingMs, totalLostEl);

    guideTextEl.style.display = "block";
    guideTextEl.textContent = "FACE REQUIRED TO UNLOCK";

    drawBoundingBox(ctx, canvas, null, false, false);
    return;
  }

  sessionState.stabilityMs += FRAME_INTERVAL;
  const remaining = Math.max(0, TIER.STABILITY - sessionState.stabilityMs);

  guideTextEl.textContent = `HOLD STILL ${Math.ceil(remaining / 1000)}s`;

  if (sessionState.stabilityMs >= TIER.STABILITY) {
    unlockSession();
    sessionState.missingSince = null;
    sessionState.lastMissingTick = null;
  }
}

/* ===================== LIVENESS ===================== */
async function sendFaceCrop(box) {
  if (!livenessClient || !box) return;

  const mini = await generateMiniCrops(video, box);
  const facebag = await generateFaceBagnetCrop(video, box);

  if (mini && facebag) {
    livenessClient.sendCrops(mini, facebag);
  }
}

/* ===================== BOOTSTRAP ===================== */
startBtn.onclick = async () => {
  startBtn.disabled = true;
  endBtn.disabled = false;
  resetSessionState();
  sessionState.mode = MODE.CALIBRATION;

  guideTextEl.textContent = "INITIALIZING...";
  guideTextEl.style.display = "block";

  faceLandmarker = await initFaceLandmarker(
    scriptTag.dataset.modelPath,
    scriptTag.dataset.wasmPath
  );

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
  });
  sessionState.wsLog = new WebSocket(
    WS_LOG_URL.replace("{sessionId}", crypto.randomUUID())
  );

  sessionState.wsLog.onopen = () => {
    console.log("Log WebSocket connected");
  };

  sessionState.wsLog.onerror = (e) => {
    console.error("Log WebSocket error", e);
  };

  sessionState.wsLog.onclose = () => {
    console.warn("Log WebSocket closed");
  };
  video.srcObject = stream;
  video.onloadedmetadata = () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    video.play();
    processFrames();
  };

  livenessClient = new LivenessClient(
    WS_LIVENESS_URL.replace("{sessionId}", crypto.randomUUID())
  );

  livenessClient.onMessage((data) => {
    spoofScoreEl.textContent =
      typeof data.spoof_score === "number" ? data.spoof_score.toFixed(3) : "--";

    livenessStatusEl.textContent = data.liveness
      ? data.liveness.toUpperCase()
      : "--";
    console.log("Received liveness data:", data);
    if (!useBackendLiveness) return;
    handleLivenessResult(data, performance.now());
  });
};

endBtn.onclick = () => {
  terminateSession("User ended session");
  applySessionEffects("User ended session");
  livenessClient?.close();
  sessionState.wsLog?.close();
  sessionState.wsLog = null;

};

const toggleLivenessBtn = document.getElementById("toggleLivenessBtn");

toggleLivenessBtn.onclick = () => {
  useBackendLiveness = !useBackendLiveness;

  toggleLivenessBtn.textContent =
    `Liveness: ${useBackendLiveness ? "ON" : "OFF"}`;

  toggleLivenessBtn.style.background =
    useBackendLiveness ? "#22c55e" : "#6b7280";
};

