/**
 * ----------------------------------------------------
 * Pure calibration logic for face positioning and stability.
 *
 * This module:
 * - Validates face position against safe zone
 * - Tracks calibration hold time
 * - Emits calibration state transitions
 *
 * It does NOT:
 * - Manipulate DOM
 * - Mutate global session state
 * - Perform rendering
 */

import { MODE, SAFE_ZONE, TIER } from "../config.js";

/**
 * Run calibration step.
 *
 * @param {Object} params
 * @param {number} params.now - Current timestamp (performance.now)
 * @param {Object} params.pose - Head pose { yaw, pitch, roll }
 * @param {Object} params.box - Face bounding box { x, y, w, h }
 * @param {Object} params.canvas - Canvas reference for normalization
 * @param {number|null} params.calibrationStartTime - Existing start time
 *
 * @returns {Object} Calibration result
 */
export function runCalibration({
  now,
  pose,
  box,
  canvas,
  calibrationStartTime
}) {
  if (!box) {
    return { state: "NO_FACE" };
  }

  const faceCX = (box.x + box.w / 2) / canvas.width;
  const faceCY = (box.y + box.h / 2) / canvas.height;

  const violations = [];

  if (faceCX < SAFE_ZONE.xMin) violations.push("Too far Right");
  if (faceCX > SAFE_ZONE.xMax) violations.push("Too far Left");
  if (faceCY < SAFE_ZONE.yMin) violations.push("Too High");
  if (faceCY > SAFE_ZONE.yMax) violations.push("Too Low");

  if (violations.length > 0) {
    return {
      state: "INVALID_POSITION",
      violations,
      resetTimer: true
    };
  }

  const startTime = calibrationStartTime ?? now;
  const elapsed = now - startTime;
  const remainingMs = Math.max(0, TIER.STABILITY - elapsed);

  if (elapsed < TIER.STABILITY) {
    return {
      state: "HOLDING",
      remainingMs,
      calibrationStartTime: startTime
    };
  }

  return {
    state: "CALIBRATED",
    mode: MODE.MONITORING,
    basePose: pose,
    calibrationStartTime: null
  };
}
