/**
 * Violation logging utilities.
 * Responsible only for payload creation and websocket transmission.
 * NO DOM access, NO UI mutation.
 */

import { sessionState } from "../state/sessionState.js";

/**
 * Build base monitoring payload.
 * @param {object|null} relativePose
 * @param {string[]} violations
 */
export function buildBasePayload(relativePose, violations) {
  return {
    type: "metadata",
    timestamp: Math.round(performance.now()),
    pose: relativePose || { yaw: 0, pitch: 0, roll: 0 },
    violations: violations || [],
    image: null
  };
}

export function sendMonitoringData(pose, violations, imageBase64) {
  const payload = {
    type: "metadata",
    timestamp: Math.round(performance.now()),
    pose: pose || { yaw: 0, pitch: 0, roll: 0 },
    violations: violations || [],
    image: imageBase64 || null
  };
  if (
    sessionState.wsLog &&
    sessionState.wsLog.readyState === WebSocket.OPEN
  ) {
    sessionState.wsLog.send(JSON.stringify(payload));
  }
}

