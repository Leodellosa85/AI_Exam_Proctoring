/**
 * Timers related to face-missing and cumulative absence handling.
 * PURE LOGIC ONLY — no DOM, no media, no sockets.
 */

import { sessionState } from "../state/sessionState.js";
import { TIER, FRAME_INTERVAL } from "../config.js";
import { lockSession, terminateSession } from "../state/sessionActions.js";

/**
 * Increment cumulative missing time and enforce termination threshold.
 */
export function addCumulativeMissingTime(deltaMs) {
  sessionState.totalMissingMs += deltaMs;

  if (sessionState.totalMissingMs >= TIER.CUMULATIVE) {
    sessionState.terminationReason =
      "Total off-screen / locked time exceeded";
    terminateSession("Total time off-screen exceeded limit (30s)");
    return { shouldLog: true, missingMs: sessionState.totalMissingMs };
  }
}

/**
 * Handle logic when no face is detected.
 * @returns {{ shouldLog: boolean }}
 */
export function handleFaceMissing(now) {
  if (!sessionState.missingSince) {
    sessionState.missingSince = now;
    sessionState.lastViolationTrigger = 0;
    sessionState.lastMissingTick = now;
    return { shouldLog: false, missingMs: 0 };
  }

  const delta = now - (sessionState.lastMissingTick || now);
  sessionState.lastMissingTick = now;

  const missingMs = now - sessionState.missingSince;

  addCumulativeMissingTime(delta);

  if (missingMs >= TIER.TERMINATE) {
    sessionState.terminationReason = "Face missing for more than 15 seconds";
    terminateSession("Session abandoned: Face missing > 15s");
    return { shouldLog: true, missingMs };
  }

  let shouldLog = false;
  if (missingMs >= TIER.FLAG) {
    if (!sessionState.lastViolationTrigger || (now - sessionState.lastViolationTrigger >= 1000)) {
      sessionState.lastViolationTrigger = now;
      shouldLog = true;
    }
  }

  if (missingMs >= TIER.BLUR) {
    lockSession();
  }

  return { shouldLog, missingMs };
}
