/**
 * Session state transitions only.
 * No DOM, no media, no WebSocket access.
 */

import { sessionState } from "../state/sessionState.js";
import { MODE } from "../config.js";

/**
 * Lock the session due to violation or spoofing.
 */
export function lockSession() {
  if (sessionState.mode === MODE.LOCKED) return;

  sessionState.mode = MODE.LOCKED;
  sessionState.stabilityMs = 0;
}

/**
 * Unlock the session after stability period.
 */
export function unlockSession() {
  sessionState.mode = MODE.MONITORING;
  sessionState.missingSince = null;
  sessionState.stabilityMs = 0;
}

/**
 * Mark the session as terminated.
 */
export function terminateSession(reason) {
  sessionState.mode = MODE.TERMINATED;
  sessionState.terminationReason = reason;
}
