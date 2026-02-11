import { MODE } from "../config.js";

/**
 * Centralized runtime session state.
 * Single source of truth for modes, timers, and counters.
 */
export const sessionState = {
  // ---- Session mode ----
  mode: MODE.CALIBRATION,

  // ---- Pose baseline ----
  basePose: { yaw: 0, pitch: 0, roll: 0 },

  // ---- Face missing / lock timers ----
  missingSince: null,
  stabilityMs: 0,
  totalMissingMs: 0,

  // ---- Frame timing ----
  lastProcessTime: 0,
  lastFpsTime: 0,
  frameCount: 0,

  // ---- Calibration ----
  calibrationStartTime: null,

  // ---- Violations ----
  lastViolationTrigger: 0,
  lastViolationLogTime: 0,
  lastMissingTick: 0,

  stats: {
    violations: 0
  },

  // ---- Liveness / spoofing ----
  spoofFrameCounter: 0,
  spoofStatus: "real",
  spoofSince: null,

  // ---- Termination ----
  terminationReason: null,

  wsLog: null
};

/**
 * Reset session state to initial values.
 * Called on exam start / restart.
 */
export function resetSessionState() {
  sessionState.mode = MODE.CALIBRATION;

  sessionState.basePose = { yaw: 0, pitch: 0, roll: 0 };

  sessionState.missingSince = null;
  sessionState.stabilityMs = 0;
  sessionState.totalMissingMs = 0;

  sessionState.lastProcessTime = 0;
  sessionState.lastFpsTime = 0;
  sessionState.frameCount = 0;

  sessionState.calibrationStartTime = null;

  sessionState.lastViolationTrigger = 0;
  sessionState.lastViolationLogTime = 0;
  sessionState.stats.violations = 0;

  sessionState.spoofFrameCounter = 0;
  sessionState.spoofStatus = "real";
  sessionState.spoofSince = null;

  sessionState.terminationReason = null;

  sessionState.wsLog = null;
}
