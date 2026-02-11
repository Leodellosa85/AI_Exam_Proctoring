// ===================== Timing =====================
export const TARGET_FPS = 8;
export const FRAME_INTERVAL = 1000 / TARGET_FPS;

// ===================== Session Modes =====================
export const MODE = {
  CALIBRATION: "calibration",
  MONITORING: "monitoring",
  LOCKED: "locked",
  TERMINATED: "terminated"
};

// ===================== Geometry & Pose =====================
export const SAFE_ZONE = { xMin: 0.2, xMax: 0.8, yMin: 0.1, yMax: 0.9 };

export const ANGLE_LIMITS = {
  PITCH_DOWN: 15,
  PITCH_UP: -25,
  YAW_LEFT: 30,
  YAW_RIGHT: -30,
  ROLL_LEFT: 25,
  ROLL_RIGHT: -25
};

// ===================== Liveness Crop Config =====================
export const MINI_INPUT_SIZE = 80;
export const FACEBAG_INPUT_SIZE = 96;

export const MAX_SCALE_V2 = 2.7;
export const MAX_SCALE_V1SE = 4.0;
export const MAX_SCALE_FACEBAG = 2.2;
export const MIN_SCALE = 1.3;

export const MIN_FACE_SIZE = 100;

// ===================== Policy Thresholds =====================
export const TIER = {
  FLAG: 3000,
  BLUR: 5000,
  TERMINATE: 15000,
  CUMULATIVE: 30000,
  STABILITY: 3000
};

export const SPOOF_POLICY = {
  SUSPICIOUS_LOCK_MS: 3000,
  FAKE_TERMINATE_MS: 1500
};

export const SPOOF_SEND_INTERVAL = 10;

export const WS_LOG_URL =
  "ws://127.0.0.1:8002/ws/v1/ws/{sessionId}";

export const WS_LIVENESS_URL =
  "ws://127.0.0.1:8001/ws2/{sessionId}";

