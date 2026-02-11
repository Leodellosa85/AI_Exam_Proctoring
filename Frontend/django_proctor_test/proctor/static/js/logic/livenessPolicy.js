import { sessionState } from "../state/sessionState.js";
import { MODE, SPOOF_POLICY } from "../config.js";
import { lockSession, terminateSession } from "../state/sessionActions.js";

export function handleLivenessResult(data, now) {
  if (!data || !data.liveness) return;

  const newStatus = data.liveness;

  if (newStatus !== sessionState.spoofStatus) {
    sessionState.spoofStatus = newStatus;
    sessionState.spoofSince = now;
  }

  const duration = now - sessionState.spoofSince;

  if (newStatus === "suspicious") {
    if (duration >= SPOOF_POLICY.SUSPICIOUS_LOCK_MS &&
        sessionState.mode === MODE.MONITORING) {
      lockSession();
    }
    return;
  }

  if (newStatus === "fake") {
    if (duration >= SPOOF_POLICY.FAKE_TERMINATE_MS) {
      terminateSession("Spoofing / Fake face detected");
    }
  }
}
