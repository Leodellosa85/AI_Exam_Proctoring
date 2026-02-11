from collections import deque
from typing import Dict, Any


class DecisionPolicy:
    """
    Spoof decision policy with:
    - Model-aware fusion
    - Hard spoof gates
    - Temporal smoothing
    """

    def __init__(
        self,
        window: int = 5,
        facebagnet_gate: float = 0.20,
        suspicious_threshold: float = 0.12,
        fake_threshold: float = 0.18,
    ):
        self.window = window
        self.facebagnet_gate = facebagnet_gate
        self.suspicious_threshold = suspicious_threshold
        self.fake_threshold = fake_threshold

    def update(self, scores: Dict[str, float], session: Dict) -> Dict[str, Any]:
        """
        Update decision using current frame scores.

        Args:
            scores: {"minifasnet": float, "facebagnet": float}
            session: session dict (persistent across frames)

        Returns:
            dict with liveness decision and metadata
        """

        history = session.setdefault(
            "spoof_history",
            deque(maxlen=self.window)
        )

        # ------------------------------
        # Sanity check
        # ------------------------------
        if not scores:
            return self._result("unstable", None, scores, history)

        # ------------------------------
        # Hard photo-attack gate
        # FaceBagNet is trusted for photos
        # ------------------------------
        facebag_score = scores.get("facebagnet")
        if facebag_score is not None and facebag_score >= self.facebagnet_gate:
            history.append(facebag_score)
            return self._result("fake", facebag_score, scores, history, reason="facebagnet_gate")

        # ------------------------------
        # Fusion logic (MAX, not AVG)
        # Prevent weak models from suppressing strong ones
        # ------------------------------
        combined = max(scores.values())
        history.append(combined)

        avg = sum(history) / len(history)

        # ------------------------------
        # Temporal decision
        # ------------------------------
        if avg >= self.fake_threshold:
            liveness = "fake"
        elif avg >= self.suspicious_threshold:
            liveness = "suspicious"
        else:
            liveness = "real"

        return self._result(liveness, avg, scores, history)

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _result(self, liveness, score, scores, history, reason=None):
        return {
            "liveness": liveness,
            "spoof_score": round(score, 4) if score is not None else None,
            "components": {k: round(v, 4) for k, v in scores.items()},
            "history_avg": round(sum(history) / len(history), 4) if history else None,
            "frames": len(history),
            "reason": reason,
        }
