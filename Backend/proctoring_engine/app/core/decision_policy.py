from collections import deque
from typing import Dict, Any


class DecisionPolicy:
    def __init__(
        self,
        window: int = 5,
        # If Realness is below 85%, it's definitely a phone screen
        facebagnet_gate: float = 0.85,      
        # Thresholds for combined 'SPOOFINESS' (1.0 - Realness)
        # Real face (0.04 spoofiness) will pass 0.07.
        # Phone attack (0.14 spoofiness) will fail 0.07.
        suspicious_threshold: float = 0.07, 
        fake_threshold: float = 0.12,
    ):
        self.window = window
        self.facebagnet_gate = facebagnet_gate
        self.suspicious_threshold = suspicious_threshold
        self.fake_threshold = fake_threshold

    def update(self, scores: Dict[str, float], session: Dict) -> Dict[str, Any]:
        history = session.setdefault("spoof_history", deque(maxlen=self.window))

        fb_score = scores.get("facebagnet")
        mini_score = scores.get("minifasnet", 0)

        # 1. HARD GATE
        if fb_score is not None and fb_score < self.facebagnet_gate:
            history.append(1.0)
            return self._result("fake", 1.0, scores, history, reason="facebagnet_gate")

        # 2. CONVERT TO SPOOFINESS
        # FaceBagNet: Low score is bad. (1 - score) = Spoofiness
        # MiniFASNet: High score is bad. (already Spoofiness)
        fb_spoofiness = (1.0 - fb_score) if fb_score is not None else 0
        
        current_frame_spoof = max(mini_score, fb_spoofiness)
        history.append(current_frame_spoof)

        avg = sum(history) / len(history)

        # 3. DECISION BASED ON AVERAGE
        if avg >= self.fake_threshold:
            liveness = "fake"
        elif avg >= self.suspicious_threshold:
            liveness = "suspicious"
        else:
            liveness = "real"

        return self._result(liveness, avg, scores, history)
    
    def _result(self, liveness, score, scores, history, reason=None):
        return {
            "liveness": liveness,
            "spoof_score": round(score, 4) if score is not None else None,
            "components": {k: round(v, 4) for k, v in scores.items()},
            "history_avg": round(sum(history) / len(history), 4) if history else None,
            "frames": len(history),
            "reason": reason,
        }

