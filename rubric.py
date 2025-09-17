from __future__ import annotations
from typing import Dict, Any

class RubricScorer:
    def __init__(self):
        pass

    def score(self, m: Dict[str,Any]) -> Dict[str,Any]:
        return {
            "Clarity": self._clarity(m),
            "Confidence": self._confidence(m),
            "Tone": self._tone(m),
            "Pacing": self._pacing(m),
            "Engagement": self._engagement(m),
            "Cadence": self._cadence(m),
            "Flow": self._flow(m),
        }
    
    def _clarity(self, m):
# Higher HNR, lower jitter/shimmer (already in clarity_index)
        ci = m.get("clarity_index",0)
        if ci >= 1.2: s = 5
        elif ci >= 0.9: s = 4
        elif ci >= 0.6: s = 3
        elif ci >= 0.3: s = 2
        else: s = 1
        why = f"clarity_index={ci:.2f} (jitter/shimmer/HNR proxy)."
        return {"score": s, "why": why}

    def _confidence(self, m):
        # Fewer long pauses and good intensity variation imply confidence
        pr = m.get("pause_ratio",0)
        istd = m.get("intensity_std_db",0)
        conf = max(0.0, 1.2 - 1.5*pr + 0.05*istd)
        s = 1 + int(min(4, max(0, conf*4)))
        why = f"pause_ratio={pr:.2f}, intensity_std={istd:.2f} dB."
        return {"score": s, "why": why}

    def _tone(self, m):
        tv = m.get("tone_variability",0)
        if tv >= 60: s=5
        elif tv >= 40: s=4
        elif tv >= 25: s=3
        elif tv >= 12: s=2
        else: s=1
        return {"score": s, "why": f"tone_variability={tv:.1f} (pitch+intensity var)."}

    def _pacing(self, m):
        ps = m.get("pacing_score",0)
        if ps >= 0.95: 
            s=5
        elif ps >= 0.75: 
            s=4
        elif ps >= 0.5: 
            s=3
        elif ps >= 0.25: 
            s=2
        else: 
            s=1
        return {"score": s, "why": f"wpm={m.get('wpm',0):.0f} → pacing_score={ps:.2f}."}

    def _engagement(self, m):
        # Low filler ratio and moderate pitch range can signal control/engagement
        fr = m.get("filler_ratio",0)
        prng = m.get("pitch_range_hz",0)
        score = max(0.0, 1.1 - 2.0*fr + min(prng, 150)/300)
        s = 1 + int(min(4, max(0, score*4)))
        why = f"filler_ratio={fr:.2%}, pitch_range={prng:.1f} Hz."
        return {"score": s, "why": why}

    def _cadence(self, m):
# Pause count/duration balance
        pr = m.get("pause_ratio",0)
        if pr <= 0.10: s=5
        elif pr <= 0.20: s=4
        elif pr <= 0.30: s=3
        elif pr <= 0.45: s=2
        else: s=1
        return {"score": s, "why": f"pause_ratio={pr:.2f}."}

    def _flow(self, m):
        # Combine pacing & pauses
        ps = m.get("pacing_score",0)
        pr = m.get("pause_ratio",0)
        val = ps - 0.5*pr
        if val >= 0.9: s=5
        elif val >= 0.7: s=4
        elif val >= 0.5: s=3
        elif val >= 0.3: s=2
        else: s=1
        return {"score": s, "why": f"pacing_score={ps:.2f}, pause_ratio={pr:.2f}."}
