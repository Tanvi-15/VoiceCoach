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

    # --- helpers ---
    def _clarity(self, m):
        ci = m.get("clarity_index",0)
        if ci >= 1.2: s = 5
        elif ci >= 0.9: s = 4
        elif ci >= 0.6: s = 3
        elif ci >= 0.3: s = 2
        else: s = 1
        return {"score": s, "why": f"clarity_index={ci:.2f} (jitter/shimmer/HNR proxy)."}

    def _confidence(self, m):
        pr = m.get("pause_ratio",0.0)
        istd = m.get("intensity_std_db",0.0)
        # Slightly stronger penalty for lots of silence; modest reward for dynamic intensity
        conf = max(0.0, 1.1 - 1.8*pr + 0.03*istd)
        s = 1 + int(min(4, max(0, conf*4)))
        return {"score": s, "why": f"pause_ratio={pr:.2f}, intensity_std={istd:.2f} dB."}

    def _tone(self, m):
        tv = m.get("tone_variability",0.0)
        if tv >= 60: s=5
        elif tv >= 40: s=4
        elif tv >= 25: s=3
        elif tv >= 12: s=2
        else: s=1
        return {"score": s, "why": f"tone_variability={tv:.1f} (pitch+intensity var)."}

    def _pacing(self, m):
        ps = m.get("pacing_score",0.0)
        if ps >= 0.95: s=5
        elif ps >= 0.75: s=4
        elif ps >= 0.5: s=3
        elif ps >= 0.25: s=2
        else: s=1
        return {"score": s, "why": f"wpm={m.get('wpm',0):.0f} → pacing_score={ps:.2f}."}

    def _engagement(self, m):
        fr = m.get("filler_ratio",0.0)
        prng = m.get("pitch_range_hz",0.0)
        score = max(0.0, 1.1 - 2.0*fr + min(prng, 150)/300)
        s = 1 + int(min(4, max(0, score*4)))
        return {"score": s, "why": f"filler_ratio={fr:.2%}, pitch_range={prng:.1f} Hz."}

    def _cadence(self, m):
        """
        Cadence now rewards 'ideal' pauses and penalizes short/long ('bad') pauses.
        """
        pr  = m.get("pause_ratio", 0.0)
        gpr = m.get("good_pause_ratio", 0.0)   # fraction of pause time in 0.25–0.60s
        bpr = m.get("bad_pause_ratio", 0.0)    # fraction of pause time <0.20s or >=1.0s

        # Base from overall silence (lower better), then adjust by good/bad mix
        val = max(0.0, 1.0 - 1.2*pr) + 0.5*gpr - 0.7*bpr
        # Map to 1..5
        if val >= 1.0: s=5
        elif val >= 0.75: s=4
        elif val >= 0.5: s=3
        elif val >= 0.25: s=2
        else: s=1
        why = f"pause_ratio={pr:.2f}, good_pause_ratio={gpr:.2f}, bad_pause_ratio={bpr:.2f}."
        return {"score": s, "why": why}

    def _flow(self, m):
        """
        Flow blends pacing with pause quality.
        """
        ps  = m.get("pacing_score", 0.0)
        gpr = m.get("good_pause_ratio", 0.0)
        bpr = m.get("bad_pause_ratio", 0.0)
        val = ps + 0.3*gpr - 0.4*bpr
        if val >= 1.1: s=5
        elif val >= 0.85: s=4
        elif val >= 0.6: s=3
        elif val >= 0.35: s=2
        else: s=1
        return {"score": s, "why": f"pacing_score={ps:.2f}, good_pause_ratio={gpr:.2f}, bad_pause_ratio={bpr:.2f}."}
