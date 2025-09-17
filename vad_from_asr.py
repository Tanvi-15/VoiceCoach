from __future__ import annotations
from typing import Dict, Any, List, Tuple

class PauseAnalyzer:
    def __init__(self, min_pause_sec: float = 0.12):
        self.min_pause = float(min_pause_sec)

    def analyze(self, asr_out: Dict[str, Any], total_duration: float) -> Dict[str, Any]:
        """Compute pauses as gaps between consecutive ASR words.
        Expects `asr_out['words']` with `start`/`end` times (seconds) and overall `total_duration`.
        """
        words = list(asr_out.get("words", []))
        words.sort(key=lambda w: float(w.get("start", 0.0)))

        pauses: List[Tuple[float, float]] = []
        prev_end = 0.0
        for w in words:
            start = float(w.get("start", 0.0))
            end = float(w.get("end", start))
            if start - prev_end >= self.min_pause:
                pauses.append((prev_end, start))
            prev_end = max(prev_end, end)
        if total_duration - prev_end >= self.min_pause:
            pauses.append((prev_end, float(total_duration)))

        pause_dur = sum(b - a for a, b in pauses)
        speech_dur = max(0.0, float(total_duration) - pause_dur)
        return {
            "speech_duration_sec": float(speech_dur),
            "pause_duration_sec": float(pause_dur),
            "pause_ratio": float(pause_dur / (float(total_duration) + 1e-8)),
            "pause_count": int(len(pauses)),
            "pauses": pauses,
        }