from __future__ import annotations
from typing import Dict, Any, List, Tuple

class PauseAnalyzer:
    """
    Compute pauses from Whisper word timestamps and bucket them:

      short:  [min_pause, short_max)
      ideal:  [good_min,  good_max]     <-- rewarded
      medium: (good_max,  long_min)     <-- neutral
      long:   [long_min,  +inf)         <-- penalized

    Defaults are reasonable for presentation speech.
    """
    def __init__(
        self,
        min_pause_sec: float = 0.12,
        short_max: float = 0.20,
        good_min: float = 0.25,
        good_max: float = 0.60,
        long_min: float = 1.00,
    ):
        self.min_pause = float(min_pause_sec)
        self.short_max = float(short_max)
        self.good_min  = float(good_min)
        self.good_max  = float(good_max)
        self.long_min  = float(long_min)

    def analyze(self, asr_out: Dict[str, Any], total_duration: float) -> Dict[str, Any]:
        words = list(asr_out.get("words", []))
        words.sort(key=lambda w: float(w.get("start", 0.0)))

        pauses: List[Tuple[float, float]] = []
        prev_end = 0.0
        for w in words:
            start = float(w.get("start", 0.0))
            end   = float(w.get("end",   start))
            if start - prev_end >= self.min_pause:
                pauses.append((prev_end, start))
            prev_end = max(prev_end, end)
        if total_duration - prev_end >= self.min_pause:
            pauses.append((prev_end, float(total_duration)))

        # Core aggregates
        pause_dur = sum(b - a for a, b in pauses)
        speech_dur = max(0.0, float(total_duration) - pause_dur)
        pause_ratio = pause_dur / (float(total_duration) + 1e-8)

        # Histogram bins (by duration)
        bins = {"short": {"count": 0, "sec": 0.0},
                "ideal": {"count": 0, "sec": 0.0},
                "medium":{"count": 0, "sec": 0.0},
                "long":  {"count": 0, "sec": 0.0}}
        for a, b in pauses:
            d = b - a
            if d < self.short_max:
                bins["short"]["count"] += 1; bins["short"]["sec"] += d
            elif self.good_min <= d <= self.good_max:
                bins["ideal"]["count"] += 1; bins["ideal"]["sec"] += d
            elif d >= self.long_min:
                bins["long"]["count"]  += 1; bins["long"]["sec"]  += d
            else:
                bins["medium"]["count"] += 1; bins["medium"]["sec"] += d

        total_pause_sec = max(pause_dur, 1e-8)
        good_pause_ratio = bins["ideal"]["sec"] / total_pause_sec
        bad_pause_ratio  = (bins["short"]["sec"] + bins["long"]["sec"]) / total_pause_sec

        return {
            "speech_duration_sec": float(speech_dur),
            "pause_duration_sec":  float(pause_dur),
            "pause_ratio":         float(pause_ratio),
            "pause_count":         int(len(pauses)),
            "pauses":              pauses,      # list of [start, end]
            "pause_bins":          bins,        # counts + seconds per bin
            "good_pause_ratio":    float(good_pause_ratio),
            "bad_pause_ratio":     float(bad_pause_ratio),
            "thresholds": {
                "min_pause": self.min_pause,
                "short_max": self.short_max,
                "good_min":  self.good_min,
                "good_max":  self.good_max,
                "long_min":  self.long_min,
            }
        }
