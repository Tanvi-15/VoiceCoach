from __future__ import annotations
from typing import Dict, Any
import re

FILLERS = {"um","uh","erm","er","like","you","know","sort","of","kind","of","you know","sort of","kind of"}

class Metrics:
    def __init__(self):
        pass

    def compute(self, asr: Dict[str,Any], praat: Dict[str,Any], lib: Dict[str,Any], vad: Dict[str,Any]) -> Dict[str,Any]:
        # --- ASR text features ---
        text = asr.get("text", "") or ""
        words = [w for w in re.findall(r"[A-Za-z']+", text.lower())]
        dur_sec = float(praat.get("duration_sec", 0.0)) or float(asr.get("duration", 0.0)) or 0.0
        wpm = self._words_per_min(len(words), dur_sec) if dur_sec > 0 else 0.0

        filler_count = sum(1 for w in words if w in FILLERS)
        filler_ratio = filler_count / (len(words) + 1e-8)

        # --- Voice quality proxies (robust to missing keys) ---
        jitter = float(praat.get("jitter_local", 0.0) or 0.0)
        shimmer = float(praat.get("shimmer_local", 0.0) or 0.0)
        hnr = float(praat.get("hnr_mean_db", 0.0) or 0.0)

        # *Clarity index* (higher is better): penalize jitter & shimmer, modestly reward energy
        rms_mean = float(lib.get("rms_mean", 0.0) or 0.0)
        clarity_index = max(0.0, 1.0 - 3.5*jitter - 2.5*shimmer) * (1.0 + 0.05*rms_mean)

        # Tone/engagement proxy: pitch & intensity variability
        f0_std = float(praat.get("f0_std_hz", 0.0) or 0.0)
        intensity_std = float(praat.get("intensity_std_db", 0.0) or 0.0)
        tone_var = f0_std + 0.6*intensity_std

        # Pacing & flow: from WPM and VAD pauses
        pacing_score = self._pacing_score(wpm)
        pause_ratio = float(vad.get("pause_ratio", 0.0) or 0.0)

        return {
            "duration_sec": dur_sec,
            "word_count": len(words),
            "wpm": float(wpm),
            "filler_count": int(filler_count),
            "filler_ratio": float(filler_ratio),
            "clarity_index": float(clarity_index),
            "tone_variability": float(tone_var),
            "pacing_score": float(pacing_score),
            "pause_ratio": float(pause_ratio),
            "pitch_std_hz": f0_std,
            "pitch_range_hz": float(praat.get("f0_range_hz", 0.0) or 0.0),
            "intensity_std_db": intensity_std,
            "hnr_mean_db": hnr,
            "jitter_local": jitter,
            "shimmer_local": shimmer,
        }

    # ---------- helpers ----------
    def _words_per_min(self, n_words: int, dur_sec: float) -> float:
        if dur_sec <= 0:
            return 0.0
        return n_words / (dur_sec / 60.0)

    def _pacing_score(self, wpm: float) -> float:
        # Optimal comprehension often ~140–170 WPM
        if wpm <= 0:
            return 0.0
        if 140 <= wpm <= 170:
            return 1.0
        if wpm < 140:
            # 60–140 maps to 0–1
            return max(0.0, 1.0 - (140 - wpm)/80.0)
        # 170–300 maps to 1–0
        return max(0.0, 1.0 - (wpm - 170)/130.0)
