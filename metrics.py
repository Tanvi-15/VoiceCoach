from __future__ import annotations
from typing import Dict, Any
import re
from nlp_features import NLPFeatures

FILLERS = {"um","uh","erm","er","like","you","know","sort","of","kind","of","you know","sort of","kind of"}

class Metrics:
    def __init__(self):
        self.nlp_features = NLPFeatures()

    def compute(self, asr: Dict[str,Any], praat: Dict[str,Any], lib: Dict[str,Any], vad: Dict[str,Any]) -> Dict[str,Any]:
        # ASR text features
        # Tokenization: extracts only alphabetic/apostrophe tokens → list of words-> words it gets from whisper transcripts
        text = asr.get("text", "") or ""
        words = [w for w in re.findall(r"[A-Za-z']+", text.lower())]
        # Duration: combines Praat and ASR duration, ensuring at least 1µs for calculations.
        dur_sec = float(praat.get("duration_sec", 0.0) or asr.get("duration", 0.0) or 0.0)
        # WPM: counts words per minute, with smoothing for very short durations.
        wpm = self._words_per_min(len(words), dur_sec) if dur_sec > 0 else 0.0

        # Filler ratio: counts filler words (from predefined list) and normalizes by total word count.
        filler_count = sum(1 for w in words if w in FILLERS)
        filler_ratio = filler_count / (len(words) + 1e-8)
    

        # Voice quality proxies
        jitter = float(praat.get("jitter_local", 0.0) or 0.0)
        shimmer = float(praat.get("shimmer_local", 0.0) or 0.0)
        hnr = float(praat.get("hnr_mean_db", 0.0) or 0.0)
        cpps = float(praat.get("cpps_smooth_db", 0.0) or 0.0)

        rms_mean = float(lib.get("rms_mean", 0.0) or 0.0)
        f0_std = float(praat.get("f0_std_hz", 0.0) or 0.0)
        f0_range = float(praat.get("f0_range_hz", 0.0) or 0.0)
        intensity_std = float(praat.get("intensity_std_db", 0.0) or 0.0)
        intensity_mean = float(praat.get("intensity_mean_db", 0.0) or 0.0)

        # Stability features
        rate_var_win = float(praat.get("rate_var_win", 0.0) or 0.0)
        f0_var_win = float(praat.get("f0_var_win", 0.0) or 0.0)
        intensity_var_win = float(praat.get("intensity_var_win", 0.0) or 0.0)

        # Contour features
        final_fall_ratio = float(praat.get("final_fall_ratio", 0.0) or 0.0)
        final_rise_ratio = float(praat.get("final_rise_ratio", 0.0) or 0.0)

        # Pause stats (new histogram-aware fields)
        pause_ratio = float(vad.get("pause_ratio", 0.0) or 0.0)
        bins = vad.get("pause_bins", {}) or {}
        good_pause_ratio = float(vad.get("good_pause_ratio", 0.0) or 0.0)
        bad_pause_ratio  = float(vad.get("bad_pause_ratio", 0.0) or 0.0)

        # Extract NLP features
        nlp_features = self.nlp_features.extract_all_features(text, dur_sec, pause_ratio)
        
        # Research-aligned composite indices
        clarity_index = self._compute_clarity_index(jitter, shimmer, hnr, cpps, rms_mean, nlp_features["articulation_rate"])
        tone_variability = self._compute_tone_variability(f0_std, f0_range, intensity_std, final_fall_ratio, final_rise_ratio)
        pacing_score = self._compute_pacing_score(wpm, pause_ratio, good_pause_ratio, bad_pause_ratio)

        short_count = int(bins.get("short", {}).get("count", 0))
        ideal_count = int(bins.get("ideal", {}).get("count", 0))
        medium_count = int(bins.get("medium", {}).get("count", 0))
        long_count = int(bins.get("long", {}).get("count", 0))

        # Rates per minute (optional but handy for dashboards)
        mins = max(dur_sec, 1e-6) / 60.0
        short_rate_pm = short_count / mins
        long_rate_pm  = long_count / mins

        return {
            "duration_sec": dur_sec,
            "word_count": len(words),
            "wpm": float(wpm),
            "filler_count": int(filler_count),
            "filler_ratio": float(filler_ratio),

            # Research-aligned composite indices
            "clarity_index": float(clarity_index),
            "tone_variability": float(tone_variability),
            "pacing_score": float(pacing_score),

            # Pause metrics (existing + new)
            "pause_ratio": float(pause_ratio),
            "good_pause_ratio": float(good_pause_ratio),
            "bad_pause_ratio": float(bad_pause_ratio),
            "pause_bins": {
                "short_count": short_count,
                "ideal_count": ideal_count,
                "medium_count": medium_count,
                "long_count": long_count,
                "short_sec": float(bins.get("short", {}).get("sec", 0.0)),
                "ideal_sec": float(bins.get("ideal", {}).get("sec", 0.0)),
                "medium_sec": float(bins.get("medium", {}).get("sec", 0.0)),
                "long_sec": float(bins.get("long", {}).get("sec", 0.0)),
                "short_rate_per_min": float(short_rate_pm),
                "long_rate_per_min": float(long_rate_pm),
            },

            # Raw prosody features
            "pitch_std_hz": f0_std,
            "pitch_range_hz": f0_range,
            "pitch_mean_hz": float(praat.get("f0_mean_hz", 0.0) or 0.0),
            "intensity_std_db": intensity_std,
            "intensity_mean_db": intensity_mean,
            "hnr_mean_db": hnr,
            "jitter_local": jitter,
            "shimmer_local": shimmer,
            "cpps_smooth_db": cpps,

            # Stability features
            "rate_var_win": rate_var_win,
            "f0_var_win": f0_var_win,
            "intensity_var_win": intensity_var_win,

            # Contour features
            "final_fall_ratio": final_fall_ratio,
            "final_rise_ratio": final_rise_ratio,

            # NLP features
            "articulation_rate": nlp_features["articulation_rate"],
            "syllable_count": nlp_features["syllable_count"],
            "speech_time_sec": nlp_features["speech_time_sec"],
            "target_range_met": nlp_features["target_range_met"],
            "repair_count": nlp_features["repair_count"],
            "repair_rate": nlp_features["repair_rate"],
            "repair_details": nlp_features["repair_details"],
            "coherence_score": nlp_features["coherence_score"],
            "sentence_count": nlp_features["sentence_count"],
            "avg_similarity": nlp_features["avg_similarity"],
            "coherence_details": nlp_features["coherence_details"],
        }

    # ---------- helpers ----------
    def _words_per_min(self, n_words: int, dur_sec: float) -> float:
        if dur_sec <= 0:
            return 0.0
        return n_words / (dur_sec / 60.0)

    def _pacing_score(self, wpm: float) -> float:
        if wpm <= 0:
            return 0.0
        if 140 <= wpm <= 160:  # Updated to research-backed range
            return 1.0
        if wpm < 140:
            return max(0.0, 1.0 - (140 - wpm)/80.0)   # 60–140 maps to 0–1
        return max(0.0, 1.0 - (wpm - 160)/100.0)     # 160–260 maps to 1–0

    def _compute_clarity_index(self, jitter: float, shimmer: float, hnr: float, cpps: float, rms_mean: float, articulation_rate: float) -> float:
        """Research-aligned clarity index combining voice quality, intelligibility, and articulation."""
        # Voice quality component (40% of clarity)
        # Normalize and combine: ↑HNR, ↑CPPs, ↓jitter, ↓shimmer
        # Use z-score normalization with reasonable ranges
        hnr_norm = min(1.0, max(0.0, (hnr - 10) / 15))  # 10-25 dB range
        cpps_norm = min(1.0, max(0.0, (cpps - 5) / 10))  # 5-15 dB range
        jitter_norm = min(1.0, max(0.0, 1.0 - jitter * 20))  # 0-0.05 range
        shimmer_norm = min(1.0, max(0.0, 1.0 - shimmer * 15))  # 0-0.067 range
        
        voice_quality = 0.35 * hnr_norm + 0.35 * cpps_norm - 0.15 * jitter_norm - 0.15 * shimmer_norm
        
        # Intelligibility proxy (40% of clarity) - using articulation rate as proxy
        # Target range: 3.5-5.5 syllables/sec
        if 3.5 <= articulation_rate <= 5.5:
            intelligibility = 1.0
        elif articulation_rate < 3.5:
            intelligibility = max(0.0, articulation_rate / 3.5)
        else:
            intelligibility = max(0.0, 1.0 - (articulation_rate - 5.5) / 2.0)
        
        # Articulation rate component (20% of clarity)
        articulation_score = min(1.0, max(0.0, (articulation_rate - 2.0) / 3.0))
        
        # Combined clarity index
        clarity = 0.4 * voice_quality + 0.4 * intelligibility + 0.2 * articulation_score
        return max(0.0, min(1.0, clarity))

    def _compute_tone_variability(self, f0_std: float, f0_range: float, intensity_std: float, final_fall_ratio: float, final_rise_ratio: float) -> float:
        """Research-aligned tone variability combining pitch dynamics, intensity, and contour."""
        # Pitch dynamics (60% of tone)
        f0_std_norm = min(1.0, max(0.0, f0_std / 50.0))  # Normalize F0 std
        f0_range_norm = min(1.0, max(0.0, f0_range / 200.0))  # Normalize F0 range
        
        # Avoid monotone (too low) and chaotic (too high) variation
        pitch_score = (f0_std_norm + f0_range_norm) / 2.0
        if pitch_score < 0.2:  # Penalize monotone
            pitch_score *= 0.5
        elif pitch_score > 0.8:  # Cap extreme variation
            pitch_score = 0.8 + (pitch_score - 0.8) * 0.2
        
        # Intensity dynamics (25% of tone)
        intensity_score = min(1.0, max(0.0, intensity_std / 8.0))
        
        # Contour shape (15% of tone)
        contour_score = 0.5 * final_fall_ratio + 0.5 * final_rise_ratio
        
        # Combined tone variability
        tone = 0.6 * pitch_score + 0.25 * intensity_score + 0.15 * contour_score
        return max(0.0, min(1.0, tone))

    def _compute_pacing_score(self, wpm: float, pause_ratio: float, good_pause_ratio: float, bad_pause_ratio: float) -> float:
        """Research-aligned pacing score combining WPM and pause quality."""
        # WPM component (60% of pacing)
        if 140 <= wpm <= 160:  # Research-backed optimal range
            wpm_score = 1.0
        elif 130 <= wpm <= 170:
            wpm_score = 0.8
        elif 120 <= wpm <= 180:
            wpm_score = 0.6
        else:
            wpm_score = 0.3
        
        # Pause quality component (40% of pacing)
        # Reward good pauses, penalize bad pauses
        pause_quality = good_pause_ratio - 0.3 * bad_pause_ratio
        pause_quality = max(0.0, min(1.0, pause_quality))
        
        # Combined pacing score
        pacing = 0.6 * wpm_score + 0.4 * pause_quality
        return max(0.0, min(1.0, pacing))
