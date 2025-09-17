from __future__ import annotations
import parselmouth as pm
import numpy as np
from typing import Dict, Any

# Default pitch floor/ceiling suitable for adult voices
PITCH_FLOOR = 75
PITCH_CEIL = 500

def _safe_float(x, default=0.0):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)

class PraatProsody:
    def __init__(self, pitch_floor: int = PITCH_FLOOR, pitch_ceil: int = PITCH_CEIL):
        self.pitch_floor = pitch_floor
        self.pitch_ceil = pitch_ceil

    def extract(self, wav_path: str) -> Dict[str, Any]:
        snd = pm.Sound(wav_path)
        dur = float(snd.get_total_duration())

        # --- Pitch & intensity ---
        pitch = snd.to_pitch(time_step=0.01,
                             pitch_floor=self.pitch_floor,
                             pitch_ceiling=self.pitch_ceil)
        intensity = snd.to_intensity(time_step=0.01, minimum_pitch=self.pitch_floor)

        # Pitch stats (Hz)
        pitch_vals = pitch.selected_array["frequency"]
        pitch_vals = pitch_vals[np.isfinite(pitch_vals) & (pitch_vals > 0)]
        f0_mean = float(np.mean(pitch_vals)) if pitch_vals.size else 0.0
        f0_std  = float(np.std(pitch_vals))  if pitch_vals.size else 0.0
        f0_rng  = float(np.max(pitch_vals) - np.min(pitch_vals)) if pitch_vals.size else 0.0

        # Intensity (dB) stats
        inten_vals = intensity.values.T.squeeze()
        inten_vals = inten_vals[np.isfinite(inten_vals)]
        i_mean = float(np.mean(inten_vals)) if inten_vals.size else 0.0
        i_std  = float(np.std(inten_vals))  if inten_vals.size else 0.0

        # --- Jitter / Shimmer / HNR ---
        # Create a PointProcess (periodic, cc) once
        point_proc = pm.praat.call(snd, "To PointProcess (periodic, cc)",
                                   self.pitch_floor, self.pitch_ceil)

        # JITTER: call on PointProcess ALONE
        try:
            jitter_local = pm.praat.call(
                point_proc, "Get jitter (local)",
                0, 0, self.pitch_floor, self.pitch_ceil, 1.3, 1.6
            )
            jitter_local = _safe_float(jitter_local, 0.0)
        except Exception:
            jitter_local = 0.0

        # SHIMMER: call on [Sound, PointProcess]
        try:
            shimmer_local = pm.praat.call(
                [snd, point_proc], "Get shimmer (local)",
                0, 0, self.pitch_floor, self.pitch_ceil, 1.3, 1.6, 1.6
            )
            shimmer_local = _safe_float(shimmer_local, 0.0)
        except Exception:
            shimmer_local = 0.0

        # HNR: use Harmonicity (cc) → mean dB
        try:
            hnr = pm.praat.call(snd, "To Harmonicity (cc)", 0.01, self.pitch_floor, 0.1, 1.0)
            hnr_mean = _safe_float(pm.praat.call(hnr, "Get mean", 0, 0), 0.0)
        except Exception:
            hnr_mean = 0.0

        return {
            "duration_sec": dur,
            "f0_mean_hz": f0_mean,
            "f0_std_hz":  f0_std,
            "f0_range_hz": f0_rng,
            "intensity_mean_db": i_mean,
            "intensity_std_db":  i_std,
            "jitter_local": jitter_local,
            "shimmer_local": shimmer_local,
            "hnr_mean_db": hnr_mean,
        }
