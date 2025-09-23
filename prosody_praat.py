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

        # CPPs (Cepstral Peak Prominence) - smoothed version for voice clarity
        try:
            cpps = pm.praat.call(snd, "To PowerCepstrogram", 0.002, 0.02, 5000, 50)
            cpps_smooth = pm.praat.call(cpps, "To PowerCepstrum (smoothed)", 0.01, 0.05, 0.001)
            cpps_mean = _safe_float(pm.praat.call(cpps_smooth, "Get mean", 0, 0), 0.0)
        except Exception:
            cpps_mean = 0.0

        # Windowed stability analysis (3-second windows)
        window_size = 3.0  # seconds
        hop_size = 1.0     # seconds
        rate_vars = []
        f0_vars = []
        intensity_vars = []
        
        try:
            for start_time in np.arange(0, dur - window_size, hop_size):
                end_time = start_time + window_size
                
                # WPM for this window (approximate)
                window_duration = min(window_size, dur - start_time)
                
                # F0 variance in window
                window_f0 = pitch_vals[(pitch_vals >= start_time) & (pitch_vals <= end_time)]
                if len(window_f0) > 5:
                    f0_vars.append(float(np.std(window_f0)))
                
                # Intensity variance in window
                window_intensity = inten_vals[(inten_vals >= start_time) & (inten_vals <= end_time)]
                if len(window_intensity) > 5:
                    intensity_vars.append(float(np.std(window_intensity)))
            
            rate_var_win = float(np.mean(rate_vars)) if rate_vars else 0.0
            f0_var_win = float(np.mean(f0_vars)) if f0_vars else 0.0
            intensity_var_win = float(np.mean(intensity_vars)) if intensity_vars else 0.0
        except Exception:
            rate_var_win = 0.0
            f0_var_win = 0.0
            intensity_var_win = 0.0

        # F0 contour analysis - final falls and rises
        try:
            # Get pitch contour with time information
            pitch_times = pitch.xs()
            pitch_freqs = pitch.selected_array["frequency"]
            
            # Analyze final portions of sentences (last 500ms)
            final_falls = 0
            final_rises = 0
            total_sentences = 0
            
            # Simple heuristic: analyze pitch slope in final portions
            for i in range(len(pitch_times) - 10):
                if i + 10 < len(pitch_times):
                    time_window = pitch_times[i:i+10]
                    freq_window = pitch_freqs[i:i+10]
                    
                    # Filter out unvoiced regions
                    valid_mask = np.isfinite(freq_window) & (freq_window > 0)
                    if np.sum(valid_mask) >= 5:
                        valid_freqs = freq_window[valid_mask]
                        valid_times = time_window[valid_mask]
                        
                        # Check if this is near the end of a sentence (simple heuristic)
                        if len(valid_times) > 0 and valid_times[-1] > dur * 0.8:  # Last 20% of speech
                            slope = np.polyfit(valid_times, valid_freqs, 1)[0]
                            if slope < -20:  # Falling
                                final_falls += 1
                            elif slope > 20:  # Rising
                                final_rises += 1
                            total_sentences += 1
            
            final_fall_ratio = final_falls / max(total_sentences, 1)
            final_rise_ratio = final_rises / max(total_sentences, 1)
        except Exception:
            final_fall_ratio = 0.0
            final_rise_ratio = 0.0

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
            "cpps_smooth_db": cpps_mean,
            "rate_var_win": rate_var_win,
            "f0_var_win": f0_var_win,
            "intensity_var_win": intensity_var_win,
            "final_fall_ratio": final_fall_ratio,
            "final_rise_ratio": final_rise_ratio,
        }
