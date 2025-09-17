from __future__ import annotations
import numpy as np, librosa
from typing import Dict, Any, Tuple

class LibrosaProsody:
    def __init__(self, sr: int = 16000):
        self.sr = sr

    def extract(self, wav_path: str) -> Dict[str, Any]:
        y, sr = librosa.load(wav_path, sr=self.sr, mono=True)
        # RMS & speaking energy stats
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        rms_mean, rms_std = float(np.mean(rms)), float(np.std(rms))
        # Tempo estimate (rough proxy)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # Spectral centroid variation (brightness variability)
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        sc_mean, sc_std = float(np.mean(sc)), float(np.std(sc))
        return {
            "rms_mean": rms_mean,
            "rms_std": rms_std,
            "tempo_bpm": float(tempo),
            "spectral_centroid_mean": sc_mean,
            "spectral_centroid_std": sc_std,
        }