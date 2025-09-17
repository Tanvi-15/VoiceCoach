from __future__ import annotations
import os, subprocess, uuid
import soundfile as sf
import numpy as np
import librosa

SUPPORTED = {".wav", ".m4a", ".mp3", ".flac"}

class AudioLoadError(Exception):
    pass

def ensure_wav(path: str, target_sr: int = 16000) -> str:
    """Return a mono 16‑bit PCM WAV path at target_sr. Transcodes if needed via ffmpeg."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED:
        raise AudioLoadError(f"Unsupported extension: {ext}")
    if ext == ".wav":
        # still resample/mono‑ize if needed
        y, sr = sf.read(path, always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != target_sr:
            y = librosa.resample(y.astype(float), orig_sr=sr, target_sr=target_sr)
            out = tmpname(".wav")
            sf.write(out, y, target_sr, subtype="PCM_16")
            return out
        return path
        # non‑wav → ffmpeg
    out = tmpname(".wav")
    cmd = [
    "ffmpeg", "-y", "-i", path,
    "-ac", "1", "-ar", str(target_sr), "-sample_fmt", "s16", out
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out

def tmpname(suffix: str) -> str:
    return os.path.join("/tmp", f"voicecoach_{uuid.uuid4().hex}{suffix}")