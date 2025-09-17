from __future__ import annotations
from faster_whisper import WhisperModel
from typing import Dict, Any

class ASR:
    def __init__(self, model_size: str = "base.en", device: str = "auto"):
        self.model = WhisperModel(model_size, device=device)

    def transcribe(self, wav_path: str) -> Dict[str, Any]:
        segments, info = self.model.transcribe(
            wav_path,
            vad_filter=True,
            beam_size=1,
            word_timestamps=True,
        )
        words = []
        text = []
        for seg in segments:
            text.append(seg.text)
            if seg.words:
                words.extend([
                {
                    "word": w.word.strip(),
                    "start": float(w.start),
                    "end": float(w.end),
                    "prob": float(getattr(w, "probability", 1.0)),
                }
            for w in seg.words
            ])
        return {
        "language": info.language,
        "duration": info.duration,
        "text": " ".join(t.strip() for t in text).strip(),
        "words": words,
        }