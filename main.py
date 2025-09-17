from __future__ import annotations
import argparse, json
from utils_audio import ensure_wav
from asr_whisper import ASR
from prosody_praat import PraatProsody
from prosody_librosa import LibrosaProsody
from vad_from_asr import PauseAnalyzer
from metrics import Metrics
from rubric import RubricScorer

# Optional: only import requests where needed to avoid failing earlier
import requests

print("Starting VoiceCoach...")
def run(audio_path: str, whisper_model: str, ollama_model: str, min_pause: float, skip_llm: bool=False) -> None:
    wav = ensure_wav(audio_path, target_sr=16000)
    print("WAV loaded successfully")
    # --- Pipeline ---
    asr = ASR(model_size=whisper_model)
    asr_out = asr.transcribe(wav)

    praat = PraatProsody().extract(wav)

    # ... above code ...
    print("PraatProsody.extract()", flush=True)
    try:
        praat = PraatProsody().extract(wav)
    except Exception as e:
        print(f"Praat failed, continuing with safe defaults: {type(e).__name__}: {e}", flush=True)
        praat = {
            "duration_sec": 0.0,
            "f0_mean_hz": 0.0, "f0_std_hz": 0.0, "f0_range_hz": 0.0,
            "intensity_mean_db": 0.0, "intensity_std_db": 0.0,
            "jitter_local": 0.0, "shimmer_local": 0.0, "hnr_mean_db": 0.0,
        }
    # ... keep the rest unchanged ...

    lib = LibrosaProsody().extract(wav)

    # Pauses from ASR word timestamps (no external VAD)
    vad = PauseAnalyzer(min_pause_sec=min_pause).analyze(asr_out, praat["duration_sec"])

    m = Metrics().compute(asr_out, praat, lib, vad)
    r = RubricScorer().score(m)

    # --- LLM coaching (robust to failure) ---
    feedback = None
    llm_error = None
    if not skip_llm:
        try:
            from coach_ollama import Coach
            coach = Coach(model=ollama_model)
            feedback = coach.coach(asr_out.get("text", ""), m, r)
        except requests.exceptions.RequestException as e:
            llm_error = f"Ollama request error: {e}" # connection/timeout/etc.
        except Exception as e:
            llm_error = f"LLM error: {type(e).__name__}: {e}"

    if feedback is None:
        feedback = (
            "(LLM feedback unavailable. Ensure Ollama is running and the model is downloaded; "
            "you can re-run with --ollama-model mistral:7b or use --skip-llm to bypass.)"
        )

    # --- Final report: always printed ---
    report = {
        "audio": audio_path,
        "asr": {
            k: (v if k != "words" else f"{len(v)} words with timestamps")
            for k, v in asr_out.items()
        },
        "praat": praat,
        "librosa": lib,
        "vad": vad,
        "metrics": m,
        "rubric": r,
        "feedback": feedback,
    }
    if llm_error:
        report["llm_error"] = llm_error

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Path to .wav/.m4a")
    ap.add_argument("--whisper", default="base.en", help="faster-whisper model size")
    ap.add_argument("--ollama-model", default="mistral:7b")
    ap.add_argument("--min-pause", type=float, default=0.12, help="Seconds defining a pause between words")
    ap.add_argument("--skip-llm", action="store_true", help="Bypass Ollama and just print metrics")
    args = ap.parse_args()
    print("Arguments parsed successfully")
    run(args.audio, args.whisper, args.ollama_model, args.min_pause, skip_llm=args.skip_llm)