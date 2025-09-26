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
def run(audio_path: str, whisper_model: str, ollama_model: str, skip_llm: bool=False) -> None:
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

    # VAD from ASR gaps with default thresholds (hard-coded in PauseAnalyzer)
    vad = PauseAnalyzer().analyze(asr_out, praat["duration_sec"])

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

    # --- Final report: comprehensive JSON matching UI display ---
    report = {
        "audio": audio_path,
        "transcript": {
            "text": asr_out.get("text", ""),
            "word_count": len(asr_out.get("words", [])),
            "words_with_timestamps": f"{len(asr_out.get('words', []))} words with timestamps"
        },
        "summary_metrics": {
            "wpm": m.get('wpm', 0.0),
            "articulation_rate": m.get('articulation_rate', 0.0),
            "pause_ratio": m.get('pause_ratio', 0.0),
            "clarity_index": m.get('clarity_index', 0.0),
            "cpps_smooth_db": m.get('cpps_smooth_db', 0.0),
            "filler_ratio": m.get('filler_ratio', 0.0),
            "tone_variability": m.get('tone_variability', 0.0),
            "pitch_range_hz": m.get('pitch_range_hz', 0.0),
            "repair_rate": m.get('repair_rate', 0.0)
        },
        "rubric_scores": {
            "clarity": {"score": r['Clarity']['score'], "weight": "20%"},
            "confidence": {"score": r['Confidence']['score'], "weight": "15%"},
            "tone": {"score": r['Tone']['score'], "weight": "15%"},
            "pacing": {"score": r['Pacing']['score'], "weight": "15%"},
            "engagement": {"score": r['Engagement']['score'], "weight": "15%"},
            "cadence": {"score": r['Cadence']['score'], "weight": "10%"},
            "flow": {"score": r['Flow']['score'], "weight": "10%"}
        },
        "detailed_rubric_breakdown": r,
        "voice_quality_metrics": {
            "hnr_mean_db": m.get('hnr_mean_db', 0.0),
            "jitter_local": m.get('jitter_local', 0.0),
            "cpps_smooth_db": m.get('cpps_smooth_db', 0.0),
            "shimmer_local": m.get('shimmer_local', 0.0),
            "intensity_mean_db": m.get('intensity_mean_db', 0.0),
            "intensity_std_db": m.get('intensity_std_db', 0.0)
        },
        "stability_analysis": {
            "rate_var_win": m.get('rate_var_win', 0.0),
            "f0_var_win": m.get('f0_var_win', 0.0),
            "intensity_var_win": m.get('intensity_var_win', 0.0),
            "final_fall_ratio": m.get('final_fall_ratio', 0.0)
        },
        "speech_analysis": {
            "syllable_count": m.get('syllable_count', 0),
            "speech_time_sec": m.get('speech_time_sec', 0.0),
            "repair_count": m.get('repair_count', 0),
            "target_range_met": m.get('target_range_met', False),
            "coherence_score": m.get('coherence_score', 0.0),
            "sentence_count": m.get('sentence_count', 0)
        },
        "pause_analysis": {
            "pause_bins": m.get("pause_bins", {}),
            "good_pause_ratio": m.get('good_pause_ratio', 0.0),
            "bad_pause_ratio": m.get('bad_pause_ratio', 0.0),
            "pause_count": vad.get('pause_count', 0) if vad else 0,
            "pause_ratio": vad.get('pause_ratio', 0.0) if vad else 0.0
        },
        "coaching_feedback": feedback,
        # Raw data for advanced analysis
        "raw_data": {
            "asr": asr_out,
            "praat": praat,
            "librosa": lib,
            "vad": vad,
            "metrics": m,
            "rubric": r
        }
    }
    if llm_error:
        report["llm_error"] = llm_error

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Path to .wav/.m4a")
    ap.add_argument("--whisper", default="base.en", help="faster-whisper model size")
    ap.add_argument("--ollama-model", default="mistral:7b")
    ap.add_argument("--skip-llm", action="store_true", help="Bypass Ollama and just print metrics")
    args = ap.parse_args()
    print("Arguments parsed successfully")
    run(args.audio, args.whisper, args.ollama_model, skip_llm=args.skip_llm)