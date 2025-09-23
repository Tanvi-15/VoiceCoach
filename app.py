
from __future__ import annotations
import io, json, os, tempfile, traceback
from typing import Dict, Any, List, Tuple

import streamlit as st
import numpy as np
import pandas as pd

# Local modules
from utils_audio import ensure_wav
from asr_whisper import ASR
from prosody_praat import PraatProsody
from prosody_librosa import LibrosaProsody
from vad_from_asr import PauseAnalyzer
from metrics import Metrics
from rubric import RubricScorer

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="VoiceCoach", page_icon="🎤", layout="wide")
st.sidebar.title("⚙️ Settings")
whisper_model = st.sidebar.selectbox("Whisper model", ["tiny.en","base.en","small.en","medium.en"], index=1)
skip_llm = st.sidebar.checkbox("Skip LLM (metrics only)", value=False)
ollama_model = st.sidebar.text_input("Ollama model", value="mistral:7b")
save_json = st.sidebar.checkbox("Offer JSON download", value=True)

st.title("🎤 VoiceCoach — Research-Aligned Speech Analysis")
st.caption("Advanced voice analysis with CPPs, articulation rate, stability metrics, NLP coherence, and research-backed scoring. Upload .wav or .m4a for comprehensive feedback.")

# ---------------- HELPERS ----------------
@st.cache_data(show_spinner=False)
def _save_temp(data: bytes, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path

def run_pipeline(audio_path: str, whisper_model: str, ollama_model: str, skip_llm: bool) -> Dict[str,Any]:
    wav = ensure_wav(audio_path, target_sr=16000)

    asr = ASR(model_size=whisper_model)
    asr_out = asr.transcribe(wav)

    praat = PraatProsody().extract(wav)
    lib = LibrosaProsody().extract(wav)

    # VAD from ASR gaps with default thresholds (hard-coded in PauseAnalyzer)
    vad = PauseAnalyzer().analyze(asr_out, praat["duration_sec"])

    m = Metrics().compute(asr_out, praat, lib, vad)
    r = RubricScorer().score(m)

    feedback = None
    llm_error = None
    if not skip_llm:
        try:
            import requests
            from coach_ollama import Coach
            coach = Coach(model=ollama_model)
            feedback = coach.coach(asr_out.get("text",""), m, r)
        except Exception as e:
            llm_error = f"LLM unavailable: {type(e).__name__}: {e}"
    if feedback is None:
        feedback = "(LLM feedback unavailable or skipped.)"

    result = {
        "audio": audio_path,
        "asr": {
            k: (v if k != "words" else f"{len(v)} words with timestamps")
            for k,v in asr_out.items()
        },
        "praat": praat,
        "librosa": lib,
        "vad": vad,
        "metrics": m,
        "rubric": r,
        "feedback": feedback,
    }
    if llm_error:
        result["llm_error"] = llm_error
    return result

def intervals_to_df(pauses: List[Tuple[float,float]], total: float) -> pd.DataFrame:
    """Build alternating speech/pause timeline from pause intervals."""
    rows = []
    t = 0.0
    for (s,e) in pauses:
        if s > t:
            rows.append({"start": t, "end": s, "label": "speech"})
        rows.append({"start": s, "end": e, "label": "pause"})
        t = e
    if t < total:
        rows.append({"start": t, "end": total, "label": "speech"})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["dur"] = df["end"] - df["start"]
    return df

# ---------------- MAIN UI ----------------
upload = st.file_uploader("Upload audio (.wav or .m4a)", type=["wav","m4a","mp3","flac"], accept_multiple_files=False)

if upload is not None:
    st.audio(upload, format=f"audio/{upload.type.split('/')[-1]}")
    path = _save_temp(upload.read(), suffix=f".{upload.name.split('.')[-1]}")

    with st.spinner("Analyzing… this can take a moment on CPU"):
        try:
            report = run_pipeline(path, whisper_model, ollama_model, skip_llm)
            st.success("Analysis complete ✅")
        except Exception:
            st.error("Analysis failed. See traceback below.")
            st.code(traceback.format_exc())
            st.stop()

    # ---- Feedback FIRST (right below success alert) ----
    st.subheader("Coaching feedback")
    st.write(report.get("feedback","(none)"))
    if report.get("llm_error"):
        st.warning(report["llm_error"])

    # ---- Analysis section (dropdown-like grouping via header) ----
    st.markdown("---")
    st.header("Analysis")

    # Transcript (full width)
    st.subheader("Transcript")
    st.write(report["asr"].get("text","(no text)"))

    # Summary (research-aligned metrics)
    st.subheader("Summary")
    m = report["metrics"]
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric("WPM", f"{m['wpm']:.0f}")
        st.metric("Articulation Rate", f"{m.get('articulation_rate', 0.0):.1f} syll/sec")
        st.metric("Pause Ratio", f"{m['pause_ratio']:.2f}")
        
    with c2:
        st.metric("Clarity Index", f"{m['clarity_index']:.3f}")
        st.metric("CPPs", f"{m.get('cpps_smooth_db', 0.0):.1f} dB")
        st.metric("Filler Ratio", f"{100*m['filler_ratio']:.1f}%")
        
    with c3:
        st.metric("Tone Variability", f"{m['tone_variability']:.3f}")
        st.metric("Pitch Range", f"{m.get('pitch_range_hz', 0.0):.0f} Hz")
        st.metric("Repair Rate", f"{100*m.get('repair_rate', 0.0):.1f}%")

    # Rubric scores with detailed breakdown
    st.subheader("Research-Aligned Rubric Scores (1–5)")
    rub = report["rubric"]
    
    # Main scores in columns
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Clarity", f"{rub['Clarity']['score']}/5", help=f"Weight: 20%")
        st.metric("Confidence", f"{rub['Confidence']['score']}/5", help=f"Weight: 15%")
    with c2:
        st.metric("Tone", f"{rub['Tone']['score']}/5", help=f"Weight: 15%")
        st.metric("Pacing", f"{rub['Pacing']['score']}/5", help=f"Weight: 15%")
    with c3:
        st.metric("Engagement", f"{rub['Engagement']['score']}/5", help=f"Weight: 15%")
        st.metric("Cadence", f"{rub['Cadence']['score']}/5", help=f"Weight: 10%")
    with c4:
        st.metric("Flow", f"{rub['Flow']['score']}/5", help=f"Weight: 10%")
    
    # Detailed breakdown in expandable sections
    with st.expander("📊 Detailed Score Breakdown"):
        for category, score_data in rub.items():
            st.write(f"**{category}** (Score: {score_data['score']}/5)")
            st.write(f"*{score_data['why']}*")
            
            # Show component details if available
            if 'details' in score_data:
                details = score_data['details']
                if isinstance(details, dict):
                    detail_cols = st.columns(len(details))
                    for i, (component, value) in enumerate(details.items()):
                        with detail_cols[i]:
                            st.metric(component.replace('_', ' ').title(), f"{value:.3f}")
            st.markdown("---")

    # Advanced Features Section
    st.subheader("🔬 Advanced Voice Analysis")
    
    # Voice Quality Metrics
    with st.expander("🎵 Voice Quality"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("HNR", f"{m.get('hnr_mean_db', 0.0):.1f} dB", help="Harmonic-to-Noise Ratio")
            st.metric("Jitter", f"{m.get('jitter_local', 0.0):.4f}", help="Pitch period variation")
        with c2:
            st.metric("CPPs", f"{m.get('cpps_smooth_db', 0.0):.1f} dB", help="Cepstral Peak Prominence")
            st.metric("Shimmer", f"{m.get('shimmer_local', 0.0):.4f}", help="Amplitude variation")
        with c3:
            st.metric("Intensity Mean", f"{m.get('intensity_mean_db', 0.0):.1f} dB")
            st.metric("Intensity Std", f"{m.get('intensity_std_db', 0.0):.1f} dB")

    # Stability Analysis
    with st.expander("📈 Stability Analysis"):
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Rate Variance", f"{m.get('rate_var_win', 0.0):.1f}", help="Speaking rate consistency")
            st.metric("F0 Variance", f"{m.get('f0_var_win', 0.0):.1f}", help="Pitch consistency")
        with c2:
            st.metric("Intensity Variance", f"{m.get('intensity_var_win', 0.0):.1f}", help="Volume consistency")
            st.metric("Final Fall Ratio", f"{m.get('final_fall_ratio', 0.0):.2f}", help="Natural sentence endings")

    # NLP Features
    with st.expander("📝 Speech Analysis"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Syllables", f"{m.get('syllable_count', 0)}", help="Total syllable count")
            st.metric("Speech Time", f"{m.get('speech_time_sec', 0.0):.1f}s", help="Time excluding pauses")
        with c2:
            st.metric("Repair Count", f"{m.get('repair_count', 0)}", help="Disfluency repairs")
            st.metric("Target Range Met", "✅" if m.get('target_range_met', False) else "❌", help="3.5-5.5 syll/sec")
        with c3:
            st.metric("Coherence Score", f"{m.get('coherence_score', 0.0):.3f}", help="Topic flow")
            st.metric("Sentences", f"{m.get('sentence_count', 0)}", help="Sentence count")

    # Pause quality (table)
    st.subheader("Pause Analysis")
    pb = report["metrics"]["pause_bins"]
    pq_df = pd.DataFrame([
        {"bucket":"short (0.12–0.25s)",  "count": pb["short_count"],  "seconds": pb["short_sec"],  "rate/min": pb["short_rate_per_min"]},
        {"bucket":"ideal (0.25–0.60s)",  "count": pb["ideal_count"],  "seconds": pb["ideal_sec"],  "rate/min": 0.0},
        {"bucket":"medium (0.60–1.00s)", "count": pb["medium_count"], "seconds": pb["medium_sec"], "rate/min": 0.0},
        {"bucket":"long (≥1.00s)",       "count": pb["long_count"],   "seconds": pb["long_sec"],   "rate/min": pb["long_rate_per_min"]},
    ])
    st.dataframe(pq_df, use_container_width=True)
    st.caption(
        f"good_pause_ratio={report['metrics']['good_pause_ratio']:.2f} · "
        f"bad_pause_ratio={report['metrics']['bad_pause_ratio']:.2f}"
    )

    # Timeline: speech vs pauses (bigger so it's visible by default)
    st.subheader("Timeline: speech vs pauses")
    v = report["vad"]
    timeline = intervals_to_df(v.get("pauses", []), report["metrics"]["duration_sec"]) \
               if v else pd.DataFrame(columns=["start","end","label","dur"])
    if not timeline.empty:
        import altair as alt
        chart = alt.Chart(timeline).mark_bar().encode(
            x=alt.X('start:Q', title='Time (s)'),
            x2='end:Q',
            color=alt.Color('label:N', legend=None)
        ).properties(height=160, width='container')
        st.altair_chart(chart, use_container_width=True)
        st.caption(f"Pauses: {v['pause_count']}  ·  Pause ratio: {v['pause_ratio']:.2f}")
    else:
        st.info("No pauses detected or word timestamps unavailable.")

    # JSON download
    if save_json:
        st.download_button(
            "⬇️ Download JSON report",
            data=json.dumps(report, indent=2).encode("utf-8"),
            file_name="voicecoach_report.json",
            mime="application/json"
        )

else:
    st.info("Upload an audio file to begin analysis.")
