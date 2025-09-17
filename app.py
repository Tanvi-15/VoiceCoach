# app.py — Streamlit UI for VoiceCoach (Praat + Whisper + Ollama)
# Place this file at the project root (same folder as main.py)
# Run:  streamlit run app.py

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

# ---- Sidebar controls ----
st.set_page_config(page_title="VoiceCoach", page_icon="🎤", layout="wide")
st.sidebar.title("⚙️ Settings")
whisper_model = st.sidebar.selectbox("Whisper model", ["tiny.en","base.en","small.en","medium.en"], index=1)
min_pause = st.sidebar.slider("Min pause (seconds)", 0.05, 0.40, 0.12, 0.01)
skip_llm = st.sidebar.checkbox("Skip LLM (metrics only)", value=False)
ollama_model = st.sidebar.text_input("Ollama model", value="mistral:7b")
save_json = st.sidebar.checkbox("Offer JSON download", value=True)

# ---- Header ----
st.title("🎤 VoiceCoach — Prosody + ASR Feedback")
st.caption("Upload .wav or .m4a, analyze prosody (Praat), speed (Whisper), pauses, and get rubric + coaching feedback.")

# ---- File uploader ----
upload = st.file_uploader("Upload audio (.wav or .m4a)", type=["wav","m4a","mp3","flac"], accept_multiple_files=False)

# Utility: run pipeline
@st.cache_data(show_spinner=False)
def _save_temp(data: bytes, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def run_pipeline(audio_path: str, whisper_model: str, ollama_model: str, min_pause: float, skip_llm: bool) -> Dict[str,Any]:
    # In-app import to keep UI reactive on code edits
    result: Dict[str,Any] = {}
    wav = ensure_wav(audio_path, target_sr=16000)

    asr = ASR(model_size=whisper_model)
    asr_out = asr.transcribe(wav)

    praat = PraatProsody().extract(wav)
    lib = LibrosaProsody().extract(wav)

    vad = PauseAnalyzer(min_pause_sec=min_pause).analyze(asr_out, praat["duration_sec"])

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

# Visualization helpers

def intervals_to_df(pauses: List[Tuple[float,float]], total: float) -> pd.DataFrame:
    # Build alternating speech/pause timeline
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
    df["dur"] = df["end"] - df["start"]
    return df

# ---- Main UI logic ----
if upload is not None:
    st.audio(upload, format=f"audio/{upload.type.split('/')[-1]}")
    path = _save_temp(upload.read(), suffix=f".{upload.name.split('.')[-1]}")

    with st.spinner("Analyzing… this can take a moment on CPU"):
        try:
            report = run_pipeline(path, whisper_model, ollama_model, min_pause, skip_llm)
            st.success("Analysis complete ✅")
        except Exception:
            st.error("Analysis failed. See traceback below.")
            st.code(traceback.format_exc())
            st.stop()

    # ---- Columns for summary + transcript ----
    c1, c2 = st.columns([1,1])
    with c1:
        st.subheader("Summary")
        m = report["metrics"]
        st.metric("WPM", f"{m['wpm']:.0f}")
        st.metric("Pause ratio", f"{m['pause_ratio']:.2f}")
        st.metric("Tone variability", f"{m['tone_variability']:.1f}")
        st.metric("Clarity index", f"{m['clarity_index']:.2f}")
        st.metric("Filler ratio", f"{100*m['filler_ratio']:.1f}%")
    with c2:
        st.subheader("Transcript")
        st.write(report["asr"].get("text","(no text)"))

    # ---- Rubric table ----
    st.subheader("Rubric scores (1–5)")
    rub = report["rubric"]
    rub_df = pd.DataFrame({k: [v["score"], v["why"]] for k,v in rub.items()}, index=["score","why"]).T
    st.dataframe(rub_df, use_container_width=True)

    # ---- Pause timeline ----
    st.subheader("Timeline: speech vs pauses")
    v = report["vad"]
    timeline = intervals_to_df(v.get("pauses", []), report["metrics"]["duration_sec"]) \
               if v else pd.DataFrame(columns=["start","end","label","dur"]) 
    if not timeline.empty:
        import altair as alt
        chart = alt.Chart(timeline).mark_bar().encode(
            x=alt.X('start:Q', title='Time (s)'),
            x2='end:Q',
            y=alt.value(20),
            color=alt.Color('label:N', legend=None)
        ).properties(height=60, width=900)
        st.altair_chart(chart, use_container_width=True)
        st.caption(f"Pauses: {v['pause_count']}  ·  Pause ratio: {v['pause_ratio']:.2f}")
    else:
        st.info("No pauses detected or word timestamps unavailable.")

    # ---- Feedback ----
    st.subheader("Coaching feedback")
    st.write(report.get("feedback","(none)"))
    if report.get("llm_error"):
        st.warning(report["llm_error"]) 

    # ---- JSON download ----
    if save_json:
        st.download_button(
            "⬇️ Download JSON report",
            data=json.dumps(report, indent=2).encode("utf-8"),
            file_name="voicecoach_report.json",
            mime="application/json"
        )

else:
    st.info("Upload an audio file to begin analysis.")
