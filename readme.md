# 🎤 VoiceCoach — Automated Speech Feedback with Whisper + Praat + LLMs

VoiceCoach is a **speech analysis and coaching tool** that takes `.wav` / `.m4a` recordings and provides **quantitative metrics** (clarity, pacing, cadence, engagement) plus **personalized feedback**.  
It combines **ASR (Whisper)**, **prosody analysis (Praat + Librosa)**, custom **pause detection**, and an **LLM feedback engine (Mistral-7B via Ollama)**.  
A **Streamlit dashboard** makes results interactive, with transcripts, rubric scores, speech/pause timelines, and JSON export.

---

## ✨ Features

- 🔊 **Automatic Speech Recognition** with [faster-whisper](https://github.com/guillaumekln/faster-whisper)  
- 🎶 **Prosody extraction** via [Praat (Parselmouth)](https://parselmouth.readthedocs.io) and [Librosa](https://librosa.org)  
  - Pitch (F0), intensity, jitter, shimmer, harmonic-to-noise ratio  
  - RMS energy, tempo, spectral centroid  
- ⏸️ **Pause detection** using Whisper word timestamps  
  - Buckets pauses into *short (0.12–0.25s)*, *good (0.25–0.60s)*, *medium (0.60–1.0s)*, *long (≥1.0s)*  
  - Computes pause ratio, good vs bad pause ratios, rates per minute  
- 📊 **Metrics and Rubric Scoring**  
  - Clarity, Confidence, Tone, Pacing, Engagement, Cadence, Flow (1–5 scale)  
- 🤖 **LLM-powered feedback** with [Ollama](https://ollama.ai) + [Mistral 7B](https://mistral.ai) (local)  
- 🌐 **Streamlit Frontend**  
  - Upload audio, view transcript, summary metrics, rubric table  
  - Pause quality breakdown and timeline visualization  
  - JSON download for results  

---

## 🛠️ Tech Stack

- **Python 3.10+**
- [Whisper (faster-whisper / CTranslate2)](https://github.com/guillaumekln/faster-whisper)  
- [Praat (Parselmouth)](https://parselmouth.readthedocs.io)  
- [Librosa](https://librosa.org)  
- [Pyphen](https://github.com/Kozea/pyphen) — syllable counting  
- [Streamlit](https://streamlit.io) — frontend dashboard  
- [Altair](https://altair-viz.github.io) — timeline visualization  
- [Ollama](https://ollama.ai) — local LLM runner  
- [Mistral 7B](https://mistral.ai) — feedback model  

---

## 📂 Project Structure

```
VoiceCoach/
   ├── app.py                    # Streamlit web dashboard
   ├── main.py                   # Command-line interface
   ├── utils_audio.py            # Audio processing utilities (FFmpeg → WAV)
   ├── asr_whisper.py            # Whisper ASR integration
   ├── prosody_praat.py          # Prosody analysis via Praat
   ├── prosody_librosa.py        # Prosody analysis via Librosa
   ├── vad_from_asr.py           # Voice Activity & pause detection
   ├── metrics.py                # Derived metrics (WPM, clarity, etc.)
   ├── rubric.py                 # Rubric scoring system (1-5 scale)
   ├── coach_ollama.py           # LLM feedback engine (Ollama + Mistral)
   ├── requirements.txt          # Python dependencies
   └── demo_assets/              # Sample audio files for testing
       ├── harvard.wav
       ├── test2.m4a
       └── vani.m4a

```

---

## 🚀 Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
Make sure [FFmpeg](https://www.ffmpeg.org/download.html) is installed (for audio conversion).

```

python main.py --audio demo_assets/harvard.wav --whisper base.en --ollama-model mistral:7b
```
Outputs a JSON report with all metrics, rubric scores, and feedback.

```
streamlit run app.py
```
Runs streamlit UI

- Upload .wav or .m4a audio
- View transcript, summary metrics, rubric scores
- Inspect pause quality and timeline
- Read coaching feedback
- Download JSON results
