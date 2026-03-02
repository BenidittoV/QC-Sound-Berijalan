import os
import io
import re
import json
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from pydub import AudioSegment
from faster_whisper import WhisperModel

# Optional (mono diarization)
try:
    from pyannote.audio import Pipeline
    import torch
except Exception:
    Pipeline = None
    torch = None


# -----------------------------
# Utilities
# -----------------------------
def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def run_cmd(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{p.stderr[:2000]}")

def to_wav_16k_mono(src_path: str, dst_path: str) -> None:
    # ffmpeg -y -i input -ac 1 -ar 16000 -vn output.wav
    run_cmd(["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", "-vn", dst_path])

def safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:120] if name else "audio"

def fmt_ts(seconds: float) -> str:
    s = max(0.0, float(seconds))
    hh = int(s // 3600)
    mm = int((s % 3600) // 60)
    ss = int(s % 60)
    ms = int(round((s - int(s)) * 1000))
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

def fmt_srt_ts(seconds: float) -> str:
    s = max(0.0, float(seconds))
    hh = int(s // 3600)
    mm = int((s % 3600) // 60)
    ss = int(s % 60)
    ms = int(round((s - int(s)) * 1000))
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

def pick_device() -> Tuple[str, str]:
    # faster-whisper: choose device/compute_type
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


# -----------------------------
# Transcription (faster-whisper)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str):
    device, compute_type = pick_device()
    return WhisperModel(model_size, device=device, compute_type=compute_type)

def transcribe_whisper(
    model: WhisperModel,
    audio_path: str,
    language: Optional[str],
    beam_size: int,
    vad_filter: bool,
) -> List[Dict]:
    segments, info = model.transcribe(
        audio_path,
        language=(None if language == "auto" else language),
        beam_size=beam_size,
        vad_filter=vad_filter,
    )
    out = []
    for seg in segments:  # generator; transcription happens while iterating :contentReference[oaicite:4]{index=4}
        out.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": (seg.text or "").strip()
        })
    return out


# -----------------------------
# Diarization (pyannote) for mono audio
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_diarization_pipeline(model_id: str, hf_token: str):
    if Pipeline is None:
        raise RuntimeError("pyannote.audio belum terpasang. Install: pip install pyannote.audio torch torchaudio")
    # Support both token= and use_auth_token= depending on pyannote version.
    try:
        return Pipeline.from_pretrained(model_id, token=hf_token)
    except TypeError:
        return Pipeline.from_pretrained(model_id, use_auth_token=hf_token)

def diarize_mono(
    pipeline,
    wav_16k_mono_path: str,
    num_speakers: int = 2,
) -> List[Dict]:
    # HF docs: pipeline("audio.wav", num_speakers=2) :contentReference[oaicite:5]{index=5}
    diar = pipeline(wav_16k_mono_path, num_speakers=num_speakers)
    turns = []
    # diar is usually an Annotation
    for turn, _, speaker in diar.itertracks(yield_label=True):
        turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        })
    return turns

def assign_speaker_by_overlap(transcript: List[Dict], turns: List[Dict]) -> List[Dict]:
    if not turns:
        for x in transcript:
            x["speaker"] = "SPK_0"
        return transcript

    for seg in transcript:
        s0, s1 = seg["start"], seg["end"]
        best_spk = None
        best_ov = 0.0
        for t in turns:
            ov = overlap(s0, s1, t["start"], t["end"])
            if ov > best_ov:
                best_ov = ov
                best_spk = t["speaker"]
        seg["speaker"] = best_spk if best_spk is not None else "SPK_0"
    return transcript


# -----------------------------
# Stereo channel split
# -----------------------------
def split_stereo_to_mono_wavs(src_path: str, out_dir: str) -> Tuple[str, str]:
    audio = AudioSegment.from_file(src_path)
    if audio.channels < 2:
        raise ValueError("Audio bukan stereo (channel < 2).")
    monos = audio.split_to_mono()
    left_path = os.path.join(out_dir, "ch0.wav")
    right_path = os.path.join(out_dir, "ch1.wav")
    monos[0].export(left_path, format="wav")
    monos[1].export(right_path, format="wav")
    return left_path, right_path

def detect_channels(src_path: str) -> int:
    audio = AudioSegment.from_file(src_path)
    return int(audio.channels)


# -----------------------------
# Export helpers
# -----------------------------
def to_txt(rows: List[Dict], speaker_map: Dict[str, str]) -> str:
    lines = []
    for r in rows:
        spk = speaker_map.get(r["speaker"], r["speaker"])
        lines.append(f"[{fmt_ts(r['start'])} - {fmt_ts(r['end'])}] {spk}: {r['text']}")
    return "\n".join(lines)

def to_srt(rows: List[Dict], speaker_map: Dict[str, str]) -> str:
    out = []
    i = 1
    for r in rows:
        text = r["text"].strip()
        if not text:
            continue
        spk = speaker_map.get(r["speaker"], r["speaker"])
        out.append(str(i))
        out.append(f"{fmt_srt_ts(r['start'])} --> {fmt_srt_ts(r['end'])}")
        out.append(f"{spk}: {text}")
        out.append("")
        i += 1
    return "\n".join(out)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Transkrip Customer-Agent (Gratis)", layout="wide")
st.title("Transkrip percakapan Customer ↔ Agent (Streamlit + Python)")

if not has_ffmpeg():
    st.error("ffmpeg tidak terdeteksi di PATH. Install ffmpeg dulu (wajib) lalu restart terminal/app.")
    st.stop()

with st.sidebar:
    st.header("Pengaturan")
    input_mode = st.radio("Sumber audio", ["Upload file", "URL audio"], index=0)

    model_size = st.selectbox("Model Whisper (lebih besar = lebih akurat, lebih berat)", [
        "small", "medium", "large-v3"
    ], index=1)

    language = st.selectbox("Bahasa", ["auto", "id", "en"], index=1)
    beam_size = st.slider("Beam size (akurasi vs lambat)", 1, 10, 5)
    vad_filter = st.checkbox("VAD filter (hapus non-speech)", value=True)

    diar_mode = st.selectbox("Mode pemisahan pembicara", [
        "Auto (stereo->channel, mono->pyannote)",
        "Paksa stereo (channel split)",
        "Paksa mono diarization (pyannote)",
        "Tanpa diarization (1 pembicara)",
    ], index=0)

    st.divider()
    st.subheader("Pyannote (untuk audio mono)")
    hf_token = st.text_input("HuggingFace token (jika mono diarization)", type="password")
    diar_model_id = st.text_input("Model diarization", value="pyannote/speaker-diarization-3.1")
    num_speakers = st.number_input("Jumlah pembicara (umumnya 2)", min_value=1, max_value=10, value=2, step=1)

    st.divider()
    st.subheader("Label")
    label_a = st.text_input("Label Speaker A / CH0", value="Agent")
    label_b = st.text_input("Label Speaker B / CH1", value="Customer")

audio_file_path = None
audio_name = "audio"

if input_mode == "Upload file":
    up = st.file_uploader("Upload audio (mp3/wav/m4a)", type=["mp3", "wav", "m4a", "aac", "ogg"])
    if up is not None:
        audio_name = safe_filename(up.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as f:
            f.write(up.read())
            audio_file_path = f.name
else:
    url = st.text_input("Masukkan URL file audio (mp3/wav)")
    if url.strip():
        audio_name = safe_filename(url.split("/")[-1] or "audio_url")
        with st.spinner("Download audio..."):
            r = requests.get(url.strip(), timeout=60)
            r.raise_for_status()
            suffix = os.path.splitext(audio_name)[1] or ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(r.content)
                audio_file_path = f.name

process = st.button("Proses transkripsi", type="primary", disabled=(audio_file_path is None))

if process and audio_file_path:
    try:
        with tempfile.TemporaryDirectory() as td:
            st.info("Memuat model Whisper...")
            whisper = load_whisper(model_size)

            channels = detect_channels(audio_file_path)
            st.write(f"Channel audio terdeteksi: **{channels}**")

            use_stereo = False
            use_pyannote = False

            if diar_mode == "Paksa stereo (channel split)":
                use_stereo = True
            elif diar_mode == "Paksa mono diarization (pyannote)":
                use_pyannote = True
            elif diar_mode == "Tanpa diarization (1 pembicara)":
                use_stereo = False
                use_pyannote = False
            else:
                # Auto
                if channels >= 2:
                    use_stereo = True
                else:
                    use_pyannote = True

            rows = []

            if use_stereo:
                if channels < 2:
                    st.warning("Audio bukan stereo. Jatuh ke mode mono (tanpa diarization).")
                    use_stereo = False

            if use_stereo:
                st.success("Mode: stereo channel-split (paling ideal untuk customer vs agent).")
                ch0, ch1 = split_stereo_to_mono_wavs(audio_file_path, td)

                with st.spinner("Transkrip CH0..."):
                    t0 = transcribe_whisper(whisper, ch0, language, beam_size, vad_filter)
                    for x in t0:
                        rows.append({**x, "speaker": "CH0"})

                with st.spinner("Transkrip CH1..."):
                    t1 = transcribe_whisper(whisper, ch1, language, beam_size, vad_filter)
                    for x in t1:
                        rows.append({**x, "speaker": "CH1"})

                rows.sort(key=lambda r: (r["start"], r["end"]))

            elif use_pyannote:
                st.success("Mode: mono diarization (pyannote) + transkripsi.")
                with st.spinner("Konversi ke WAV 16k mono..."):
                    wav_mono = os.path.join(td, "mono_16k.wav")
                    to_wav_16k_mono(audio_file_path, wav_mono)

                with st.spinner("Transkripsi audio..."):
                    transcript = transcribe_whisper(whisper, wav_mono, language, beam_size, vad_filter)

                if not hf_token.strip():
                    st.warning(
                        "HF token kosong → diarization tidak dijalankan. "
                        "Output jadi 1 pembicara. Untuk diarization mono, pyannote butuh token untuk download model. "
                    )
                    for x in transcript:
                        rows.append({**x, "speaker": "SPK_0"})
                else:
                    with st.spinner("Memuat pipeline diarization..."):
                        pipeline = load_diarization_pipeline(diar_model_id.strip(), hf_token.strip())
                        if torch is not None and torch.cuda.is_available():
                            try:
                                pipeline.to(torch.device("cuda"))
                            except Exception:
                                pass

                    with st.spinner("Menjalankan diarization..."):
                        turns = diarize_mono(pipeline, wav_mono, int(num_speakers))

                    labeled = assign_speaker_by_overlap(transcript, turns)
                    rows = labeled

            else:
                st.success("Mode: transkripsi tanpa diarization (1 pembicara).")
                with st.spinner("Transkripsi audio..."):
                    transcript = transcribe_whisper(whisper, audio_file_path, language, beam_size, vad_filter)
                for x in transcript:
                    rows.append({**x, "speaker": "SPK_0"})

            # Speaker mapping for display
            speaker_map = {
                "CH0": label_a,
                "CH1": label_b,
                "SPK_0": label_a,
            }

            df = pd.DataFrame([{
                "start": fmt_ts(r["start"]),
                "end": fmt_ts(r["end"]),
                "speaker": speaker_map.get(r["speaker"], r["speaker"]),
                "text": r["text"],
            } for r in rows if (r.get("text") or "").strip()])

            st.subheader("Hasil transkrip")
            st.dataframe(df, use_container_width=True, hide_index=True)

            txt = to_txt(rows, speaker_map)
            srt = to_srt(rows, speaker_map)

            st.download_button("Download TXT", data=txt.encode("utf-8"),
                               file_name=f"transkrip_{audio_name}.txt", mime="text/plain")
            st.download_button("Download SRT", data=srt.encode("utf-8"),
                               file_name=f"transkrip_{audio_name}.srt", mime="text/plain")

            st.caption(
                "Catatan akurasi: untuk mendekati target tinggi, rekaman stereo terpisah (customer/agent) "
                "jauh lebih stabil daripada diarization mono."
            )

    except Exception as e:
        st.error(f"Gagal: {e}")
