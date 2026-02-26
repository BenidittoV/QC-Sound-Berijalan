import os
import io
import json
import math
import uuid
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import requests
import streamlit as st

import soundfile as sf

from sklearn.cluster import AgglomerativeClustering

from faster_whisper import WhisperModel

import torch
from speechbrain.inference.speaker import EncoderClassifier


# ----------------------------
# Utils: download + ffmpeg convert
# ----------------------------
def download_file(url: str, out_path: str, timeout: int = 60) -> None:
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

def ffmpeg_to_wav16k_mono(in_path: str, out_path: str) -> None:
    # Convert to 16kHz mono wav PCM16
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        out_path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{p.stderr}")


# ----------------------------
# VAD segmentation (WebRTC VAD)
# ----------------------------
@dataclass
class Segment:
    start: float
    end: float

def read_wav(path: str) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), sr

def float_to_pcm16(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    return pcm.tobytes()

def vad_segments(
    wav_path: str,
    vad_aggressiveness: int = 2,
    frame_ms: int = 30,
    min_seg_s: float = 0.6,
    merge_gap_s: float = 0.3
) -> List[Segment]:
    audio, sr = read_wav(wav_path)
    if sr != 16000:
        raise ValueError("Expected 16kHz wav. Convert with ffmpeg_to_wav16k_mono first.")

    vad = webrtcvad.Vad(vad_aggressiveness)
    frame_len = int(sr * frame_ms / 1000)
    pcm = float_to_pcm16(audio)

    # iterate frames
    segments = []
    voiced = False
    seg_start = 0

    n_frames = int(len(audio) / frame_len)
    for i in range(n_frames):
        start_i = i * frame_len
        end_i = start_i + frame_len
        frame_bytes = pcm[start_i*2:end_i*2]  # int16 => 2 bytes
        is_speech = vad.is_speech(frame_bytes, sr)

        t0 = start_i / sr
        t1 = end_i / sr

        if is_speech and not voiced:
            voiced = True
            seg_start = t0
        elif (not is_speech) and voiced:
            voiced = False
            seg_end = t1
            segments.append(Segment(seg_start, seg_end))

    if voiced:
        segments.append(Segment(seg_start, n_frames * frame_len / sr))

    # merge close segments
    merged = []
    for s in segments:
        if not merged:
            merged.append(s)
            continue
        prev = merged[-1]
        if s.start - prev.end <= merge_gap_s:
            prev.end = max(prev.end, s.end)
        else:
            merged.append(s)

    # filter very short
    merged = [s for s in merged if (s.end - s.start) >= min_seg_s]
    return merged


# ----------------------------
# Speaker embeddings + clustering (2 speakers)
# ----------------------------
def get_embeddings_for_segments(
    wav_path: str,
    segments: List[Segment],
    device: str = "cpu"
) -> np.ndarray:
    audio, sr = read_wav(wav_path)
    assert sr == 16000

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )

    embs = []
    for seg in segments:
        s = int(seg.start * sr)
        e = int(seg.end * sr)
        chunk = audio[s:e]

        # pad if too short (speechbrain likes >= ~1s; but we keep VAD min ~0.6s)
        if len(chunk) < sr:
            pad = sr - len(chunk)
            chunk = np.pad(chunk, (0, pad), mode="constant")

        # speechbrain expects torch tensor shape [batch, time]
        wav_t = torch.from_numpy(chunk).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = encoder.encode_batch(wav_t).squeeze().cpu().numpy()
        embs.append(emb)

    return np.vstack(embs)

def cluster_two_speakers(embeddings: np.ndarray) -> np.ndarray:
    # 2 clusters (Agent vs Customer)
    if embeddings.shape[0] < 2:
        # fallback: all same speaker
        return np.zeros((embeddings.shape[0],), dtype=int)

    clustering = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average")
    labels = clustering.fit_predict(embeddings)
    return labels.astype(int)

def label_agent_customer(segments: List[Segment], spk_labels: np.ndarray) -> Dict[int, str]:
    """
    Heuristik sederhana:
    - speaker yang muncul paling awal dianggap "Agent" (sering terjadi di call center),
      sisanya "Customer".
    Kamu bisa ubah aturan ini kalau di data kamu kebalik.
    """
    first_idx = int(np.argmin([s.start for s in segments])) if segments else 0
    first_spk = int(spk_labels[first_idx]) if len(spk_labels) else 0
    other_spk = 1 - first_spk

    return {first_spk: "Agent", other_spk: "Customer"}


# ----------------------------
# Transcription: faster-whisper
# ----------------------------
@dataclass
class WhisperSeg:
    start: float
    end: float
    text: str

def transcribe_whisper(
    wav_path: str,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "id"  # ubah ke "en" atau None untuk auto
) -> List[WhisperSeg]:
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        wav_path,
        language=language,
        vad_filter=False,  # kita sudah VAD sendiri untuk diarization
        beam_size=5
    )
    out = []
    for s in segments:
        txt = (s.text or "").strip()
        if txt:
            out.append(WhisperSeg(float(s.start), float(s.end), txt))
    return out


# ----------------------------
# Align diarization segments with whisper segments
# ----------------------------
def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

@dataclass
class LabeledUtterance:
    speaker: str
    start: float
    end: float
    text: str

def assign_speaker_to_whisper(
    whisper_segs: List[WhisperSeg],
    diar_segs: List[Segment],
    diar_labels: np.ndarray,
    label_map: Dict[int, str]
) -> List[LabeledUtterance]:
    if not diar_segs:
        return [LabeledUtterance("Unknown", s.start, s.end, s.text) for s in whisper_segs]

    out = []
    for w in whisper_segs:
        # find diar segment with max overlap
        best_i = None
        best_ov = 0.0
        for i, d in enumerate(diar_segs):
            ov = overlap(w.start, w.end, d.start, d.end)
            if ov > best_ov:
                best_ov = ov
                best_i = i

        if best_i is None or best_ov <= 0.0:
            spk = "Unknown"
        else:
            spk_id = int(diar_labels[best_i])
            spk = label_map.get(spk_id, f"Speaker {spk_id}")

        out.append(LabeledUtterance(spk, w.start, w.end, w.text))
    return out

def merge_consecutive_same_speaker(utts: List[LabeledUtterance], gap_s: float = 0.4) -> List[LabeledUtterance]:
    if not utts:
        return []
    merged = [utts[0]]
    for u in utts[1:]:
        prev = merged[-1]
        if u.speaker == prev.speaker and (u.start - prev.end) <= gap_s:
            prev.end = max(prev.end, u.end)
            prev.text = (prev.text + " " + u.text).strip()
        else:
            merged.append(u)
    return merged


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Agent vs Customer Diarization + Transcription", layout="wide")

st.title("Diarization (Agent vs Customer) + Transcription dari Link Audio")
st.write("Input link audio → download → transkripsi (Whisper) → pisahkan Agent/Customer (VAD + speaker embedding + clustering).")

with st.sidebar:
    st.header("Pengaturan")
    model_size = st.selectbox("Whisper model", ["tiny", "base", "small", "medium"], index=2)
    language = st.selectbox("Language", ["id", "en", "auto"], index=0)
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    compute_type = st.selectbox("Compute type (cpu)", ["int8", "int8_float16", "float16", "float32"], index=0)
    vad_aggr = st.slider("VAD aggressiveness", 0, 3, 2)

audio_url = st.text_input("Link audio (http/https):", placeholder="https://.../call_recording.mp3")

colA, colB = st.columns([1, 2])
run = colA.button("Proses", type="primary")

if run:
    if not audio_url.strip().lower().startswith(("http://", "https://")):
        st.error("URL harus diawali http:// atau https://")
        st.stop()

    tmpdir = tempfile.mkdtemp(prefix="callai_")
    try:
        raw_path = os.path.join(tmpdir, "input_audio")
        wav_path = os.path.join(tmpdir, "audio_16k_mono.wav")

        with st.status("Download audio...", expanded=False):
            download_file(audio_url, raw_path)

        with st.status("Konversi audio (FFmpeg) ...", expanded=False):
            ffmpeg_to_wav16k_mono(raw_path, wav_path)

        with st.status("VAD segmentation ...", expanded=False):
            diar_segs = vad_segments(wav_path, vad_aggressiveness=vad_aggr)

        with st.status("Speaker embedding + clustering (2 speaker) ...", expanded=False):
            if device == "cuda" and not torch.cuda.is_available():
                st.warning("CUDA dipilih tapi tidak tersedia. Pakai CPU.")
                device_eff = "cpu"
            else:
                device_eff = device

            embs = get_embeddings_for_segments(wav_path, diar_segs, device=device_eff)
            diar_labels = cluster_two_speakers(embs)
            label_map = label_agent_customer(diar_segs, diar_labels)

        with st.status("Transkripsi (Whisper) ...", expanded=False):
            lang_eff = None if language == "auto" else language
            whisper_segs = transcribe_whisper(
                wav_path,
                model_size=model_size,
                device=device_eff,
                compute_type=compute_type if device_eff == "cpu" else "float16",
                language=lang_eff
            )

        with st.status("Align diarization ↔ transcription ...", expanded=False):
            utts = assign_speaker_to_whisper(whisper_segs, diar_segs, diar_labels, label_map)
            utts = merge_consecutive_same_speaker(utts)

        st.success("Selesai.")

        # Display summary
        st.subheader("Ringkasan Label Speaker")
        st.write(label_map)

        # Show transcript
        st.subheader("Transkrip (Agent vs Customer)")
        transcript_lines = []
        for u in utts:
            line = f"[{u.start:0.2f}–{u.end:0.2f}] {u.speaker}: {u.text}"
            transcript_lines.append(line)
        transcript_text = "\n".join(transcript_lines)
        st.text_area("Output", transcript_text, height=350)

        # Table view
        st.subheader("Tabel")
        table = [{"start": u.start, "end": u.end, "speaker": u.speaker, "text": u.text} for u in utts]
        st.dataframe(table, use_container_width=True)

        # Download
        st.download_button(
            "Download TXT",
            data=transcript_text.encode("utf-8"),
            file_name="transcript_agent_customer.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
