from typing import List, Dict, Optional
from faster_whisper import WhisperModel

def load_whisper(model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
    """
    model_size: tiny | base | small | medium | large-v3 (lebih berat)
    """
    return WhisperModel(model_size, device=device, compute_type=compute_type)

def transcribe_segment(
    model: WhisperModel,
    audio_path: str,
    start_s: float,
    end_s: float,
    language: Optional[str] = None,
) -> str:
    """
    Transcribe only a slice by using vad_filter + condition_on_previous_text=False.
    faster-whisper doesn't natively accept time slicing,
    so we pass the full file but constrain via 'clip_timestamps' (supported by faster-whisper >= 0.10).
    """
    segments, _info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
        condition_on_previous_text=False,
        clip_timestamps=[(start_s, end_s)],
        beam_size=1,
    )
    text_parts = []
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            text_parts.append(t)
    return " ".join(text_parts).strip()

def transcribe_full(model: WhisperModel, audio_path: str, language: Optional[str] = None) -> str:
    segments, _ = model.transcribe(audio_path, language=language, vad_filter=True, beam_size=1)
    return " ".join([(s.text or "").strip() for s in segments if (s.text or "").strip()]).strip()
