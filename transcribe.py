from typing import List, Dict, Optional
from faster_whisper import WhisperModel

def load_whisper(model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
    return WhisperModel(model_size, device=device, compute_type=compute_type)

def transcribe_with_timestamps(
    model: WhisperModel,
    audio_path: str,
    language: Optional[str] = None,
) -> List[Dict]:
    """
    Returns: [{"start": float, "end": float, "text": str}, ...]
    """
    segments, _info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
        beam_size=1,
        condition_on_previous_text=False,
    )
    out = []
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue
        out.append({"start": float(s.start), "end": float(s.end), "text": text})
    return out
