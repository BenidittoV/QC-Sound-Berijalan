from typing import List, Dict
import os
import numpy as np

# SpeechBrain diarization pipeline
from speechbrain.inference.speaker import SpeakerDiarization

def diarize_2speakers(audio_path: str) -> List[Dict]:
    """
    Returns list of segments:
      [{"start": float, "end": float, "speaker": "SPEAKER_0"}, ...]
    """
    # Model diarization speechbrain (download on first run)
    # If streamlit cloud timeout, use shorter audio or lower compute.
    diar = SpeakerDiarization.from_hparams(
        source="speechbrain/speaker-diarization-3.1",
        savedir="pretrained_models/speechbrain_diarization",
        run_opts={"device": "cpu"},
    )

    # Speechbrain returns (timestamps, speakers) in RTTM-like form
    # diar() can output RTTM file path if out_dir set
    out_dir = "diar_out"
    os.makedirs(out_dir, exist_ok=True)

    rttm_path = diar.diarize_file(audio_path, out_dir=out_dir)
    segments = _parse_rttm(rttm_path)
    # Normalize / sort
    segments = sorted(segments, key=lambda x: (x["start"], x["end"]))
    return segments

def _parse_rttm(rttm_path: str) -> List[Dict]:
    segs = []
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            # RTTM: SPEAKER <file-id> 1 <start> <dur> <...> <speaker-id> <...>
            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]
            segs.append({"start": start, "end": start + dur, "speaker": speaker})
    # merge tiny gaps
    return _merge_close(segs, gap=0.25)

def _merge_close(segs: List[Dict], gap: float = 0.25) -> List[Dict]:
    if not segs:
        return segs
    out = [segs[0].copy()]
    for s in segs[1:]:
        last = out[-1]
        if s["speaker"] == last["speaker"] and s["start"] - last["end"] <= gap:
            last["end"] = max(last["end"], s["end"])
        else:
            out.append(s.copy())
    return out
