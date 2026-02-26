import streamlit as st
import pandas as pd
from utils import download_file, temp_dir
from diarize import diarize_2speakers
from transcribe import load_whisper, transcribe_segment

st.set_page_config(page_title="Call Diarization + Transcript", layout="wide")

st.title("Call Diarization (Customer vs Agent) + Transcript")

with st.sidebar:
    st.header("Settings")
    whisper_size = st.selectbox("Whisper model", ["tiny", "base", "small"], index=1)
    language = st.selectbox("Language", ["auto", "id", "en", "ja"], index=0)
    st.caption("Model lebih besar = lebih akurat tapi lebih berat di Streamlit Cloud.")

audio_url = st.text_input("Audio URL (mp3/wav/m4a):", placeholder="https://.../call.mp3")

run = st.button("Process", type="primary", disabled=not bool(audio_url.strip()))

@st.cache_resource
def _get_whisper(model_size: str):
    return load_whisper(model_size=model_size, device="cpu", compute_type="int8")

def _lang_opt(lang: str):
    return None if lang == "auto" else lang

if run:
    workdir = temp_dir()
    st.write(f"Working dir: `{workdir}`")

    with st.status("Downloading audio...", expanded=False) as status:
        try:
            audio_path = download_file(audio_url.strip(), workdir)
            status.update(label=f"Downloaded: {audio_path}", state="complete")
        except Exception as e:
            status.update(label="Download failed", state="error")
            st.exception(e)
            st.stop()

    st.audio(audio_path)

    with st.status("Running speaker diarization (2 speakers)...", expanded=False) as status:
        try:
            diar_segments = diarize_2speakers(audio_path)
            if not diar_segments:
                raise RuntimeError("No diarization segments found. Audio may be too short or silent.")
            status.update(label=f"Diarization done. Segments: {len(diar_segments)}", state="complete")
        except Exception as e:
            status.update(label="Diarization failed", state="error")
            st.exception(e)
            st.stop()

    # Map speakers to roles (simple heuristic: speaker with more total duration = Agent)
    durations = {}
    for s in diar_segments:
        durations[s["speaker"]] = durations.get(s["speaker"], 0.0) + (s["end"] - s["start"])
    speakers_sorted = sorted(durations.items(), key=lambda x: x[1], reverse=True)
    agent_spk = speakers_sorted[0][0]
    cust_spk = speakers_sorted[1][0] if len(speakers_sorted) > 1 else None

    role_map = {}
    role_map[agent_spk] = "AGENT"
    if cust_spk:
        role_map[cust_spk] = "CUSTOMER"

    whisper = _get_whisper(whisper_size)

    st.subheader("Transcript per segment")
    rows = []
    with st.status("Transcribing segments...", expanded=False) as status:
        try:
            for i, seg in enumerate(diar_segments, start=1):
                start = float(seg["start"])
                end = float(seg["end"])
                spk = seg["speaker"]
                role = role_map.get(spk, spk)

                # Skip ultra-short segments (noise)
                if end - start < 0.6:
                    continue

                text = transcribe_segment(
                    whisper,
                    audio_path=audio_path,
                    start_s=start,
                    end_s=end,
                    language=_lang_opt(language),
                )

                if text.strip():
                    rows.append(
                        {
                            "idx": i,
                            "start_s": round(start, 2),
                            "end_s": round(end, 2),
                            "speaker_raw": spk,
                            "speaker_role": role,
                            "text": text,
                        }
                    )

            status.update(label=f"Transcription done. Rows: {len(rows)}", state="complete")
        except Exception as e:
            status.update(label="Transcription failed", state="error")
            st.exception(e)
            st.stop()

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=420)

    st.subheader("Per-speaker combined")
    if not df.empty:
        agent_text = " ".join(df.loc[df["speaker_role"] == "AGENT", "text"].tolist()).strip()
        cust_text = " ".join(df.loc[df["speaker_role"] == "CUSTOMER", "text"].tolist()).strip()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### AGENT")
            st.write(agent_text if agent_text else "(empty)")
        with c2:
            st.markdown("### CUSTOMER")
            st.write(cust_text if cust_text else "(empty)")

        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="diarized_transcript.csv",
            mime="text/csv",
        )

    st.caption("Heuristik role mapping: pembicara dengan durasi total lebih besar dianggap AGENT. Kalau kebalik, ganti mapping manual di kode.")
