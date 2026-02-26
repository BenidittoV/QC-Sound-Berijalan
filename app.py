import streamlit as st
import pandas as pd
from utils import download_file, temp_dir
from diarize import split_to_channels
from transcribe import load_whisper, transcribe_with_timestamps

st.set_page_config(page_title="Call Transcript (Agent vs Customer)", layout="wide")
st.title("Agent vs Customer Transcript (Dual-Channel)")

with st.sidebar:
    st.header("Settings")
    whisper_size = st.selectbox("Whisper model", ["tiny", "base", "small"], index=1)
    language = st.selectbox("Language", ["auto", "id", "en", "ja"], index=0)
    swap_roles = st.checkbox("Swap Agent/Customer (if channel mapping is reversed)", value=False)
    st.caption("Mode ini butuh audio STEREO (2 channel). Kalau MONO, tidak bisa pisahkan speaker tanpa diarization model/token.")

audio_url = st.text_input("Audio URL (mp3/wav/m4a):", placeholder="https://.../call.mp3")
run = st.button("Process", type="primary", disabled=not bool(audio_url.strip()))

@st.cache_resource
def _get_whisper(size: str):
    return load_whisper(model_size=size, device="cpu", compute_type="int8")

def _lang_opt(lang: str):
    return None if lang == "auto" else lang

def merge_by_time(agent_segs, cust_segs):
    rows = []
    for s in agent_segs:
        rows.append({"start_s": s["start"], "end_s": s["end"], "speaker_role": "AGENT", "text": s["text"]})
    for s in cust_segs:
        rows.append({"start_s": s["start"], "end_s": s["end"], "speaker_role": "CUSTOMER", "text": s["text"]})
    rows.sort(key=lambda x: (x["start_s"], x["end_s"]))
    # beautify time + index
    for i, r in enumerate(rows, 1):
        r["idx"] = i
        r["start_s"] = round(r["start_s"], 2)
        r["end_s"] = round(r["end_s"], 2)
    return rows

if run:
    workdir = temp_dir()

    with st.status("Downloading audio...", expanded=False) as status:
        audio_path = download_file(audio_url.strip(), workdir)
        status.update(label=f"Downloaded: {audio_path}", state="complete")

    st.audio(audio_path)

    with st.status("Splitting channels...", expanded=False) as status:
        ch0, ch1, nchan = split_to_channels(audio_path, out_dir=workdir)
        status.update(label=f"Channels detected: {nchan}", state="complete")

    whisper = _get_whisper(whisper_size)

    if nchan == 1:
        st.error("Audio kamu MONO (1 channel). Mode gratis ini tidak bisa memisahkan agent vs customer tanpa diarization model/token.")
        st.info("Kalau rekaman call kamu seharusnya dual-channel, pastikan file aslinya stereo (mis. dari sistem call recorder), bukan hasil convert.")
        # Still show full transcript
        with st.status("Transcribing (mono)...", expanded=False) as status:
            mono_segs = transcribe_with_timestamps(whisper, ch0, language=_lang_opt(language))
            status.update(label=f"Done. Segments: {len(mono_segs)}", state="complete")
        df = pd.DataFrame([{
            "idx": i+1,
            "start_s": round(s["start"], 2),
            "end_s": round(s["end"], 2),
            "speaker_role": "UNKNOWN",
            "text": s["text"]
        } for i, s in enumerate(mono_segs)])
        st.dataframe(df, use_container_width=True, height=420)
        st.stop()

    # stereo: assume ch0=AGENT, ch1=CUSTOMER (can swap)
    agent_path = ch0
    cust_path = ch1
    if swap_roles:
        agent_path, cust_path = cust_path, agent_path

    with st.status("Transcribing AGENT channel...", expanded=False) as status:
        agent_segs = transcribe_with_timestamps(whisper, agent_path, language=_lang_opt(language))
        status.update(label=f"AGENT done. Segments: {len(agent_segs)}", state="complete")

    with st.status("Transcribing CUSTOMER channel...", expanded=False) as status:
        cust_segs = transcribe_with_timestamps(whisper, cust_path, language=_lang_opt(language))
        status.update(label=f"CUSTOMER done. Segments: {len(cust_segs)}", state="complete")

    rows = merge_by_time(agent_segs, cust_segs)
    df = pd.DataFrame(rows)

    st.subheader("Transcript (merged by time)")
    st.dataframe(df[["idx", "start_s", "end_s", "speaker_role", "text"]], use_container_width=True, height=420)

    st.subheader("Per-speaker combined")
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
        file_name="agent_customer_transcript.csv",
        mime="text/csv",
    )
