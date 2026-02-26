from pydub import AudioSegment
import os

def split_to_channels(audio_path: str, out_dir: str):
    """
    Return (ch0_path, ch1_path, channels)
    - If mono: (mono_path, None, 1)
    - If stereo: (left_path, right_path, 2)
    """
    os.makedirs(out_dir, exist_ok=True)

    audio = AudioSegment.from_file(audio_path)
    channels = audio.channels

    if channels == 1:
        mono_path = os.path.join(out_dir, "mono.wav")
        audio.set_frame_rate(16000).set_channels(1).export(mono_path, format="wav")
        return mono_path, None, 1

    # stereo (or more) -> take first two channels
    ch = audio.split_to_mono()
    left = ch[0].set_frame_rate(16000).set_channels(1)
    right = ch[1].set_frame_rate(16000).set_channels(1)

    left_path = os.path.join(out_dir, "ch0_left.wav")
    right_path = os.path.join(out_dir, "ch1_right.wav")
    left.export(left_path, format="wav")
    right.export(right_path, format="wav")
    return left_path, right_path, 2
