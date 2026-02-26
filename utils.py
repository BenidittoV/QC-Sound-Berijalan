import os
import re
import tempfile
from pathlib import Path
import requests

def safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return name[:180] if name else "audio"

def download_file(url: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    # try infer filename from headers or url
    fname = None
    cd = r.headers.get("content-disposition", "")
    if "filename=" in cd:
        fname = cd.split("filename=")[-1].strip().strip('"')

    if not fname:
        fname = url.split("?")[0].split("/")[-1]
    fname = safe_filename(fname)
    if "." not in fname:
        fname += ".audio"

    out_path = str(Path(out_dir) / fname)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return out_path

def temp_dir() -> str:
    d = tempfile.mkdtemp(prefix="call_")
    return d
