"""Downloads the model weights from Hugging Face into ~/.cache/hcawn_fc/ (SHA-256 is verified)."""
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

from .errors import HcawnError
from .local import _cache_dir

_HERE = os.path.dirname(os.path.abspath(__file__))


def _manifest():
    p = os.path.join(_HERE, "data", "weights.json")
    if not os.path.isfile(p):
        raise HcawnError("this build has no weights manifest; nothing to download")
    return json.load(open(p, encoding="utf-8"))


def download_weights(model="int8", base_url=None, progress=True):
    """Download the weights for 'int8' or 'int4' (or 'all'). Returns the list of paths. HCAWN_FC_WEIGHTS_URL overrides the source."""
    man = _manifest()
    names = list(man["files"]) if model == "all" else [model]
    base = (base_url or os.environ.get("HCAWN_FC_WEIGHTS_URL") or f"https://huggingface.co/{man['repo']}/resolve/main").rstrip("/")
    out = []
    os.makedirs(_cache_dir(), exist_ok=True)
    for n in names:
        if n not in man["files"]:
            raise HcawnError(f"unknown model '{n}', choose from: {', '.join(man['files'])}, all")
        info = man["files"][n]
        dst = os.path.join(_cache_dir(), f"{n}.bin")
        if os.path.isfile(dst) and _sha(dst) == info["sha256"]:
            if progress:
                print(f"{n}: already downloaded ({dst})")
            out.append(dst)
            continue
        url = f"{base}/{info['file']}"
        if progress:
            print(f"{n}: downloading {url} ({info['size'] / 1e6:.1f} MB)", flush=True)
        tmp = dst + ".part"
        h = hashlib.sha256()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hcawn-fc"}), timeout=60) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b); h.update(b)
        except (urllib.error.URLError, OSError) as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise HcawnError(f"download failed: {e}")
        if h.hexdigest() != info["sha256"]:
            os.remove(tmp)
            raise HcawnError(f"{n}: checksum mismatch, the file was not saved")
        os.replace(tmp, dst)
        if progress:
            print(f"{n}: saved to {dst}")
        out.append(dst)
    return out


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
