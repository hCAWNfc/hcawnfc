"""Local model: runs on the CPU, no network and no account needed."""
import datetime
import json
import os
import re
import threading
import time

from . import _native
from .errors import NativeUnavailable, WeightsNotFound
from .result import Call, Result

_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_HERE = os.path.dirname(os.path.abspath(__file__))
MAX_CHARS = 300


def _cache_dir():
    return os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "hcawn_fc")


def find_weights(model="int8", model_path=None):
    """Path to the .bin file: explicit path, then the package (data/<model>.bin), then ~/.cache/hcawn_fc/<model>.bin."""
    if model_path:
        p = os.path.expanduser(model_path)
        if not os.path.isfile(p):
            raise WeightsNotFound(f"model file not found: {p}")
        return p
    for p in (os.path.join(_HERE, "data", f"{model}.bin"), os.path.join(_cache_dir(), f"{model}.bin")):
        if os.path.isfile(p):
            return p
    raise WeightsNotFound(
        f"no local weights for '{model}'. Run: hcawn-fc download {model}   (saves to {_cache_dir()}), "
        f"or pass Local(model_path=...), or use the hosted API: hcawn_fc.Client().")


def available_models():
    out = []
    for m in ("int8", "int4"):
        try:
            find_weights(m); out.append(m)
        except WeightsNotFound:
            pass
    return out


def _refresh_context(template, now):
    c = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?", now.strftime("%Y-%m-%dT%H:%M:%S"), template)
    return re.sub("|".join(_DAYS), _DAYS[now.weekday()], c)


class Local:
    """
        >>> from hcawn_fc import Local
        >>> m = Local()                                   # int8 model, 1 thread
        >>> r = m.call("Add a calendar event Team Sync for next Monday at 2pm")
        >>> r.status, str(r.calls[0])

    model:      "int8" (default, fastest on most CPUs) or "int4" (smaller file)
    model_path: explicit path to a .bin file (overrides the bundled weights)
    threads:    CPU threads for the engine (<=0 means all cores)
    min_conf:   calls with confidence below this are held back (status "held"); 0.9 is the calibrated default
    """

    def __init__(self, model="int8", model_path=None, tools_path=None, threads=1, min_conf=0.9, score="min_vt", warmup=True):
        lib = _native.lib_path()
        if lib is None:
            raise NativeUnavailable(
                "the local engine is not available in this install (it ships for Linux x86-64). "
                "Use hcawn_fc.Client() for the hosted API.")
        self.model, self.min_conf, self.score, self.isa = model, float(min_conf), score, _native.best_isa()
        self.weights = find_weights(model, model_path)
        tp = tools_path or os.path.join(_HERE, "data", "tools.json")
        if not os.path.isfile(tp):
            raise WeightsNotFound(f"tool definitions not found: {tp}")
        blob = json.load(open(tp, encoding="utf-8"))
        self.tools, self._ctx = blob["tools"], blob.get("context", "")
        from ._core.hcawn_cpp import CppModel, PrefixCache
        self._cm = CppModel(self.weights, lib_path=lib, threads=threads)
        self._cache = PrefixCache(self._cm)
        self._lock = threading.Lock()                      # the engine is not thread-safe: calls are serialized
        if warmup:
            self._generate("turn on the flashlight", datetime.datetime.now())

    @property
    def tool_names(self):
        return [t["name"] for t in self.tools]

    def _generate(self, query, now):
        from ._core.fc_common import render_prompt, parse_calls
        from ._core.fc_conf import scores
        from ._core.hcawn_cpp import encode_prompt, cpp_generate
        ctx = _refresh_context(self._ctx, now)
        conf = {}
        ex = {"tools": self.tools, "query": query, "context": ctx}
        out = cpp_generate(self._cm, encode_prompt(render_prompt(ex)), self.tools, 500, self._cache, query=query.strip(), context=ctx, conf=conf)
        return parse_calls(out), scores(conf.get("steps", []))[self.score], bool(conf.get("vetoed"))

    def call(self, query, now=None, min_conf=None):
        """One query -> Result. now: datetime used to resolve 'tomorrow' / 'next Monday' (default: current local time)."""
        query = " ".join(str(query).split())
        if not query or any(ord(c) < 32 for c in query):
            raise ValueError("query must be a non-empty single line of text")
        if len(query) > MAX_CHARS:
            raise ValueError(f"query is longer than {MAX_CHARS} characters")
        thr = self.min_conf if min_conf is None else float(min_conf)
        now = now or datetime.datetime.now()
        t0 = time.perf_counter()
        with self._lock:
            calls, score, vetoed = self._generate(query, now)
        ms = round((time.perf_counter() - t0) * 1000)
        mk = lambda xs: [Call(c["name"], c.get("arguments") or {}) for c in xs]
        if not calls:
            return Result("none", latency_ms=ms, model=self.model, reason="no keyword match" if vetoed else "no action fits this request")
        conf = round(float(score), 2)
        if score < thr:
            return Result("held", held_calls=mk(calls), confidence=conf, latency_ms=ms, model=self.model, reason="not sure enough")
        return Result("ok", calls=mk(calls), confidence=conf, latency_ms=ms, model=self.model)
