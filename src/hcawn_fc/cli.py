"""The `hcawn-fc` command."""
import argparse
import json
import os
import sys

from . import __version__, _native
from .errors import HcawnError


def _print(res, as_json):
    if as_json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return
    if res.status == "ok":
        for c in res.calls:
            print(f"{c}")
        print(f"  confidence {res.confidence}  {res.latency_ms} ms  [{res.model}]")
    elif res.status == "held":
        print("held (not sure enough):", ", ".join(str(c) for c in res.held_calls))
        print(f"  confidence {res.confidence}  {res.latency_ms} ms  [{res.model}]")
    else:
        print("no action:", res.reason or "")


def _backend(a):
    if a.api:
        from .client import Client
        return Client(a.url) if a.url else Client()
    from .local import Local
    from .errors import NativeUnavailable, WeightsNotFound
    try:
        return Local(model=a.model, model_path=a.model_path, threads=a.threads, min_conf=a.min_conf)
    except (NativeUnavailable, WeightsNotFound) as e:
        if a.model_path:
            raise
        print("note: no local model here, using the hosted API (get the weights with: hcawn-fc download).", file=sys.stderr)
        from .client import Client
        return Client(a.url) if a.url else Client()


def doctor():
    from . import local
    print(f"hcawn-fc {__version__}  python {sys.version.split()[0]}  {sys.platform}")
    lib = _native.lib_path()
    print("engine:   ", f"{os.path.basename(lib)} (cpu pick: {_native.best_isa()})" if lib else "NOT BUILT (API client only)")
    models = local.available_models()
    print("weights:  ", ", ".join(models) if models else "not downloaded yet. Run: hcawn-fc download")
    if lib and models:
        try:
            m = local.Local(model=models[0])
            r = m.call("Turn on the flashlight")
            print("self-test:", r.status, ", ".join(str(c) for c in r.calls), f"({r.latency_ms} ms)")
        except HcawnError as e:
            print("self-test: failed:", e); return 1
    elif not lib or not models:
        print("local mode unavailable; hosted API: hcawn-fc --api \"Turn on the flashlight\"")
    return 0


def chat(be, as_json):
    print("Type a request, empty line to quit.")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if not q:
            return
        try:
            _print(be.call(q), as_json)
        except (HcawnError, ValueError) as e:
            print("error:", e)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hcawn-fc", description="Tiny function-calling model for 7 phone actions.")
    ap.add_argument("query", nargs="*", help="request, e.g. \"Turn on the flashlight\" (or: doctor, chat, download [int8|int4|sr-int8|sr-int4|all])")
    ap.add_argument("--api", action="store_true", help="use the hosted API instead of the local model")
    ap.add_argument("--url", help="API base url (default https://cawnai.com)")
    ap.add_argument("--model", default="int8", help="int8 (default), int4, sr-int8 or sr-int4 (Serbian, Latin script)")
    ap.add_argument("--model-path", help="path to a model .bin file")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--min-conf", type=float, default=0.9)
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)
    words = a.query
    if words == ["doctor"]:
        return doctor()
    if words and words[0] == "download":
        from .weights import download_weights
        try:
            download_weights(words[1] if len(words) > 1 else "int8")
            return 0
        except HcawnError as e:
            print("error:", e, file=sys.stderr); return 2
    try:
        be = _backend(a)
        if words == ["chat"] or not words:
            if not words and not sys.stdin.isatty():
                words = [sys.stdin.read().strip()]
            else:
                chat(be, a.json); return 0
        _print(be.call(" ".join(words)), a.json)
        return 0
    except HcawnError as e:
        print("error:", e, file=sys.stderr); return 2
    except ValueError as e:
        print("error:", e, file=sys.stderr); return 2


if __name__ == "__main__":
    sys.exit(main())
