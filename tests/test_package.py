import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hcawn_fc import Client, Local, Result, RateLimited, available_models, _native


def test_result_roundtrip():
    d = {"status": "ok", "model": "int8", "confidence": 0.99, "latency_ms": 5, "reason": None,
         "calls": [{"name": "show_map", "arguments": {"query": "Paris"}}], "held_calls": []}
    r = Result.from_dict(d)
    assert r.ok and str(r.calls[0]) == "show_map(query='Paris')" and r.to_dict() == d


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        code, out = (429, {"detail": "slow down"}) if body["query"] == "limit" else (
            200, {"status": "ok", "calls": [{"name": "open_wifi_settings", "arguments": {}}], "model": body.get("model")})
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps(out).encode())


@pytest.fixture
def server():
    s = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_port}"
    s.shutdown()


def test_client(server):
    c = Client(server)
    assert c.call("wifi", model="int4").model == "int4"
    with pytest.raises(RateLimited):
        c.call("limit")


def test_local_validation():
    if _native.lib_path() is None or not available_models():
        pytest.skip("no engine or weights in this build")
    m = Local(warmup=False)
    for bad in ("", "a\x01b", "x" * 400):
        with pytest.raises(ValueError):
            m.call(bad)
    r = m.call("Turn on the flashlight")
    assert r.status in ("ok", "held", "none") and r.latency_ms >= 0


def test_isa_override(monkeypatch):
    for isa in _native.available():
        monkeypatch.setenv("HCAWN_FC_ISA", isa)
        assert _native.best_isa() == isa
