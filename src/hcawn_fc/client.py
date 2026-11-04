"""Client for the hosted API (https://cawnai.com). No third-party dependencies."""
import json
import urllib.error
import urllib.request

from . import __version__
from .errors import ApiError, RateLimited, ServerBusy
from .result import Result

DEFAULT_URL = "https://cawnai.com"


class Client:
    """
        >>> from hcawn_fc import Client
        >>> r = Client().call("Turn on the flashlight")
        >>> r.status, [str(c) for c in r.calls]
        ('ok', ['turn_on_flashlight()'])
    """

    def __init__(self, base_url=DEFAULT_URL, model=None, timeout=30.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _request(self, path, body=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base_url + path, data=data, method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json", "Accept": "application/json",
                                              "User-Agent": f"hcawn-fc/{__version__}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode("utf-8")).get("detail", "")
            except Exception:
                detail = ""
            msg = detail if isinstance(detail, str) and detail else f"HTTP {e.code}"
            if e.code == 429:
                raise RateLimited(msg, e.code) from None
            if e.code == 503:
                raise ServerBusy(msg, e.code) from None
            if e.code == 422:
                raise ValueError(msg) from None
            raise ApiError(msg, e.code) from None
        except urllib.error.URLError as e:
            raise ApiError(f"cannot reach {self.base_url}: {e.reason}") from None

    def call(self, query, model=None, tz_offset_min=None):
        """Send one query; returns a Result. tz_offset_min: UTC offset in minutes (used to resolve 'tomorrow', '2pm')."""
        if tz_offset_min is None:
            import time
            tz_offset_min = int(-(time.altzone if time.localtime().tm_isdst > 0 else time.timezone) / 60)
        body = {"query": query, "tz_offset_min": tz_offset_min}
        m = model or self.model
        if m:
            body["model"] = m
        return Result.from_dict(self._request("/call", body))

    def models(self):
        """Models offered by the server (id, size, accuracy)."""
        return self._request("/meta").get("models", [])

    def health(self):
        return self._request("/health")
