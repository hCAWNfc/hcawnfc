"""hcawn-fc: a tiny function-calling model for phone actions, on CPU, plus a client for the hosted API.

    from hcawn_fc import Local, Client
    Local().call("Turn on the flashlight")      # runs on your CPU, no network
    Client().call("Turn on the flashlight")     # hosted API (cawnai.com)
"""
try:
    from importlib.metadata import version as _v
    __version__ = _v("hcawn-fc")
except Exception:                                    # not installed (running from a source checkout)
    __version__ = "0.0.0+dev"

from .client import Client
from .errors import ApiError, HcawnError, NativeUnavailable, RateLimited, ServerBusy, WeightsNotFound
from .local import Local, available_models
from .result import Call, Result
from .weights import download_weights

__all__ = ["Local", "Client", "Result", "Call", "available_models", "download_weights", "HcawnError", "ApiError", "RateLimited", "ServerBusy",
           "NativeUnavailable", "WeightsNotFound", "__version__"]
