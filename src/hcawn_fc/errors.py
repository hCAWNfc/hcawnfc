class HcawnError(Exception):
    """Base class for all errors raised by this package."""


class NativeUnavailable(HcawnError):
    """The native engine is not available on this platform (use Client, the hosted API)."""


class WeightsNotFound(HcawnError):
    """No model weights found for local inference."""


class ApiError(HcawnError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class RateLimited(ApiError):
    """Too many requests (HTTP 429)."""


class ServerBusy(ApiError):
    """The server is busy (HTTP 503); retry in a few seconds."""
