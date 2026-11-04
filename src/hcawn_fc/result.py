from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Call:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {"name": self.name, "arguments": self.arguments}

    def __str__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"{self.name}({args})"


@dataclass
class Result:
    """Model answer; same shape for the local model and the API.

    status: "ok"   the call(s) are confident enough to act on -> calls
            "held" the model proposed a call but is not sure enough -> held_calls (do not run it without confirmation)
            "none" none of the actions fits the request
    """
    status: str
    calls: List[Call] = field(default_factory=list)
    held_calls: List[Call] = field(default_factory=list)
    confidence: Optional[float] = None
    latency_ms: int = 0
    reason: Optional[str] = None
    model: Optional[str] = None

    @property
    def ok(self):
        return self.status == "ok"

    def to_dict(self):
        return {"status": self.status, "model": self.model, "confidence": self.confidence, "latency_ms": self.latency_ms,
                "reason": self.reason, "calls": [c.to_dict() for c in self.calls], "held_calls": [c.to_dict() for c in self.held_calls]}

    @classmethod
    def from_dict(cls, d):
        mk = lambda xs: [Call(c["name"], c.get("arguments") or {}) for c in (xs or [])]
        return cls(status=d.get("status", "none"), calls=mk(d.get("calls")), held_calls=mk(d.get("held_calls")),
                   confidence=d.get("confidence"), latency_ms=int(d.get("latency_ms") or 0), reason=d.get("reason"), model=d.get("model"))
