# hcawn-fc

A tiny function-calling model (5.7M parameters, byte-level, CPU only) for **7 phone actions**:
`send_email`, `create_calendar_event`, `create_contact`, `show_map`, `open_wifi_settings`, `turn_on_flashlight`, `turn_off_flashlight`.

It runs locally in well under a second, always returns valid JSON, and holds back calls it is not sure about.
Try it in the browser first: **https://cawnai.com**

> **Scope:** a narrow model for a known tool set, not a general function-calling model. It does not take arbitrary tool schemas.

## Install

```bash
pip install hcawn-fc
hcawn-fc doctor
hcawn-fc "Add a calendar event Team Sync tomorrow at 2pm"
```

Python 3.9+, any OS. Until you download the weights, `hcawn-fc` calls the hosted API (free demo). The local CPU engine ships for Linux x86-64, see "Run it locally".

## Run it locally

The local engine runs on your CPU with no network. The weights (6 MB int8, 3.9 MB int4, CC BY 4.0) are hosted on Hugging Face and downloaded once:

```bash
hcawn-fc download            # int8 (default); also: hcawn-fc download int4 | all
hcawn-fc doctor              # shows the engine, the weights and a self-test
```

They are saved to `~/.cache/hcawn_fc/` and checked against a SHA-256 sum. You can also pass your own file with `--model-path` / `Local(model_path=...)`.

## Use it from Python

```python
from hcawn_fc import Local

m = Local()                                   # needs the engine (Linux x86-64) and `hcawn-fc download`
r = m.call("Email dana@example.com subject Lunch body Are we still on for noon?")

r.status        # "ok" | "held" | "none"
r.calls         # [Call(name="send_email", arguments={...})]
r.confidence    # 0.0 - 1.0
r.latency_ms
```

Statuses:

| status | meaning |
|---|---|
| `ok` | the call is confident enough to act on (`r.calls`) |
| `held` | the model proposed a call but is not sure; ask the user first (`r.held_calls`) |
| `none` | none of the 7 actions fits the request |

Relative dates ("tomorrow", "next Monday") are computed from `now` (default: local time). Pass `m.call(q, now=datetime(...))` to override.
`Local(model="int4")` uses the smaller file; `Local(min_conf=0.95)` makes the gate stricter. One `Local` instance handles one request at a time (calls are serialized); use one instance per process for parallelism.

## Hosted API

```python
from hcawn_fc import Client
Client().call("Turn on the flashlight")        # https://cawnai.com, free demo, rate limited. Works on any OS.
```

or `hcawn-fc --api "Turn on the flashlight"`. Raises `RateLimited` / `ServerBusy` / `ApiError`.
Need higher limits or an SLA? Contact us (below).

## Command line

```
hcawn-fc "Show me Paris on the map"      one request
hcawn-fc chat                            interactive
hcawn-fc doctor                          what engine / weights / CPU variant is in use
  --api  --model int4  --json  --min-conf 0.95  --model-path file.bin
```

The engine ships in three builds (AVX2, SSE4.1, generic) and the fastest one your CPU supports is picked at start-up. Override with `HCAWN_FC_ISA=avx2|sse41|generic`.

## How good is it

Strict exact match (tool names and every argument value) on the Mobile Actions eval set, 961 rows, about ±2 points of noise. These numbers are for model **plus** the deterministic layers around it (grammar-constrained decoding, grounding of values in the query, date/time rules, keyword triggers, confidence gate):

| | Exact |
|---|---|
| int8 | 90.9% |
| int8 on rephrased queries (905 rows, paraphrased by a different LLM) | 83.2% |
| model + grammar only, no other layers | 86.5% |

Weaker on unusual phrasings ("tell him that ...", "... saying X"). The confidence gate is not a guarantee: it stops most, not all, wrong calls.

## Data and licences

* Code: Apache-2.0.
* Trained on the **Mobile Actions** dataset (Google, CC BY 4.0) plus query-form augmentation and 16,000 paraphrases of the training queries written by an LLM (Gemini 2.5 Flash Lite) and filtered automatically. Evaluation paraphrases were written by GPT-4o-mini and were never used for training.
* Model weights: CC BY 4.0 (Hugging Face: https://huggingface.co/cawnai/hcawn-fc7). See `NOTICE`.

## Custom models

We build small CPU-only models like this one for **your** tools and your phrasing, and run them on-device or on your own servers.
Contact: https://cawnai.com (contact form at the bottom).
