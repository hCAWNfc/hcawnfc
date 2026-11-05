#!/usr/bin/env python3
"""Adds the Serbian models to the hcawn-fc package (version 0.2.0). Run it from the fc_pypi folder:  python patch_pkg_sr.py
Safe to run twice. Backups: <file>.bak_sr. Stops with a clear message if a file does not look as expected."""
import os, re, shutil, sys

def edit(path, fn, must=True):
    if not os.path.isfile(path):
        if must: sys.exit(f"missing {path}")
        return
    s = open(path, encoding="utf-8").read()
    new = fn(s)
    if new is None:
        print(f"  {path}: already done"); return
    if not os.path.isfile(path + ".bak_sr"): shutil.copy(path, path + ".bak_sr")
    open(path, "w", encoding="utf-8").write(new)
    print(f"  {path}: ok")

def sub1(s, old, new, what):
    if old not in s:
        sys.exit(f"cannot find the {what} in the file; send me that file")
    return s.replace(old, new, 1)

# 1) tools/sync_core.py: fc_sr into CORE, --sr-int8 / --sr-int4 / --sr-repo
def f_sync(s):
    if "sr-int8" in s: return None
    s = sub1(s, '"fc_augment", "hcawn_cpp"]', '"fc_augment", "fc_sr", "hcawn_cpp"]', "CORE list")
    s = sub1(s, 'ap.add_argument("--int8"); ap.add_argument("--int4")',
             'ap.add_argument("--int8"); ap.add_argument("--int4")\n'
             '    ap.add_argument("--sr-int8"); ap.add_argument("--sr-int4")\n'
             '    ap.add_argument("--sr-repo", default="datatab/hcawn-fc7-sr", help="Hugging Face repo with the Serbian weights (int8.bin, int4.bin)")', "--int8 argument")
    s = sub1(s, 'for tag, p in (("int8", a.int8), ("int4", a.int4)):',
             'for tag, p in (("int8", a.int8), ("int4", a.int4), ("sr-int8", a.sr_int8), ("sr-int4", a.sr_int4)):', "weights loop")
    s = sub1(s, 'files[tag] = {"file": f"{tag}.bin", "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}',
             'files[tag] = {"file": (tag[3:] if tag.startswith("sr-") else tag) + ".bin", "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}\n'
             '            if tag.startswith("sr-"): files[tag]["repo"] = a.sr_repo', "manifest entry")
    return s
edit("tools/sync_core.py", f_sync)

# 2) tools/build_release.sh: pass the Serbian files
def f_rel(s):
    if "--sr-int8" in s: return None
    return sub1(s, "--int4 model5_C_pack.bin --no-bundle", "--int4 model5_C_pack.bin --sr-int8 model9sr_int8.bin --sr-int4 model9sr_C_pack.bin --no-bundle", "sync_core call")
edit("tools/build_release.sh", f_rel, must=False)

# 3) src/hcawn_fc/weights.py: a file may come from its own repo
def f_w(s):
    if 'info.get("repo")' in s: return None
    return sub1(s, 'url = f"{base}/{info[\'file\']}"',
                'src = base if (base_url or os.environ.get("HCAWN_FC_WEIGHTS_URL") or not info.get("repo")) else f"https://huggingface.co/{info[\'repo\']}/resolve/main"\n'
                '        url = f"{src}/{info[\'file\']}"', "download url line")
edit("src/hcawn_fc/weights.py", f_w)

# 4) src/hcawn_fc/local.py: the new names are known models
def f_l(s):
    if "sr-int8" in s: return None
    s = sub1(s, 'for m in ("int8", "int4"):', 'for m in ("int8", "int4", "sr-int8", "sr-int4"):', "available_models loop")
    s = s.replace('model:      "int8" (default, fastest on most CPUs) or "int4" (smaller file)',
                  'model:      "int8" (default, fastest on most CPUs) or "int4" (smaller file); "sr-int8" / "sr-int4" are the Serbian (Latin) models', 1)
    return s
edit("src/hcawn_fc/local.py", f_l)

# 5) src/hcawn_fc/cli.py: help text
def f_c(s):
    if "sr-int8" in s: return None
    s = s.replace('(or: doctor, chat, download [int8|int4|all])', '(or: doctor, chat, download [int8|int4|sr-int8|sr-int4|all])')
    s = s.replace('help="int8 (default) or int4")', 'help="int8 (default), int4, sr-int8 or sr-int4 (Serbian, Latin script)")')
    return s
edit("src/hcawn_fc/cli.py", f_c)

# 6) version 0.2.0
def f_v(s):
    if 'version = "0.2.0"' in s: return None
    return re.sub(r'(?m)^version = "[^"]+"', 'version = "0.2.0"', s, count=1)
edit("pyproject.toml", f_v)

# 7) README: Serbian section
def f_r(s):
    if "## Serbian" in s: return None
    return s.rstrip() + '''

## Serbian (Latin script)

A separate model understands Serbian requests (Latin script, with or without diacritics, typos included). It is not the English model: it was fine-tuned on Serbian only and is not meant for English.

```bash
hcawn-fc download sr-int8        # or sr-int4 (3.9 MB)
hcawn-fc --model sr-int8 "upali lampu"
hcawn-fc --model sr-int8 "zakazi sastanak sa Markom sutra u 15h"
```

```python
from hcawn_fc import Local

m = Local(model="sr-int8")
r = m.call("dodaj kontakt Marko Petrovic 064 123 4567")
print(r.status, r.calls)
```

On 450 requests rewritten as people type them (no diacritics, typos, lowercase), `sr-int8` gets 85.1% exact match and `sr-int4` 84.9%. At the default confidence gate (0.9) the model answers about 80% of requests and about 97% of those answers are exactly right. The weakest tool is `show_map`. The weights are in the Hugging Face repo [datatab/hcawn-fc7-sr](https://huggingface.co/datatab/hcawn-fc7-sr).
'''
edit("README.md", f_r)

# 8) tests
os.makedirs("tests", exist_ok=True)
t = '''"""Serbian rules (no weights and no native engine needed)."""
import json
from hcawn_fc._core import fc_sr, fc_triggers

NAMES = ["open_wifi_settings", "create_contact", "show_map", "create_calendar_event", "send_email", "turn_off_flashlight", "turn_on_flashlight"]


def test_serbian_triggers():
    assert "turn_on_flashlight" in fc_triggers.matched("upali lampu", NAMES)
    assert "send_email" in fc_triggers.matched("posalji mejl na ana@example.com sa temom Zdravo", NAMES)
    assert not fc_triggers.matched("koliko je sati", NAMES)


def test_dates_are_normalised():
    assert "tomorrow" in fc_sr.norm_for_dates("sastanak sutra u 15h")


def test_phone_is_completed():
    out = json.dumps([{"name": "create_contact", "arguments": {"first_name": "A", "phone_number": "555 1212"}}])
    res, changed = fc_sr.fix_phone(out, "dodaj kontakt A B 063 555 1212")
    assert changed and json.loads(res)[0]["arguments"]["phone_number"] == "063 555 1212"


def test_duplicate_call_dropped():
    c = {"name": "turn_on_flashlight", "arguments": {}}
    res, changed = fc_sr.dedupe_calls(json.dumps([c, c]))
    assert changed and len(json.loads(res)) == 1


def test_map_place_taken_from_request():
    out = json.dumps([{"name": "show_map", "arguments": {"query": "Tree Republice, Beogradu"}}])
    res, changed = fc_sr.fix_map_ground(out, "gde je Trg republike u Beogradu")
    assert changed and json.loads(res)[0]["arguments"]["query"] == "Trg republike u Beogradu"
'''
if not os.path.isfile("tests/test_serbian.py"):
    open("tests/test_serbian.py", "w", encoding="utf-8").write(t); print("  tests/test_serbian.py: ok")
print("done. Next: copy the fc files + both Serbian models into from_fc, then bash tools/build_release.sh")
