# hcawn_cpp.py - Python wrapper around the C++ engine (libhcawn.so) + constrained generation for function calling.
import ctypes, os, json, numpy as np
BOS, EOS, VOC = 256, 257, 258
_here = os.path.dirname(os.path.abspath(__file__))


class CppModel:
    def __init__(self, bin_path, lib_path=None, threads=1, decode_threads=False):
        self.lib = L = ctypes.CDLL(lib_path or os.path.join(_here, "libhcawn.so"))
        L.hc_load.restype = ctypes.c_void_p; L.hc_load.argtypes = [ctypes.c_char_p]
        L.hc_new_state.restype = ctypes.c_void_p; L.hc_new_state.argtypes = [ctypes.c_void_p]
        L.hc_clone_state.restype = ctypes.c_void_p; L.hc_clone_state.argtypes = [ctypes.c_void_p]
        L.hc_free_state.argtypes = [ctypes.c_void_p]
        L.hc_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        L.hc_prefill.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        L.hc_state_bytes.argtypes = [ctypes.c_void_p]; L.hc_state_bytes.restype = ctypes.c_int
        L.hc_set_threads.argtypes = [ctypes.c_int, ctypes.c_int]
        L.hc_set_threads(int(threads), int(bool(decode_threads)))   # threads<=0 => all cores
        self.m = L.hc_load(bin_path.encode())
        if not self.m:
            raise RuntimeError("cannot load " + bin_path)
        self._lg = np.zeros(VOC, np.float32)

    def new_state(self):
        return L_State(self, self.lib.hc_new_state(self.m))


class L_State:
    def __init__(self, cm, p):
        self.cm, self.p = cm, p

    def clone(self):
        return L_State(self.cm, self.cm.lib.hc_clone_state(self.p))

    def feed(self, ids, want_logits=True):
        """Feeds bytes; returns the logits of the last one (or None)."""
        a = np.asarray(ids, dtype=np.int32)
        lg = self.cm._lg if want_logits else None
        self.cm.lib.hc_prefill(self.cm.m, self.p, a.ctypes.data, len(a), lg.ctypes.data if lg is not None else None)
        return lg

    def nbytes(self):
        return self.cm.lib.hc_state_bytes(self.p)

    def __del__(self):
        try:
            self.cm.lib.hc_free_state(self.p)
        except Exception:
            pass


def encode_prompt(text):
    return [BOS] + list(text.encode("utf-8", errors="replace"))


_MARK = list(b"\n\n### Context:")
_MARK2 = list(b"\n\n### Query:")


def split_prefix(ids):
    """Boundary below which the prompt is the same when the tools are the same: everything up to '\\n\\n### Context:' (or '### Query:')."""
    for mk in (_MARK, _MARK2):
        n = len(mk)
        for i in range(len(ids) - n):
            if ids[i:i + n] == mk:
                return i
    return 0


class PrefixCache:
    """Remembers the model state after the tool list. Same tools => that part of the prefill is skipped."""
    def __init__(self, cm, max_items=64):
        self.cm, self.d, self.max = cm, {}, max_items
        self.hits = self.miss = 0

    def start(self, ids):
        cut = split_prefix(ids)
        key = tuple(ids[:cut])
        st = self.d.get(key)
        if st is None:
            self.miss += 1
            st = self.cm.new_state(); st.feed(ids[:cut], want_logits=False)
            if len(self.d) >= self.max:
                self.d.pop(next(iter(self.d)))
            self.d[key] = st
        else:
            self.hits += 1
        s = st.clone()
        return s, ids[cut:]


GROUND_DEFAULT = "title,subject,body,to,first_name,last_name,phone_number,email"
GROUND_ENV = tuple(k for k in os.environ.get("FC_GROUND", GROUND_DEFAULT).split(",") if k)      # FC_GROUND="" turns it off   # e.g. FC_GROUND=title,subject,body,to,first_name,last_name,phone_number,email


VETO_ENV = os.environ.get("FC_VETO", "1") == "1"      # FC_VETO=0 turns it off (without a trigger there is no call)
FIX_ENV = os.environ.get("FC_FIX", "1") == "1"      # FC_FIX=0 turns off the subject rule (fc_fix.py)
DATES_ENV = os.environ.get("FC_DATES", "1") == "1"      # FC_DATES=0 turns it off   # relative dates are computed by a rule (fc_dates.py), not by the model


def cpp_generate(cm, ids, tools, max_new=500, cache=None, typed=True, enforce_required=True, query=None, ground=(), context=None, conf=None):
    """Constrained (grammar) generation on the C++ engine. Same output as constrained_generate.
    If ground is not given, it is taken from the FC_GROUND environment variable (fields whose value must be a literal part of the query; requires query)."""
    from .fc_constrained import FCGrammar
    ground = ground or GROUND_ENV
    from .fc_triggers import restrict
    gtools, force = restrict(tools, str(query)) if query is not None else (tools, False)
    g = FCGrammar(gtools, typed, enforce_required, query, ground)
    g.force_call = force
    if conf is not None and force:
        conf["triggered"] = [t["name"] for t in gtools]
    if cache is not None:
        st, rest = cache.start(ids)
    else:
        st, rest = cm.new_state(), ids
    pending, logits = list(rest), None
    out = []
    for _ in range(max_new):
        al = g.allowed()
        if not al:
            break
        if len(al) == 1:
            nxt = next(iter(al))
        else:
            if pending:
                logits = st.feed(pending); pending = []
            lg = logits.copy()
            m = np.full(VOC, -np.inf, np.float32); m[sorted(al)] = 0.0
            nxt = int((lg + m).argmax())
            if conf is not None:                       # probabilities at the places where the model really chose (not on forced bytes)
                ph = g.phase
                ea = np.exp(lg - lg.max()); p_full = float(ea[nxt] / ea.sum())
                idx = sorted(al); la = lg[idx]; ex = np.exp(la - la.max()); p_res = float(ex[idx.index(nxt)] / ex.sum())
                vpos = len(getattr(getattr(g, "fsm", None), "sbuf", b"")) if ph == "VAL" else 0     # which byte of the value is being chosen
                conf.setdefault("steps", []).append((ph, p_res, p_full, getattr(g, "cur_key", None), vpos,
                                                         chr(nxt) if nxt < 256 else "<EOS>", bytes(t for t in out[-30:] if t < 256).decode("utf-8", errors="replace")))
        if nxt == EOS:
            break
        g.advance(nxt); out.append(nxt); pending.append(nxt)
    text = bytes(t for t in out if t < 256).decode("utf-8", errors="replace")
    if conf is not None:
        conf["raw"] = text                              # model output before the date rules
    if DATES_ENV and query is not None and context is not None:
        from .fc_dates import fix_dates_ex
        text, changed, applied = fix_dates_ex(text, query, context)
        if applied and conf is not None:                # the rule determines the date/time (changed or confirmed it), so the model's uncertainty about those digits is not counted
            def _rule_digit(s):
                return s[0] == "VAL" and s[3] == "datetime" and ((("d" in applied) and s[4] < 10) or (("t" in applied) and 11 <= s[4] <= 18))
            conf["steps"] = [s for s in conf.get("steps", []) if not _rule_digit(s)]
            conf["dates_fixed"] = True
    if FIX_ENV and query is not None:
        from .fc_fix import fix_subject, fix_address, fix_title
        text, ch = fix_subject(text, str(query))
        text, ch2 = fix_address(text, str(query)); ch = ch or ch2
        text, ch3 = fix_title(text, str(query)); ch = ch or ch3
        if conf is not None and ch:
            conf["subject_fixed"] = True
    if VETO_ENV and query is not None and not force:        # no trigger matched: the model may not make a call on its own (e.g. "Hello there" -> flashlight)
        from .fc_triggers import has_trigger_tools, ENABLED as TRIG_ON
        try:
            calls = json.loads(text)
        except Exception:
            calls = []
        if TRIG_ON and calls and has_trigger_tools([t["name"] for t in tools]):
            text = "[]"
            if conf is not None:
                conf["vetoed"] = True
    return text
