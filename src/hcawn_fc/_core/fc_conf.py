#!/usr/bin/env python3
# ==========================================================================
# fc_conf.py - confidence score and call rejection for function calling (no extra training).
# The score is computed from the probabilities the model already gives, but ONLY where it really chooses (bytes forced by the grammar are not counted).
#   min_res   lowest probability of the chosen byte among the allowed bytes (the grammar renormalizes)
#   min_full  lowest probability of the chosen byte among ALL bytes (catches the grammar forcing the model into something it does not want)
#   mean_lp   mean log-probability (length-normalized)
#   tool_p    product of the probabilities when choosing the tool name
#   val_min   lowest probability inside the argument values
# Calibration (measures everything on your set; one half of the set picks the threshold, the other half checks it):
#   python fc_conf.py calibrate model3_int8.bin fc_mobile_eval.jsonl [--procs 8] [--n N]
#   python fc_conf.py analyze conf_results.json          # compares combined scores from already measured results (no model)
# Use in the chat:  python fc_chat.py model3_int8.bin --score min_full --min-conf 0.5
# The "out of scope" test is built automatically: the tool that is in the gold answer is removed from the query's tool list; the correct behaviour is to refuse.
# ==========================================================================
import sys, os, json, math, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

SCORE_NAMES = ["min_res", "min_full", "mean_lp", "tool_p", "val_min", "min_vt", "val_or_tool"]


def scores(steps):
    if not steps:
        return {k: 1.0 for k in SCORE_NAMES}
    pr = [s[1] for s in steps]
    pf = [s[2] for s in steps]
    name = [s[1] for s in steps if s[0] == "NAME"]
    val = [s[1] for s in steps if s[0] == "VAL"]
    v = min(val) if val else 1.0
    t = float(np.prod(name)) if name else 1.0
    return {"min_vt": min(v, t),                    # min(values, tool choice)
            "val_or_tool": v if val else t,         # if there are no argument values (e.g. flashlight), the tool choice is used
            "min_res": min(pr), "min_full": min(pf),
            "mean_lp": float(np.exp(np.mean(np.log(np.maximum(pr, 1e-9))))),     # geometric mean (0..1)
            "tool_p": float(np.prod(name)) if name else 1.0,
            "val_min": min(val) if val else 1.0}


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def pick_threshold(s, ok, target):
    """Lowest threshold (highest coverage) with accuracy among answered >= target. Returns None if there is none."""
    s, ok = np.asarray(s, float), np.asarray(ok, bool)
    best = None
    for t in sorted(set(s.tolist()), reverse=True):
        m = s >= t
        if m.sum() >= 10 and ok[m].mean() >= target:
            best = t
    return best


# ------------------------------------------------------------------ calibration
_PROG = None


def _work(task):
    binp, rows, jobs = task
    from .fc_common import render_prompt, score_one
    from .hcawn_cpp import CppModel, PrefixCache, encode_prompt, cpp_generate
    cm = CppModel(binp, threads=1)
    cache = PrefixCache(cm)
    out = []
    for (i, kind) in jobs:
        ex = rows[i]
        tools = ex["tools"]
        if kind == "irr":
            gold = {c["name"] for c in ex["calls"]}
            tools = [t for t in tools if t["name"] not in gold]
        e2 = dict(ex, tools=tools)
        conf = {}
        o = cpp_generate(cm, encode_prompt(render_prompt(e2)), tools, 500, cache,
                         query=str(ex["query"]).strip(), context=ex.get("context", ""), conf=conf)
        sc = scores(conf.get("steps", []))
        empty = o.strip() == "[]"
        ex_ok = None if kind == "irr" else bool(score_one(o, ex["calls"])["exact"])
        out.append((kind, i, ex_ok, empty, sc, o))
        if _PROG is not None:
            cnt, tot, lock = _PROG
            with lock:
                cnt.value += 1
                if cnt.value % 200 == 0:
                    print(f"  {cnt.value}/{tot}", flush=True)
    return out


def cmd_calibrate(argv):
    flags = {"--procs", "--n"}
    args, skip = [], False
    for a in argv:
        if skip: skip = False; continue
        if a in flags: skip = True; continue
        if a.startswith("--"): continue
        args.append(a)
    binp, path = args[0], args[1]
    g = lambda n, d: int(argv[argv.index(n) + 1]) if n in argv else d
    procs = g("--procs", 1) or (os.cpu_count() or 1)
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    random.Random(42).shuffle(rows)
    rows = rows[:g("--n", 10 ** 9)]
    jobs = [(i, "in") for i in range(len(rows))]
    for i, ex in enumerate(rows):                                  # "out of scope": remove the tool(s) of the gold answer
        gold = {c["name"] for c in ex["calls"]}
        if gold and any(t["name"] not in gold for t in ex["tools"]) and all(c["name"] in {t["name"] for t in ex["tools"]} for c in ex["calls"]):
            jobs.append((i, "irr"))
    print(f"in scope: {len(rows)} | out of scope (tool removed): {len(jobs) - len(rows)}")
    shards = [jobs[p::procs] for p in range(procs)]
    tasks = [(binp, rows, sh) for sh in shards if sh]
    t0 = time.perf_counter()
    res = []
    if len(tasks) == 1:
        res = _work(tasks[0])
    else:
        import multiprocessing as mp
        global _PROG
        ctx = mp.get_context("fork")
        _PROG = (ctx.Value("i", 0), len(jobs), ctx.Lock())
        with ctx.Pool(len(tasks)) as pool:
            for o in pool.imap_unordered(_work, tasks):
                res += o
    print(f"done in {time.perf_counter() - t0:.0f} s\n")
    json.dump([(k, i, ok, e, sc, o) for k, i, ok, e, sc, o in res], open("conf_results.json", "w"), ensure_ascii=False)

    inn = sorted([r for r in res if r[0] == "in"], key=lambda r: r[1])
    irr = sorted([r for r in res if r[0] == "irr"], key=lambda r: r[1])
    ok = np.array([r[2] for r in inn], bool)
    print(f"accuracy in scope: {ok.mean():.1%} | out of scope: the model still makes a call in {1 - np.mean([r[3] for r in irr]):.1%} of cases\n")
    print(f"{'score':10}{'AUROC':>8}   (how well it separates correct from incorrect in scope; 0.5 = random)")
    au = {}
    for k in SCORE_NAMES:
        s = [r[4][k] for r in inn]
        au[k] = auroc(s, ok)
        print(f"{k:10}{au[k]:8.3f}")
    A = np.array([r[1] % 2 == 0 for r in inn], bool)
    Ai = np.array([r[1] % 2 == 0 for r in irr], bool)
    print("\nthe threshold is picked on half A and checked on half B (so we do not evaluate on what we selected on):")
    for k in SCORE_NAMES:
        s = np.array([r[4][k] for r in inn]); si = np.array([r[4][k] for r in irr])
        for target in (0.95, 0.98):
            t = pick_threshold(s[A], ok[A], target)
            if t is None:
                print(f"  {k:9} target {target:.0%}: no threshold"); continue
            m = s[~A] >= t
            accB = ok[~A][m].mean() if m.any() else float("nan")
            ab = (si[~Ai] < t).mean() if (~Ai).any() else float("nan")
            print(f"  {k:9} target {target:.0%}: threshold {t:.3f} | B: answers {m.mean():.0%} of examples, correct among answered {accB:.1%}"
                  f" | out of scope refused {ab:.0%}")
    best = max(au, key=lambda k: au[k] if au[k] == au[k] else -1)
    s = np.array([r[4][best] for r in inn])
    print(f"\nbest score by AUROC: {best}. Coverage / accuracy for a few thresholds (whole set, informational):")
    for t in (0.2, 0.4, 0.6, 0.8, 0.9):
        m = s >= t
        print(f"  threshold {t:.1f}: answers {m.mean():5.1%} | correct among answered {ok[m].mean() if m.any() else float('nan'):6.1%}")
    print(f"\nusage: python fc_chat.py model3_int8.bin --score {best} --min-conf <threshold>   (per-example results: conf_results.json)")


def combos(sc):
    v, t = sc["val_min"], sc["tool_p"]
    return {"val_min": v,
            "min(val,tool)": min(v, t),
            "val*tool": v * t,
            "val_or_tool_alt": v if v < 0.9999 else t,          # if there are no argument values (v=1), use the tool choice
            "min(val,sqrt(tool))": min(v, t ** 0.5)}


def cmd_analyze(path):
    """Does not run the model: reads conf_results.json from 'calibrate' and compares combined scores."""
    res = json.load(open(path))
    inn = sorted([r for r in res if r[0] == "in"], key=lambda r: r[1])
    irr = sorted([r for r in res if r[0] == "irr"], key=lambda r: r[1])
    ok = np.array([r[2] for r in inn], bool)

    def noargs(o):
        try:
            c = json.loads(o)
            return bool(c) and all(not (x.get("arguments") or {}) for x in c)
        except Exception:
            return False
    z = np.array([noargs(r[5]) for r in inn], bool)
    print(f"in scope {len(inn)} | accuracy {ok.mean():.1%} | answers WITHOUT any arguments: {z.sum()} (accuracy among them {ok[z].mean() if z.any() else float('nan'):.1%})\n")
    C = {k: np.array([combos(r[4])[k] for r in inn]) for k in combos(inn[0][4])}
    Ci = {k: np.array([combos(r[4])[k] for r in irr]) for k in C}
    A = np.array([r[1] % 2 == 0 for r in inn], bool); Ai = np.array([r[1] % 2 == 0 for r in irr], bool)
    print(f"{'score':22}{'AUROC':>7}{'AUROC no-arg':>15}")
    for k, s in C.items():
        print(f"{k:22}{auroc(s, ok):7.3f}{(auroc(s[z], ok[z]) if z.sum() > 20 else float('nan')):15.3f}")
    print("\nthreshold on half A, check on B (target 95%):")
    print(f"{'score':22}{'thresh':>7}{'answers':>10}{'correct':>8}{'out-of-scope refused':>24}{'no-arg correct among answered':>31}")
    for k, s in C.items():
        t = pick_threshold(s[A], ok[A], 0.95)
        if t is None:
            print(f"{k:22}  no threshold"); continue
        m = s[~A] >= t
        mz = m & z[~A]
        print(f"{k:22}{t:7.3f}{m.mean():10.0%}{ok[~A][m].mean():8.1%}{(Ci[k][~Ai] < t).mean():24.0%}"
              f"{(ok[~A][mz].mean() if mz.sum() else float('nan')):31.1%}")


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "calibrate":
        cmd_calibrate(sys.argv[2:])
    elif len(sys.argv) >= 3 and sys.argv[1] == "analyze":
        cmd_analyze(sys.argv[2])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
