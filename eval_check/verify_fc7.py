#!/usr/bin/env python3
"""Independent re-check of the hCAWN-FC7 numbers, using ONLY the public pip package `hcawn-fc`
(the same thing a stranger would install) and a separate, simple scorer.

    pip install hcawn-fc && hcawn-fc download int8 && hcawn-fc download int4
    python3 verify_fc7.py ../fc_mobile_eval.jsonl --model int8
    python3 verify_fc7.py ../fc_mobile_eval.jsonl --model int4 --procs 4
    python3 verify_fc7.py rewritten_eval.jsonl --model int8 --name rewritten

Each row of the .jsonl: {"query": str, "context": str (has the date), "tools": [...], "calls": [{"name":..,"arguments":{..}}]}
Scoring (strict exact match): same tool names in the same order AND every argument equal
(strings compared after strip(), numbers as floats). Nothing else is forgiven.
"""
import argparse, collections, datetime, json, math, os, re, statistics, sys, time

DT = re.compile(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})")


def norm(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {str(k): norm(x) for k, x in sorted(v.items())}
    return v


def sig(calls):
    return [(c["name"], norm(c.get("arguments") or {})) for c in calls]


def row_now(row):
    m = DT.search(row.get("context") or "")
    if not m:
        return None
    return datetime.datetime(*map(int, m.groups()))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def work(task):
    model, model_path, threads, min_conf, rows, idxs, allow_long = task
    from hcawn_fc import Local
    if allow_long:                                  # the package refuses queries over 300 chars; the benchmark has some
        import hcawn_fc.local as _loc
        _loc.MAX_CHARS = 10 ** 6
    m = Local(model=model, model_path=model_path, threads=threads, min_conf=min_conf)
    names = set(m.tool_names)
    out = []
    for i in idxs:
        r = rows[i]
        tools_ok = {t.get("name") for t in r.get("tools", [])} <= names if r.get("tools") else True
        now = row_now(r)
        t0 = time.perf_counter()
        try:
            res = m.call(r["query"], now=now)
            err = None
        except Exception as e:                      # keep going, count it as a miss
            res, err = None, repr(e)[:200]
        dt = time.perf_counter() - t0
        pred = []
        status = "error"
        if res is not None:
            status = res.status
            pred = [c.to_dict() for c in (res.calls or res.held_calls)]
        gold = r["calls"]
        out.append({"i": i, "query": r["query"], "status": status, "conf": getattr(res, "confidence", None),
                    "pred": pred, "gold": gold, "exact": int(sig(pred) == sig(gold)),
                    "names": int([c["name"] for c in pred] == [c["name"] for c in gold]),
                    "valid": int(res is not None), "ms": dt * 1000, "tools_ok": tools_ok, "err": err})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--model", default="int8", choices=["int8", "int4"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--procs", type=int, default=1, help="independent processes (each 1 thread); speeds up the run")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--min-conf", type=float, default=0.9)
    ap.add_argument("--allow-long", action="store_true", help="also run queries longer than the package limit of 300 characters (as the site benchmark does)")
    ap.add_argument("--name", default=None, help="label for the output files")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    n = len(rows)
    try:
        import hcawn_fc
        ver = hcawn_fc.__version__
    except ImportError:
        sys.exit("hcawn-fc is not installed:  pip install hcawn-fc")
    print(f"hcawn-fc {ver} | model {a.model} | {n} rows | procs {a.procs}", flush=True)

    t0 = time.time()
    procs = max(1, a.procs)
    tasks = [(a.model, a.model_path, a.threads, a.min_conf, rows, list(range(p, n, procs)), a.allow_long) for p in range(procs)]
    if procs == 1:
        res = work(tasks[0])
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(procs) as pool:
            res = [x for part in pool.map(work, tasks) for x in part]
    res.sort(key=lambda x: x["i"])
    wall = time.time() - t0

    k = sum(x["exact"] for x in res)
    lo, hi = wilson(k, n)
    ans = [x for x in res if x["status"] == "ok"]
    ka = sum(x["exact"] for x in ans)
    ms = [x["ms"] for x in res]
    bad_tools = sum(1 for x in res if not x["tools_ok"])

    print("\n=== RESULT (strict exact match) ===")
    print(f"exact match, model output (no confidence gate): {k}/{n} = {100*k/n:.1f}%   95% CI {100*lo:.1f}-{100*hi:.1f}")
    short = [x for x in res if len(x["query"]) <= 300]
    ks = sum(x["exact"] for x in short)
    print(f"requests up to 300 chars (what the demo accepts): {ks}/{len(short)} = {100*ks/max(1,len(short)):.1f}%   "
          f"| longer than 300: {sum(x['exact'] for x in res if len(x['query'])>300)}/{n-len(short)}")
    print(f"valid JSON/answers: {sum(x['valid'] for x in res)}/{n}   right tool names: {sum(x['names'] for x in res)/n*100:.1f}%")
    if ans:
        print(f"with the gate (min_conf {a.min_conf}): answered {len(ans)}/{n} = {100*len(ans)/n:.1f}%, correct among answered {100*ka/len(ans):.1f}%")
    print(f"latency per request: median {statistics.median(ms):.0f} ms, mean {statistics.mean(ms):.0f} ms (wall {wall:.0f} s, procs {procs})")
    if bad_tools:
        print(f"WARNING: {bad_tools} rows use tools that are not in the package tool list")

    per = collections.defaultdict(lambda: [0, 0])
    for x in res:
        key = x["gold"][0]["name"] if x["gold"] else "(no call)"
        per[key][1] += 1
        per[key][0] += x["exact"]
    print("\nper first gold tool:")
    for t, (c, tot) in sorted(per.items(), key=lambda z: -z[1][1]):
        print(f"  {t:<24} {c:>4}/{tot:<4} {100*c/tot:5.1f}%")

    os.makedirs(a.out, exist_ok=True)
    tag = a.name or os.path.splitext(os.path.basename(a.data))[0]
    base = os.path.join(a.out, f"{tag}_{a.model}")
    with open(base + "_errors.jsonl", "w", encoding="utf-8") as f:
        for x in res:
            if not x["exact"]:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    summ = {"package": ver, "model": a.model, "data": a.data, "n": n, "exact": k, "exact_pct": round(100 * k / n, 2),
            "ci95": [round(100 * lo, 2), round(100 * hi, 2)], "answered": len(ans), "correct_answered": ka,
            "min_conf": a.min_conf, "median_ms": round(statistics.median(ms)), "date": datetime.date.today().isoformat()}
    json.dump(summ, open(base + "_summary.json", "w"), indent=1)
    print(f"\nwrote {base}_summary.json and {base}_errors.jsonl ({n-k} misses to look at)")


if __name__ == "__main__":
    main()
