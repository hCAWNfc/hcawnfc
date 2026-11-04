#!/usr/bin/env python3
# ==========================================================================
# fc_triggers.py - regex triggers (like Needle's "triggers"): if the query matches a tool's trigger, the model chooses ONLY among
# the matched tools and MUST produce a call (it cannot answer []). Tools without a defined trigger always stay allowed.
# If no trigger matches, nothing changes (the model works on its own and may also decline).
#   python fc_triggers.py check fc_mobile_eval.jsonl     # no model: trigger coverage on your set and on query variants
#   FC_TRIGGERS=0 turns it off (on by default)
# ==========================================================================
import re, os, sys, json

I = re.I
TRIGGERS = {
    "send_email": [r"e-?mail", r"\bmail\b", r"[\w.+-]+@[\w-]+\.\w+", r"\bmessage\b", r"\bsubject\b", r"\bbody\b", r"\bping\b", r"\bnote\b", r"\b(?:tell|inform|notify|message|write to|let)\b.{0,60}\b(?:that|know|saying|about)\b"],
    "create_calendar_event": [r"\bcalendar\b", r"\bschedul\w*", r"\bappointments?\b", r"\bmeetings?\b", r"\bevents?\b", r"\bremind\w*", r"\bbook\b"],
    "create_contact": [r"contact", r"[\w.+-]+@[\w-]+\.\w+", r"\bphone\b", r"\bnumber\b", r"\bsave\b", r"\badd\b.{0,40}\b(?:number|phone)\b"],
    "show_map": [r"\bmaps?\b", r"\bwhere\b", r"\bdirections?\b", r"\bnavigat\w*", r"\blocat(?:e|ion)\b", r"\bshow me\b", r"\bfind\b", r"\baddress\b", r"\b(?:nearest|nearby|closest)\b", r"\broute\b"],
    "open_wifi_settings": [r"\bwi-?fi\b", r"\bwireless\b", r"\binternet settings\b", r"\bnetwork settings\b"],
    "turn_on_flashlight": [r"\bflash ?light\b", r"\btorch\b", r"\blights?\b"],
    "turn_off_flashlight": [r"\bflash ?light\b", r"\btorch\b", r"\blights?\b"],
}
RX = {k: [re.compile(p, I) for p in v] for k, v in TRIGGERS.items()}
ENABLED = os.environ.get("FC_TRIGGERS", "1") == "1"


def matched(query, names):
    """Set of tool names (from `names`) whose trigger matched; [] if none."""
    return [n for n in names if n in RX and any(r.search(query) for r in RX[n])]


def has_trigger_tools(names):
    return any(n in RX for n in names)


def restrict(tools, query):
    """Returns (tools_for_the_grammar, force_call). If nothing matched, returns the original tools and False."""
    if not ENABLED or not query:
        return tools, False
    names = [t["name"] for t in tools]
    m = set(matched(query, names))
    if not m:
        return tools, False
    keep = [t for t in tools if t["name"] in m or t["name"] not in RX]       # tools without a trigger stay
    return keep, True


def cmd_check(path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from .fc_augment import has_quotes, strip_quotes, saying_form, dropthe_form, emailverb_form, droptitled_form
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    forms = {"orig": lambda q: q, "noquote": strip_quotes, "saying": saying_form, "emailverb": emailverb_form, "droptitled": droptitled_form}
    print(f"{'form':11}{'no match':>14}{'single tool':>12}{'has gold':>10}{'GOLD DROPPED':>14}   (n={len(rows)})")
    bad = []
    for fn, f in forms.items():
        none = one = hit = miss = 0
        for ex in rows:
            q = f(str(ex["query"]))
            names = [t["name"] for t in ex["tools"]]
            gold = {c["name"] for c in ex["calls"]}
            m = set(matched(q, names))
            if not m:
                none += 1; continue
            keep = {n for n in names if n in m or n not in RX}
            one += len(keep) == 1
            if gold <= keep: hit += 1
            else:
                miss += 1
                if fn == "orig" and len(bad) < 8: bad.append((q[:110], sorted(gold), sorted(keep)))
        n = len(rows)
        print(f"{fn:11}{none/n:>14.1%}{one/n:>12.1%}{hit/n:>10.1%}{miss/n:>14.1%}")
    print("\nGOLD DROPPED = a trigger matched some tool, but the right tool is not among the matched ones (the rule hurts there).")
    for q, g, k in bad:
        print(f"\n  query: {q}\n  gold: {g} | allowed: {k}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "check":
        cmd_check(sys.argv[2])
    else:
        print(__doc__)
