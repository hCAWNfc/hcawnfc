# eval_check

Independent re-check of the published hCAWN-FC7 numbers. It uses only the public `hcawn-fc` pip package and its own short scorer
(strict exact match: same tools in the same order and every argument equal), so anyone gets our numbers from what we published.

```bash
pip install hcawn-fc
hcawn-fc download int8 && hcawn-fc download int4
python3 verify_fc7.py data/fc_mobile_eval.jsonl --model int8 --procs 4 --allow-long
python3 verify_fc7.py data/fc_mobile_eval.jsonl --model int4 --procs 4 --allow-long
python3 verify_fc7.py data/fc_mobile_eval_para.jsonl --model int8 --procs 4 --allow-long --name rewritten
```

Expected (Mobile Actions, 961 requests): int8 90.4%, int4 90.3%. Rewritten requests (905): int8 82.8%. Noise is about 2 points.
`--allow-long` also runs the requests longer than the package limit of 300 characters (46 of the 961; the script also prints the result for the 915 up to 300).
Outputs go to `out/`: `*_summary.json` and `*_errors.jsonl` (every miss, for error analysis). Needs Linux x86-64 for the local engine.

Data in `data/`: requests and expected calls derived from the Mobile Actions dataset (Google, CC BY 4.0) and 905 rewrites of them by GPT-4o-mini.
