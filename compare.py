"""Print every score for every method at one tier, best lr per method, with no combined number.

python3 compare.py 1m [base method, default plain]
Each method's lr is the one with the lowest loss on the tuning rows, which are not the rows reported.
n is how many items a score has. A star means the gap to the base method survives false-win correction (Benjamini-Hochberg at 5%
across every method and score in the table), item by item. "moved" says how many methods' scores changed during training: a score no
method moved cannot judge anything at this tier. Only scores in judges.json, written by audit.py, are shown.
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); TIER = sys.argv[1] if len(sys.argv) > 1 else "1m"; BASE = sys.argv[2] if len(sys.argv) > 2 else "plain"
rows = [json.loads(l) for l in open(os.path.join(HERE, "results.jsonl"))]; best = {}
for r in rows:
    if r["tier"] == TIER and not r["tag"] and (r["method"] not in best or r["scores"]["text.tune.loss"] <= best[r["method"]]["scores"]["text.tune.loss"]):
        best[r["method"]] = r
known = ("plain", "token", "ve", "stem", "mome", "bigram", "engram", "tngram", "moe", "reader", "vgram", "mix", "mix3", "mixbig")
order = [m for m in known if m in best] + sorted(set(best) - set(known))
items = {m: np.load(os.path.join(HERE, "runs", f"{TIER}_{m}_lr{best[m]['lr']:g}.npz")) for m in order}
keys = [k for k in best[order[0]]["scores"] if k != "text.tune.loss" and all(k in best[m]["scores"] for m in order)]
judges = json.load(open(os.path.join(HERE, "judges.json"))) if os.path.exists(os.path.join(HERE, "judges.json")) else keys
live = [k for k in keys if any(best[m]["moved"][k] for m in order) and k in judges]; dead = [k for k in keys if k not in live]
tests = []
for k in live:
    for m in order:
        if m != BASE and BASE in items and len(items[m][k]) == len(items[BASE][k]) > 1:
            d = items[m][k] - items[BASE][k]; se = d.std() / len(d) ** 0.5
            tests.append((math.erfc(abs(d.mean()) / max(se, 1e-12) / 2 ** 0.5), m, k))
tests.sort(key=lambda t: t[0]); cut = max([i + 1 for i, t in enumerate(tests) if t[0] <= 0.05 * (i + 1) / len(tests)], default=0); star = {(m, k) for _, m, k in tests[:cut]}
print(f"tier {TIER}, stars against {BASE}: {cut} of {len(tests)} comparisons. lr " + ", ".join(f"{m} {best[m]['lr']:g}" for m in order))
print("| score | " + " | ".join(order) + " | moved | n |\n|---|" + "---|" * (len(order) + 2))
for k in live:
    print(f"| {k} | " + " | ".join(f"{best[m]['scores'][k]:.3f}" + ("*" if (m, k) in star else "") for m in order) + f" | {sum(best[m]['moved'][k] for m in order)}/{len(order)} | {best[order[0]]['n'][k]} |")
print("| scores left out, no method moved them or audit.py says they can't judge | " + ", ".join(dead) + " |")
print("| weights total | " + " | ".join(str(best[m]["cost"]["total"]) for m in order) + " |  |  |")
print("| weights read per token | " + " | ".join(str(best[m]["cost"]["active"]) for m in order) + " |  |  |")
print("| table rows hit | " + " | ".join(str(best[m]["cost"].get("rows_hit", "-")) for m in order) + " |  |  |")
print("| train seconds | " + " | ".join(str(best[m]["train_s"]) for m in order) + " |  |  |")
