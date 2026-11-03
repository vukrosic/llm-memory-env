"""Does a design beat every published baseline? python3 beat.py TIER [method ...]

Win rule, written before any new design ran. At the lr the tuning rows pick:
1. Web loss is lower than every baseline's, by at least 2 standard errors of the row-by-row gap.
2. On the 5 other headline scores, no baseline is better by 2 standard errors or more.
3. At 1m and 5m, total weights are no more than Engram's. 5s has one memory layer, so there the limit is 1.25x Engram's.
A win at 5s only earns a run at 1m; a claim needs 1m and 5m.
Each cell is the design's value and z against the best baseline on that score; a positive z means the design is better.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); HID = "--hidden" in sys.argv; sys.argv = [x for x in sys.argv if x != "--hidden"]; T = sys.argv[1]  # --hidden scores the hidden items, at the lr the tuning rows chose
BASE = ("plain", "token", "engram", "moe", "ve", "mome", "stem", "bigram", "tngram")
HEAD = ("text.web.loss", "facts.stored.copies32.nll", "facts.new-wording.copies32.nll", "facts.new-wording.name-far.nll", "changed.other-relation.gap", "ambiguous.usual.loss")
LOW = (".loss", ".nll", ".delta"); sign = lambda k: -1 if k.endswith(LOW) else 1
judges = json.load(open(os.path.join(HERE, "judges.json")))
best = {}
for r in map(json.loads, open(os.path.join(HERE, "results.jsonl"))):
    if r["tier"] == T and not r["tag"] and (r["method"] not in best or r["scores"]["text.tune.loss"] <= best[r["method"]]["scores"]["text.tune.loss"]): best[r["method"]] = r
if HID:
    best = {m: r for m, r in best.items() if os.path.exists(os.path.join(HERE, "runs", f"final_{T}_{m}_lr{r['lr']:g}_hid.npz"))}
items = {m: np.load(os.path.join(HERE, "runs", f"final_{T}_{m}_lr{best[m]['lr']:g}_hid.npz" if HID else f"{T}_{m}_lr{best[m]['lr']:g}.npz")) for m in best}
if HID:
    for m in best: best[m] = dict(best[m], scores={k: float(items[m][k].mean()) for k in items[m].files})
base = [m for m in BASE if m in best]; ours = sys.argv[2:] or [m for m in best if m not in BASE]; cap = best["engram"]["cost"]["total"] * (1.25 if T == "5s" else 1)


def z(m, b, k):
    d = (items[m][k] - items[b][k]) * sign(k); return d.mean() / max(d.std() / len(d) ** 0.5, 1e-9)


top = lambda k: max(base, key=lambda b: sign(k) * best[b]["scores"][k])
print(f"tier {T}, baselines {', '.join(f'{m} {best[m]['lr']:g}' for m in base)}, weight limit {cap / 1e6:.2f}M")
print("| score | best baseline | " + " | ".join(HEAD) + " |\n|---|---|" + "---|" * len(HEAD))
print("| baseline best | | " + " | ".join(f"{best[top(k)]['scores'][k]:.3f} {top(k)}" for k in HEAD) + " |")
for m in ours:
    print(f"| {m} {best[m]['lr']:g} | {best[m]['cost']['total'] / 1e6:.2f}M | " + " | ".join(f"{best[m]['scores'][k]:.3f} ({z(m, top(k), k):+.1f})" for k in HEAD) + " |")
print()
for m in ours:
    web = min(z(m, b, HEAD[0]) for b in base if b != m); lost = [(k, b) for k in HEAD[1:] for b in base if b != m and z(m, b, k) <= -2]
    w = sum(z(m, top(k), k) >= 2 for k in judges if k in items[m]); l = sum(z(m, top(k), k) <= -2 for k in judges if k in items[m])
    ok = web >= 2 and not lost and best[m]["cost"]["total"] <= cap
    print(f"{m}: {'WIN' if ok else 'no'}. Worst web z against a baseline {web:+.1f}. Headline losses: {', '.join(f'{k.split('.')[-2] if k.count('.') > 1 else k} to {b}' for k, b in lost) or 'none'}. All judging scores against the best baseline on each: {w} better, {l} worse. Weights {'over' if best[m]['cost']['total'] > cap else 'within'} the limit.")
