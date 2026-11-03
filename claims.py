"""Check published rankings of memory modules against this environment.

python3 claims.py
Each claim is one published comparison on validation loss. A claim holds when the gap has the published sign and |z| >= 2; a "tie" claim
holds when |z| < 2. Each method uses the lr with the lowest loss on the tuning rows. z comes from per-row gaps on the web text.
Claims from the same paper can disagree across backbones, so those are listed with both results.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# (a, relation, b, source). "<" means a has lower validation loss than b.
CLAIMS = [
    ("engram", "<", "plain", "Engram; TN-gram Table 1, 1.209 vs 1.251"),
    ("tngram", "<", "plain", "TN-gram Table 1, 1.208 vs 1.251"),
    ("bigram", "<", "plain", "MoME Table 2, .8636 vs .8785"),
    ("ve", "<", "plain", "MoME Table 2, .8633 vs .8785"),
    ("mome", "<", "plain", "MoME Table 2, .8621 vs .8785"),
    ("stem", "<", "plain", "MoME Table 3, .8633 vs .8852"),
    ("mome", "<", "ve", "MoME Table 2, .8621 vs .8633"),
    ("mome", "tie", "ve", "MoME Table 3 at 125M, .8623 vs .8620"),
    ("mome", "<", "bigram", "MoME Table 2, .8621 vs .8636"),
    ("bigram", "<", "mome", "MoME Table 16 at 1.4B, .6943 vs .6990"),
    ("bigram", "tie", "ve", "MoME Table 2, .8636 vs .8633"),
    ("bigram", "<", "engram", "MoME Table 16, .6943 vs .7029"),
    ("ve", "<", "stem", "MoME Table 3, .8620 vs .8633"),
    ("mome", "<", "stem", "MoME Table 3, .8623 vs .8633"),
    ("tngram", "tie", "engram", "TN-gram Table 1, 1.208 vs 1.209 and 1.071 vs 1.070"),
    ("vgram", "<", "bigram", "MoME Table 4, bigram-indexed MoME .8624 vs .8636"),
]
rows = [json.loads(l) for l in open(os.path.join(HERE, "results.jsonl"))]


def best(tier):
    b = {}
    for r in rows:
        if r["tier"] == tier and not r["tag"] and (r["method"] not in b or r["scores"]["text.tune.loss"] <= b[r["method"]]["scores"]["text.tune.loss"]): b[r["method"]] = r
    return b


tiers = [t for t in ("5s", "1m", "1mv", "5m") if best(t)]; B = {t: best(t) for t in tiers}; hits = {t: 0 for t in tiers}; tried = {t: 0 for t in tiers}
print("| claim | source | " + " | ".join(tiers) + " |\n|---|---|" + "---|" * len(tiers))
for a, rel, b, src in CLAIMS:
    cells = []
    for t in tiers:
        if a not in B[t] or b not in B[t]: cells.append("-"); continue
        x, y = [np.load(os.path.join(HERE, "runs", f"{t}_{m}_lr{B[t][m]['lr']:g}.npz"))["text.web.loss"] for m in (a, b)]
        d = x - y; z = d.mean() / max(d.std() / len(d) ** 0.5, 1e-9); ok = abs(z) < 2 if rel == "tie" else z <= -2
        hits[t] += ok; tried[t] += 1; cells.append(f"{'yes' if ok else 'NO'} {d.mean():+.3f} (z {z:+.1f})")
    print(f"| {a} {rel} {b} | {src} | " + " | ".join(cells) + " |")
print("| claims that hold | | " + " | ".join(f"{hits[t]}/{tried[t]}" for t in tiers) + " |")
