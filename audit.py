"""Which scores can judge a module. python3 audit.py

Uses only the published modules, so our own designs can't shape the verdict. For each score:
moved: share of runs where training changed it. spread: best minus worst module in standard errors of the item-by-item gap.
lr rho: rank agreement across modules between lr .01 and .02. tier rho: rank agreement between the two tiers, each module at its tuned lr.
hidden rho, if final_results.jsonl has them: rank agreement between public and hidden items at the same lr.
A score judges when moved >= 1/2, spread >= 3, lr rho >= .4 and tier rho >= .3, cut-offs set before reading the table, and hidden rho >= .4,
set after reading it. Tiers 5m and 1m are each checked against the other. A score must pass both to go in judges.json, which compare.py reads.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = ("plain", "token", "ve", "stem", "mome", "bigram", "engram", "tngram")
rows = [json.loads(l) for l in open(os.path.join(HERE, "results.jsonl"))]; rows = [r for r in rows if not r["tag"] and r["method"] in PUB]
final = [json.loads(l) for l in open(os.path.join(HERE, "final_results.jsonl"))] if os.path.exists(os.path.join(HERE, "final_results.jsonl")) else []


def rho(a, b):
    if len(a) < 4: return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() and rb.std() else 0.0


def at(tier, lr): return {r["method"]: r for r in rows if r["tier"] == tier and r["lr"] == lr}


def tuned(tier):
    b = {}
    for r in rows:
        if r["tier"] == tier and (r["method"] not in b or r["scores"]["text.tune.loss"] <= b[r["method"]]["scores"]["text.tune.loss"]): b[r["method"]] = r
    return b


def table(T, T2):
    A, C = at(T, 0.01), at(T, 0.02); B, B2 = tuned(T), tuned(T2); keys = [k for k in next(iter(B.values()))["scores"] if k != "text.tune.loss"]; good = []
    print(f"tier {T}, check tier {T2}, modules {sorted(B)}\n| score | moved | spread | lr rho | tier rho | hidden rho | judges |\n|---|---|---|---|---|---|---|")
    for k in keys:
        ms = [m for m in B if k in B[m]["scores"]]; v = {m: B[m]["scores"][k] for m in ms}; hi, lo = max(v, key=v.get), min(v, key=v.get)
        x, y = [np.load(os.path.join(HERE, "runs", f"{T}_{m}_lr{B[m]['lr']:g}.npz"))[k] for m in (hi, lo)]; d = x - y; spread = abs(d.mean()) / max(d.std() / len(d) ** 0.5, 1e-9) if len(d) > 1 else 0.0
        moved = np.mean([B[m]["moved"][k] for m in ms]); same = [m for m in A if m in C and k in A[m]["scores"] and k in C[m]["scores"]]
        lr = rho([A[m]["scores"][k] for m in same], [C[m]["scores"][k] for m in same]); both = [m for m in ms if m in B2 and k in B2[m]["scores"]]; tr = rho([B[m]["scores"][k] for m in both], [B2[m]["scores"][k] for m in both])
        F = {r["method"]: r["scores"] for r in final if r["tier"] == T and r["method"] in B and r["lr"] == B[r["method"]]["lr"] and k in r["scores"]}; hid = rho([B[m]["scores"][k] for m in F], [F[m][k] for m in F])
        ok = moved >= 0.5 and spread >= 3 and lr >= 0.4 and tr >= 0.3 and not hid < 0.4; good += [k] * bool(ok)
        print(f"| {k} | {moved:.2f} | {spread:.1f} | {lr:.2f} | {tr:.2f} | {hid:.2f} | {'yes' if ok else ''} |")
    print(f"| judges: {len(good)} of {len(keys)} |\n"); return good


a, b = table("5m", "1m"), table("1m", "5m"); both = [k for k in a if k in b]
with open(os.path.join(HERE, "judges.json"), "w") as f: json.dump(both, f, indent=0)
print(f"pass both: {len(both)}\n" + "\n".join(both))
