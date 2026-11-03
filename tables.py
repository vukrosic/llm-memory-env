"""Print the six headline scores of every method at one tier, at the lr the tuning rows pick: python3 tables.py TIER [method ...]. RESULTS.md holds this output."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); T = sys.argv[1]
HEAD = ("text.web.loss", "facts.stored.copies32.nll", "facts.new-wording.copies32.nll", "facts.new-wording.name-far.nll", "changed.other-relation.gap", "ambiguous.usual.loss")
best = {}
for r in map(json.loads, open(os.path.join(HERE, "results.jsonl"))):
    if r["tier"] == T and not r["tag"] and (r["method"] not in best or r["scores"]["text.tune.loss"] <= best[r["method"]]["scores"]["text.tune.loss"]): best[r["method"]] = r
print("| method | lr | weights | web loss | stored facts | new wording | name far | relation gap | usual phrase |\n|---|---|---|---|---|---|---|---|---|")
for m in sys.argv[2:] or best:
    r = best[m]; print(f"| {m} | {r['lr']:g} | {r['cost']['total'] / 1e6:.2f}M | " + " | ".join(f"{r['scores'][k]:.3f}" for k in HEAD) + " |")
