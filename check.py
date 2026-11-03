"""Bug checks for the environment. python3 check.py [tier, default 5s]

Every module: can it see a future token, does every weight get a gradient, does changing the table change the output.
The data: do test rows appear in training, do stored and new-wording questions differ, is the answer already in the question, what are scores before training.
"""
import sys
import numpy as np, torch, torch.nn.functional as F
import env, evals, methods

T = sys.argv[1] if len(sys.argv) > 1 else "5s"; c = dict(env.TIERS[T]); W = env.World(c, c["steps"]); X = torch.as_tensor(W.look[W.rows[:8]])
print("| method | weights | future leak | past changes | weights without gradient | table matters |\n|---|---|---|---|---|---|")
for name in methods.METHODS:
    torch.manual_seed(0)
    try: parts = methods.build(name, c, W, "cpu")
    except (TypeError, ValueError): print(f"| {name} | no table budget left at this tier | | | | |"); continue  # MoE's extra weights come out of the table, and at small tiers nothing is left
    m = env.LM(c, W.V, parts); m.eval(); x2 = X.clone(); x2[:, 30] = (x2[:, 30] + 7) % W.V
    with torch.no_grad():
        torch.manual_seed(1); a = m(X); torch.manual_seed(1); b = m(x2)
    leak, past = (a - b)[:, :30].abs().max().item(), (a - b)[:, 30:].abs().max().item()
    m.train(); F.cross_entropy(m(X)[:, :-1].flatten(0, 1), X[:, 1:].flatten()).backward()
    dead = [n for n, w in m.named_parameters() if w.grad is None or (w.grad.to_dense() if w.grad.is_sparse else w.grad).abs().max() == 0]
    m.eval(); tabs = [t for n, t in m.named_modules() if ".tab" in n and isinstance(t, torch.nn.Embedding)]; matter = "-"
    if tabs:
        with torch.no_grad():
            torch.manual_seed(1); a = m(X)
            for t in tabs: t.weight.mul_(0)
            torch.manual_seed(1); matter = f"{(a - m(X)).abs().max().item():.2f}"
    print(f"| {name} | {sum(w.numel() for w in m.parameters()) / 1e6:.3f}M | {leak:.1e} | {past:.2f} | {', '.join(dead[:4]) or 'none'}{' +' + str(len(dead) - 4) if len(dead) > 4 else ''} | {matter} |")

train = {r.tobytes() for r in W.rows}; print("\ntest rows found in training:", {n: sum(r.tobytes() in train for r in getattr(W, n)) for n in ("val", "tune", "hid")}, "of", len(W.val))
piece = {r[:12].tobytes() for r in W.rows}; print("test rows whose first 12 tokens start a training row:", sum(r[:12].tobytes() in piece for r in W.val))
ctx = evals.Ctx(None, W, c, "cpu"); stored, new, fake, page, natural = evals.fact_sets(ctx)
same = sum(a["q"] == b["q"] for a, b in zip(stored, new)); seen = sum(a["y"] in W.look[np.array(a["q"])].tolist() for a in new)
print(f"facts asked {len(stored)}, same question in both wordings {same}, new-wording answer already in the question {seen}, natural {len(natural)}")
dec = lambda t: W.enc.decode(t)
for a, b, n in list(zip(stored, new, [0] * 3))[:3]: print(f"  copies {a['copies']}, candidates {len(a['cand'])}: [{dec(a['q'])}] [{dec(b['q'])}] -> [{dec(a['a'])}]")
print("candidates per fact, median", int(np.median([len(i["cand"]) for i in stored])), "chance of pick", round(float(np.mean([1 / len(i["cand"]) for i in stored])), 3))
for s in ("copies", "maj", "near", "early"): print(s, {v: sum(i[s] == v for i in stored) for v in sorted({i[s] for i in stored})})
