"""Switch off each memory of a trained model at test time and read the relation gap, its two sides and the stored-fact score."""
import sys, os, numpy as np, torch
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); os.chdir(HERE)
import env, methods, evals
T, m, lr, tag = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else ""
c = dict(env.TIERS[T]); c["lr"] = lr; W = env.World(c, c["steps"]); parts = methods.build(m, c, W, "cpu"); model = env.LM(c, W.V, parts)
model.load_state_dict(torch.load(f"runs/end_{T}_{m}_lr{lr:g}{tag}.pt", map_location="cpu")); model.eval()
ctx = evals.Ctx(model, W, c, "cpu"); stored, *_ = evals.fact_sets(ctx)
rng = np.random.default_rng(4); pre = lambda p, x: [env.EOT] + W.E(p[:p.find("[Y]")].replace("[X]", x))
pool = [i for i in stored if i["copies"] >= 8 and not i["maj"] and not i["inq"]]; items = []
for i in pool:
    other = [j for j in pool if j["f"]["r"] == i["f"]["r"] and j["y"] != i["y"]]; rel = [j for j in pool if j["f"]["r"] != i["f"]["r"]]
    if not other or not rel: continue
    o, r = other[rng.integers(len(other))], rel[rng.integers(len(rel))]
    wording = r["f"]["S"][rng.integers(len(r["f"]["S"]))]; own = i["f"]["S"][rng.integers(len(i["f"]["S"]))]
    items.append((i["q"], pre(wording, i["f"]["s"]), i["y"]))


def score(label):
    lp = lambda k: np.array([r[1][it[2]] for r, it in zip(ctx.probe([(it[k], [env.EOT]) for it in items]), items)])
    own, wrong = lp(0), lp(1); g = own - wrong
    print(f"{label:28s} own {own.mean():7.3f}  wrong relation {wrong.mean():7.3f}  gap {g.mean():6.3f} (se {g.std() / len(g) ** .5:.2f})", flush=True)


score("all on")
class Off(torch.nn.Module):
    def forward(s, ids, e): return e


vals = [(k, b.val) for k, b in enumerate(model.blocks) if b.val is not None]; mems = dict(model.mems)
for k, mem in mems.items():
    model.mems[k] = Off(); ctx = evals.Ctx(model, W, c, "cpu"); score(f"{type(mem).__name__} after {k} off"); model.mems[k] = mem
for k, v in vals:
    pair, fwd = getattr(v, "pair", None), v.forward
    if pair is not None: v.pair = lambda ids, h, q, k_: 0.0
    else: v.forward = lambda ids, h, x: x
    ctx = evals.Ctx(model, W, c, "cpu"); score(f"{type(v).__name__} in block {k} off")
    if pair is not None: del v.pair
    else: del v.forward
for k in mems: model.mems[k] = Off()
for k, v in vals:
    if hasattr(v, "pair"): v.pair = lambda ids, h, q, k_: 0.0
ctx = evals.Ctx(model, W, c, "cpu"); score("all off")
