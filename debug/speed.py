"""Training steps per second of each method, one at a time on one CPU thread, same batches, two passes, best pass kept. python3 speed.py TIER method ..."""
import sys, os, time, numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); os.chdir(HERE)
import env, methods
torch.set_num_threads(1); T = sys.argv[1]; ms = sys.argv[2:]; c = dict(env.TIERS[T]); c["lr"] = .01; W = env.World(c, c["steps"]); X = torch.as_tensor(W.look[W.rows]); N = 12; best = {}


def go(m):
    torch.manual_seed(0); model = env.LM(c, W.V, methods.build(m, c, W, "cpu")); opts = env.optimizers(model, c["lr"], c["embed_lr"]); model.train(); ts = []
    for step in range(N + 2):
        x = X[(step * c["batch"] + np.arange(c["batch"])) % len(X)]; t = time.perf_counter()
        loss = F.cross_entropy(model(x)[:, :-1].flatten(0, 1), x[:, 1:].flatten(), ignore_index=W.unk) + sum(getattr(b.val, "aux", 0.0) for b in model.blocks); loss.backward()
        for o in opts: o.step(); o.zero_grad(set_to_none=True)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts[2:])), sum(p.numel() for p in model.parameters())


for p in range(2):
    for m in ms:
        s, n = go(m); best[m] = (min(s, best.get(m, (9e9,))[0]), n)
b = best["engram"][0]
for m in ms: print(f"{m:10s} {best[m][1] / 1e6:6.2f}M  {best[m][0]:.3f} s/step  {best[m][0] / b:.2f}x engram", flush=True)
