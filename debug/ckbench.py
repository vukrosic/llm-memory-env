import sys, numpy as np, torch
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env, evals, methods
c = dict(env.TIERS["1h"]); W = env.World(c, c["steps"]); parts = methods.build("plain", c, W, "cpu"); m = env.LM(c, W.V, parts)
z = torch.load(env.RUNS + "/ckpt_1h_plain_lr0.01.pt", map_location="cpu", weights_only=False); m.load_state_dict(z["model"]); m.eval(); S = z["S"]; f = S["first"]
ctx = evals.Ctx(m, W, c, "cpu"); out = evals.benchmarks(ctx); print("step", S["step"], "bench items", len(out["bench.arc.acc"]))
for k, v in out.items():
    d = v - f[k]; print(k.ljust(26), f"start {f[k].mean():.3f} now {v.mean():.3f} moved {abs(d.mean()) > 2 * d.std() / len(d) ** .5} se {v.std() / len(v) ** .5:.3f}")
