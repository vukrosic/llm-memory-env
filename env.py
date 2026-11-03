"""Memory test environment: train any memory method at any budget on fresh text and print the same scores.

One knob, --tier, sets the model size, vocabulary and steps. Everything else (data source, methods, evals) is the same at
every tier. Steps are fixed per tier so every method reads the same tokens; "5s" means plain takes about 5 s on one Mac core.
The training text is fresh web text, each token read once, plus FACTSHARE of ParaRel fact sentences. The world depends only
on the tier, so every method is scored on the same items and two methods can be compared item by item.

python3 env.py --method engram --tier 5s [--lr 0.02] [--steps 50]
Results go to results.jsonl, one per-item score file per run to runs/. compare.py prints the tables.
"""
import argparse, collections, glob, json, os, re, time
import numpy as np, tiktoken, torch, torch.nn as nn, torch.nn.functional as F
import evals, methods

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data"); RUNS = os.path.join(HERE, "runs")
FW = os.environ.get("FWDIR", os.path.join(DATA, "web")); PR = os.path.join(DATA, "pararel")
EOT, FACTSHARE, COPIES = 50256, 0.1, (2, 8, 32)
TIERS = {
    "5s": dict(width=48, layers=2, heads=4, V=4000, batch=16, L=48, steps=250, slots=16384, lr=0.02, embed_lr=0.01, val_rows=200, evals=3, bench=60),
    "1m": dict(width=64, layers=4, heads=4, V=8000, batch=32, L=48, steps=600, slots=65536, lr=0.02, embed_lr=0.01, val_rows=300, evals=2, bench=150),
    "1mv": dict(width=64, layers=4, heads=4, V=1000, batch=32, L=48, steps=600, slots=8192, lr=0.02, embed_lr=0.01, val_rows=300, evals=2, bench=150),  # 1m with 8x the tokens per table row
    "5m": dict(width=96, layers=4, heads=4, V=12000, batch=32, L=64, steps=1500, slots=131072, lr=0.02, embed_lr=0.01, val_rows=500, evals=3, bench=250),
    "1h": dict(width=256, layers=6, heads=8, V=20000, batch=32, L=128, steps=10000, slots=524288, lr=0.02, embed_lr=0.01, val_rows=1000, evals=4, bench=500),
    "5h": dict(width=512, layers=8, heads=8, V=50257, batch=64, L=256, steps=60000, slots=4194304, lr=0.02, embed_lr=0.01, val_rows=4000, evals=10, bench=600),
}
STOP = set("""a an the of in on at to for from by with as is was are were be been being has had have do does did
and or but not no it its this that these those which who whom whose what where when there their his her he she they
him them one also into onto about over under than then so such can could may might will would shall should""".split())


def test_pattern(p):
    x, y = p.find("[X]"), p.find("[Y]")
    if x < 0 or y < 0 or x > y: return False
    return any(w not in STOP for w in p[:y].replace("[X]", " ").lower().replace(".", " ").replace(",", " ").split())


def fact_table():
    """ParaRel relations, cleaned by the ParaRel* rule: per relation one wording group is held out for testing. Cached."""
    path = os.path.join(DATA, "facts.json")
    if os.path.exists(path): return json.load(open(path))
    rng = np.random.default_rng(0); out = []
    for f in sorted(glob.glob(os.path.join(PR, "pattern_data", "graphs_json", "P*.jsonl"))):
        r = os.path.basename(f)[:-6]; ff = os.path.join(PR, "trex_lms_vocab", r + ".jsonl")
        if not os.path.exists(ff): continue
        xy = [p for p in map(json.loads, filter(str.strip, open(f))) if 0 <= p["pattern"].find("[X]") < p["pattern"].find("[Y]")]
        groups = sorted({p["extended_lemma"] for p in xy if test_pattern(p["pattern"])})
        if len(groups) < 2: continue
        G = groups[rng.integers(len(groups))]
        S = [p["pattern"] for p in xy if p["extended_lemma"] != G]; T = [p["pattern"] for p in xy if p["extended_lemma"] == G and test_pattern(p["pattern"])]
        if not S or not T: continue
        pairs = list(dict.fromkeys((d["sub_label"], d["obj_label"]) for d in map(json.loads, filter(str.strip, open(ff)))))
        n = collections.Counter(s for s, _ in pairs)
        out.append(dict(r=r, S=S, T=T, pairs=[(s, o) for s, o in pairs if n[s] == 1]))
    os.makedirs(DATA, exist_ok=True); json.dump(out, open(path, "w")); return out


class World:
    """Everything that depends only on the tier: vocabulary, training rows, held-out rows, fact items."""
    def __init__(s, c, method_free_steps):
        s.c = c; s.enc = tiktoken.get_encoding("gpt2"); rng = np.random.default_rng(0)
        shard = np.memmap(os.path.join(FW, "fineweb_train_000001.bin"), dtype=np.uint16, mode="r", offset=1024)
        cp = os.path.join(DATA, "counts.npy")
        if not os.path.exists(cp):
            os.makedirs(DATA, exist_ok=True); np.save(cp, np.bincount(np.asarray(shard[:20_000_000]), minlength=65536))
        counts = np.load(cp); s.counts = counts
        s.keep = np.unique(np.concatenate([np.argsort(counts)[-(c["V"] - 2):], [EOT]])); s.V = len(s.keep) + 1; s.unk = s.V - 1
        s.look = np.full(65536, s.unk, np.int64); s.look[s.keep] = np.arange(len(s.keep)); kept = set(s.keep.tolist())
        canon = {}; s.comp = np.zeros(s.V, np.int64)  # Engram's tokenizer compression: lowercase, no spaces
        for i, t in enumerate(s.keep.tolist() + [-1]):
            s.comp[i] = canon.setdefault(s.enc.decode([t]).strip().lower() if t >= 0 else "<unk>", len(canon))
        s.ncanon = len(canon)
        s.E = lambda t: s.enc.encode(" " + t.strip())
        tokens = method_free_steps * c["batch"] * c["L"]
        s.facts = s.pick_facts(rng, kept, int(tokens * FACTSHARE))
        doc = lambda f: np.array([EOT] + s.E(f["S"][rng.integers(len(f["S"]))].replace("[X]", f["s"]).replace("[Y]", f["o"])), np.int64)
        early = [doc(f) for f in s.facts if f["early"] for _ in range(f["copies"])]; docs = [doc(f) for f in s.facts if not f["early"] for _ in range(f["copies"])]
        nfact = sum(map(len, docs + early)); nweb = tokens + 2 * c["L"] - nfact
        if nweb > len(shard): raise SystemExit(f"tier needs {nweb} web tokens, the shard has {len(shard)}: add more shards")
        web = np.asarray(shard[:nweb]).astype(np.int64); cut = np.flatnonzero(web == EOT)
        wdocs = np.split(web, cut[1:]) if len(cut) > 1 else [web]
        alld = docs + wdocs; go = rng.random(len(alld)) < (tokens / 3 - sum(map(len, early))) / max(1, tokens - sum(map(len, early)))

        def rows_of(part):
            st = np.concatenate([part[i] for i in rng.permutation(len(part))]); r = st[:len(st) // c["L"] * c["L"]].reshape(-1, c["L"]); return r[rng.permutation(len(r))]
        part1 = rows_of(early + [d for d, g in zip(alld, go) if g]); s.rows = np.concatenate([part1, rows_of([d for d, g in zip(alld, go) if not g])]); s.split = len(part1)
        val = np.memmap(os.path.join(FW, "fineweb_val_000000.bin"), dtype=np.uint16, mode="r", offset=1024)  # rows spread across the file: reported, tuning and hidden rows alternating
        held = np.stack([np.asarray(val[a:a + c["L"]]) for a in np.linspace(0, len(val) - c["L"] - 1, 3 * c["val_rows"]).astype(np.int64)]).astype(np.int64)
        s.val, s.tune, s.hid = held[::3], held[1::3], held[2::3]
        s.nfact, s.nweb = nfact, nweb
        print(f"world: V={s.V}, {len(s.rows)} rows, {nweb} web tokens, {len(s.facts)} facts ({nfact} tokens)", flush=True)

    def pick_facts(s, rng, kept, budget):
        """As many facts as FACTSHARE of the tokens can hold; one third each written 2, 8 and 32 times (how famous)."""
        pool = []
        for g in fact_table():
            ok = [(a, b) for a, b in g["pairs"] if all(t in kept for t in s.E(a) + s.E(b))]
            if not ok: continue
            maj = collections.Counter(b for _, b in ok).most_common(1)[0][0]; objs = sorted({b for _, b in ok})
            pool += [dict(r=g["r"], s=a, o=b, S=g["S"], T=g["T"], maj=maj, objs=objs) for a, b in ok]
        pool = [pool[i] for i in rng.permutation(len(pool))]; avg = sum(COPIES) / 3 * 13  # tokens a fact costs on average
        n = min(max(40, int(budget / avg)), max(len(pool) - 400, len(pool) * 3 // 4)); s.natural = [f for f in pool[n:n + 400] if f["o"] != f["maj"]]; pool = pool[:n]  # never planted: only web text can teach these
        for i, f in enumerate(pool):  # every 4th fact appears only in the first third of training, to measure forgetting
            f["copies"] = 8 if i % 4 == 3 else COPIES[i % 3]; f["early"] = i % 4 == 3
        return pool


def rope(x, base=10000.0):
    t, d = x.shape[-2], x.shape[-1]
    f = 1.0 / base ** (torch.arange(0, d, 2, device=x.device).float() / d); ang = torch.outer(torch.arange(t, device=x.device).float(), f); c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x2 * c + x1 * s], -1).flatten(-2)


class Block(nn.Module):
    def __init__(s, d, h, moe=0, topk=2):
        super().__init__()
        s.h, s.dk, s.moe, s.topk = h, d // h, moe, topk
        s.q, s.k, s.v, s.o = (nn.Linear(d, d, bias=False) for _ in range(4)); hid = int(8 * d / 3)
        if moe:  # 1 shared expert of half the dense width + top-k of E routed experts, active width = dense width
            hs, he = hid // 2, hid // (2 * topk)
            s.gate, s.up, s.down = nn.Linear(d, hs, bias=False), nn.Linear(d, hs, bias=False), nn.Linear(hs, d, bias=False)
            s.eg = nn.ModuleList(nn.Linear(d, he, bias=False) for _ in range(moe)); s.eu = nn.ModuleList(nn.Linear(d, he, bias=False) for _ in range(moe))
            s.ed = nn.ModuleList(nn.Linear(he, d, bias=False) for _ in range(moe)); s.router = nn.Linear(d, moe, bias=False)
            s.register_buffer("rbias", torch.zeros(moe)); s.load = None
        else:
            s.gate, s.up, s.down = nn.Linear(d, hid, bias=False), nn.Linear(d, hid, bias=False), nn.Linear(hid, d, bias=False)
        s.n1, s.n2, s.qn, s.kn = nn.RMSNorm(d), nn.RMSNorm(d), nn.RMSNorm(s.dk), nn.RMSNorm(s.dk); s.val = s.stem = None

    def forward(s, x, ids):
        b, t, d = x.shape
        h = s.n1(x); q, k, v = (m(h).view(b, t, s.h, s.dk).transpose(1, 2) for m in (s.q, s.k, s.v))
        if s.val is not None: v = s.val(ids, h, v)
        if hasattr(s.val, "query"): q = s.val.query(ids, q)
        rq, rk = rope(s.qn(q)), rope(s.kn(k)); y = F.scaled_dot_product_attention(rq, rk, v, is_causal=True)
        if hasattr(s.val, "pair"): y = y + s.val.pair(ids, h, rq, rk)
        if hasattr(s.val, "after"): y = s.val.after(y, h)
        x = x + s.o(y.transpose(1, 2).reshape(b, t, d)); h = s.n2(x)
        out = s.down(F.silu(s.gate(h)) * (s.stem.tab(ids) if s.stem is not None else s.up(h)))
        if s.moe:
            hf = h.reshape(-1, d); sc = torch.sigmoid(s.router(hf)); idx = (sc + s.rbias).topk(s.topk, -1).indices
            wt = sc.gather(1, idx); wt = wt / wt.sum(-1, keepdim=True); r = torch.zeros_like(hf)
            for e in range(s.moe):
                m = idx == e; rows = m.any(-1).nonzero().squeeze(1)
                if len(rows):
                    xe = hf[rows]; r.index_add_(0, rows, s.ed[e](F.silu(s.eg[e](xe)) * s.eu[e](xe)) * (wt * m)[rows].sum(-1, keepdim=True))
            s.load = torch.bincount(idx.flatten(), minlength=s.moe).float(); out = out + r.view(b, t, d)
        return x + out


class LM(nn.Module):
    def __init__(s, c, V, parts):
        super().__init__()
        d = c["width"]; s.emb = nn.Embedding(V, d); nn.init.normal_(s.emb.weight, std=d ** -0.5)
        s.blocks = nn.ModuleList(Block(d, c["heads"], parts.get("moe", 0)) for _ in range(c["layers"])); s.norm = nn.RMSNorm(d)
        s.neg = parts.get("neg", 0)
        s.mems = nn.ModuleDict({str(k): m for k, m in parts.get("mems", {}).items()})
        for k, m in parts.get("value", {}).items(): s.blocks[k].val = m
        for k, m in parts.get("stem", {}).items(): s.blocks[k].stem = m; s.blocks[k].up = None
        for m in s.modules():
            if hasattr(m, "bind"): m.bind(s.emb)

    def forward(s, x):
        e = s.emb(x)
        for k, b in enumerate(s.blocks):
            if str(k) in s.mems: e = s.mems[str(k)](x, e)
            e = b(e, x)
        if str(len(s.blocks)) in s.mems: e = s.mems[str(len(s.blocks))](x, e)  # a memory after the last block
        e = s.norm(e)
        if "out" in s.mems: e = s.mems["out"](x, e)  # a memory after the final norm adds straight to the word scores
        return e @ s.emb.weight.T


def negative(model, x, rate, every=8, tail=3):
    """Every 8 tokens, swap the last 3 tokens for another row's and ask the model for the original next token. The penalty counts only a rare next token that the model rates above its base rate, and that isn't already on the page. So a name read under the wrong phrase is trained not to give its answer, and common words and copying are left alone."""
    b, t = x.shape; c = x.clone(); o = x[torch.randperm(b, device=x.device)]; pos = torch.arange(every, t - 1, every, device=x.device)
    for p in pos.tolist(): c[:, p - tail + 1:p + 1] = o[:, p - tail + 1:p + 1]
    y = x[:, pos + 1]; lp = model(c)[:, pos].log_softmax(-1).gather(-1, y[..., None])[..., 0]; base = rate[y]
    seen = torch.stack([(x[:, :p + 1] == x[:, p + 1:p + 2]).any(1) for p in pos.tolist()], 1); m = (base < 1e-4) & ~seen
    return (F.relu(lp - base.log()) * m).sum() / m.sum().clamp_min(1)


def newton_schulz(G, steps=5):
    X = G / (G.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for _ in range(steps):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


class Muon(torch.optim.Optimizer):
    def __init__(s, params, lr, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(s):
        for g in s.param_groups:
            for w in g["params"]:
                if w.grad is None: continue  # an expert that got no tokens this step
                buf = s.state[w].setdefault("buf", torch.zeros_like(w)); buf.mul_(g["momentum"]).add_(w.grad)
                w.add_(newton_schulz(w.grad.add(buf, alpha=g["momentum"])) * max(1.0, w.shape[0] / w.shape[1]) ** 0.5, alpha=-g["lr"])


class LazyAdam(torch.optim.Optimizer):
    """SparseAdam for devices without sparse tensors: dense gradients, only rows hit this step move."""
    def __init__(s, params, lr, betas):
        super().__init__(params, dict(lr=lr, betas=betas))

    @torch.no_grad()
    def step(s):
        for g in s.param_groups:
            b1, b2 = g["betas"]
            for w in g["params"]:
                if w.grad is None: continue
                st = s.state[w]; hit = (w.grad != 0).any(1)
                if not st: st["t"], st["m"], st["v"] = 0, torch.zeros_like(w), torch.zeros_like(w)
                st["t"] += 1; d = w.grad[hit]
                st["m"][hit] = st["m"][hit] * b1 + d * (1 - b1); st["v"][hit] = st["v"][hit] * b2 + d * d * (1 - b2)
                w[hit] -= g["lr"] * (st["m"][hit] / (1 - b1 ** st["t"])) / ((st["v"][hit] / (1 - b2 ** st["t"])).sqrt() + 1e-8)


def optimizers(model, lr, embed_lr):
    """Muon on the attention and MLP matrices and the memory's projections; Adam on the embedding; tables move only where hit."""
    hid = [w for n, w in model.named_parameters() if w.ndim == 2 and "router" not in n and ".tab" not in n and (n.startswith("blocks") or re.search(r"\.w\w*\.weight$", n))]
    tabs = [w for n, w in model.named_parameters() if ".tab" in n]; used = {id(w) for w in hid + tabs}
    opts = [Muon(hid, lr), torch.optim.Adam([w for w in model.parameters() if id(w) not in used], lr=embed_lr, betas=(0.9, 0.95))]
    if tabs: opts.append((LazyAdam if tabs[0].device.type == "mps" else torch.optim.SparseAdam)(tabs, lr=embed_lr * 5, betas=(0.9, 0.95)))
    for o in opts:
        for g in o.param_groups: g["base"] = g["lr"]
    return opts


def run(a):
    torch.manual_seed(0); torch.set_num_threads(a.threads); dev = a.dev or ("cuda" if torch.cuda.is_available() else "cpu"); t0 = time.time()
    c = dict(TIERS[a.tier]); c["steps"] = a.steps or c["steps"]; c["lr"] = a.lr or c["lr"]; c["slots"] = int(a.method.split("@")[1]) if "@" in a.method else c["slots"]
    W = World(c, c["steps"]); parts = methods.build(a.method, c, W, dev)
    model = LM(c, W.V, parts).to(dev); opts = optimizers(model, c["lr"], c["embed_lr"]); X = torch.as_tensor(W.look[W.rows])
    mac = dev == "mps"; tparts = methods.build(a.method, c, W, "cpu") if mac else parts; twin = LM(c, W.V, tparts) if mac else model  # batches of changing shape are very slow on the Mac GPU, so its evals run on the CPU
    ctx = evals.Ctx(twin, W, c, "cpu" if mac else dev); prep = time.time() - t0; name = f"{a.tier}_{a.method}_lr{c['lr']:g}{a.tag}"; ck = os.path.join(RUNS, f"ckpt_{name}.pt"); os.makedirs(RUNS, exist_ok=True)
    forget = int(W.split / c["batch"]); marks = sorted({0, c["steps"], forget} | {int(c["steps"] * (i + 1) / c["evals"]) for i in range(c["evals"])})
    S = dict(step=0, first=None, mid=None, curve=[], eval_s=0.0, train_s=0.0)
    if os.path.exists(ck):  # a long tier continues across calls
        z = torch.load(ck, map_location=dev, weights_only=False); model.load_state_dict(z["model"]); S = z["S"]
        for o, sd in zip(opts, z["opts"]): o.load_state_dict(sd)
        print(f"resumed at step {S['step']}", flush=True)
    last = S["first"]; t1 = time.time(); rate = np.zeros(W.V); rate[:len(W.keep)] = W.counts[W.keep]; rate = torch.as_tensor(rate / rate.sum(), dtype=torch.float32, device=dev).clamp_min(1e-9)
    for step in range(S["step"], c["steps"] + 1):
        if a.max_seconds and time.time() - t0 > a.max_seconds:
            S["step"] = step; S["train_s"] += time.time() - t1; torch.save(dict(model=model.state_dict(), opts=[o.state_dict() for o in opts], S=S), ck)
            print(f"paused at step {step} of {c['steps']}, run the same command to continue", flush=True); return
        if step in marks:
            te = time.time(); twin.load_state_dict(model.state_dict()); twin.eval(); last = evals.run_all(ctx, heavy=step in (0, c["steps"])); model.train(); d = time.time() - te; S["eval_s"] += d; t1 += d
            S["first"] = S["first"] if S["first"] is not None else last
            if step == forget: S["mid"] = last
            S["curve"].append(dict(step=step, **{k: round(float(v.mean()), 4) for k, v in last.items()}))
            print(f"step {step}: web loss {S['curve'][-1]['text.web.loss']} ({time.time() - t0:.0f}s)", flush=True)
        if step == c["steps"]: break
        sc = min((step + 1) / 20, 1.0, (c["steps"] - step) / (0.3 * c["steps"]))
        idx = (step * c["batch"] + np.arange(c["batch"])) % len(X); x = X[idx].to(dev)
        neg = model.neg * negative(model, x, rate) if model.neg else 0.0
        loss = F.cross_entropy(model(x)[:, :-1].flatten(0, 1), x[:, 1:].flatten(), ignore_index=W.unk) + neg + sum(getattr(b.val, "aux", 0.0) for b in model.blocks); loss.backward()
        for o in opts:
            for g in o.param_groups: g["lr"] = g["base"] * sc
            o.step()
        model.zero_grad(set_to_none=True)
        for b in model.blocks:  # aux-loss-free balancing: raise the bias of under-used experts
            if b.moe: b.rbias += 1e-3 * torch.sign(b.load.mean() - b.load)
    train_s = S["train_s"] + time.time() - t1; first = S["first"]
    moved = {k: bool(abs((last[k] - first[k]).mean()) > 2 * (last[k] - first[k]).std() / max(1, len(last[k])) ** 0.5) for k in last}
    if S["mid"] is not None:  # loss on facts that stopped appearing after the first third, since they stopped; positive means forgotten
        last["forget.after-third.delta"] = last["forget.early.nll"] - S["mid"]["forget.early.nll"]; d = last["forget.after-third.delta"]; moved["forget.after-third.delta"] = bool(abs(d.mean()) > 2 * d.std() / max(1, len(d)) ** 0.5)
    if a.final:  # the hidden rows, facts and benchmark halves, written apart from the normal results
        hid = evals.run_all(evals.Ctx(twin, W, c, ctx.dev, final=True))
        np.savez_compressed(os.path.join(RUNS, "final_" + name + ".npz"), **hid)
        with open(os.path.join(HERE, "final_results.jsonl"), "a") as f: f.write(json.dumps(dict(method=a.method, tier=a.tier, lr=c["lr"], scores={k: round(float(v.mean()), 4) for k, v in hid.items()})) + "\n")
    res = dict(method=a.method, tier=a.tier, lr=c["lr"], steps=c["steps"], train_s=round(train_s, 1), eval_s=round(S["eval_s"], 1), prep_s=round(prep, 1),
               tokens_per_s=int(c["steps"] * c["batch"] * c["L"] / max(train_s, 1e-6)), cost=evals.cost(twin, tparts, W, X), n={k: len(v) for k, v in last.items()},
               scores={k: round(float(v.mean()), 4) for k, v in last.items()}, moved=moved, curve=S["curve"], tag=a.tag, facts=len(W.facts), V=W.V)
    np.savez_compressed(os.path.join(RUNS, name + ".npz"), **last)
    with open(os.path.join(HERE, "results.jsonl"), "a") as f: f.write(json.dumps(res) + "\n")
    if os.path.exists(ck): os.remove(ck)
    if a.keep: torch.save(model.state_dict(), os.path.join(RUNS, f"end_{name}.pt"))
    print(f"{a.method} {a.tier} lr={c['lr']:g}: train {train_s:.1f}s, evals {S['eval_s']:.1f}s, prep {prep:.1f}s, {res['tokens_per_s']} tokens/s", flush=True)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="plain"); p.add_argument("--tier", default="5s", choices=list(TIERS))
    p.add_argument("--lr", type=float, default=0); p.add_argument("--steps", type=int, default=0); p.add_argument("--tag", default="")
    p.add_argument("--dev", default=""); p.add_argument("--threads", type=int, default=1); p.add_argument("--max-seconds", type=float, default=0); p.add_argument("--final", action="store_true"); p.add_argument("--keep", action="store_true")
    run(p.parse_args())
