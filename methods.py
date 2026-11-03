"""Memory methods. A method is a function (c, W, dev) -> dict(mems={layer: module}, value={layer: module}, stem={layer: module}, moe=experts).

mems: takes (token ids, hidden state) and returns a hidden state, run before that layer's block.
value: takes (token ids, hidden state, attention values) and returns the values that block's attention reads.
stem: its .tab replaces that block's FFN up-projection.
Tables must be sparse nn.Embeddings under .tab and projections named .w*, so the optimizers find them. A new design only needs
a new entry in METHODS. Published settings are kept where the paper gives them; the docstring says what was adapted.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def primes_near(n, k):
    out, c = [], n
    while len(out) < k:
        c += 1
        if all(c % q for q in range(2, int(c ** 0.5) + 1)): out.append(c)
    return out


class Read(nn.Module):
    """Engram's read path: key and value projections, gate sigmoid(RMSNorm(h) . RMSNorm(k) / sqrt d), causal depthwise conv + residual."""
    def __init__(s, d, dtab):
        super().__init__()
        s.wk, s.wv = nn.Linear(dtab, d, bias=False), nn.Linear(dtab, d, bias=False)
        s.nh, s.nk, s.nv = nn.RMSNorm(d), nn.RMSNorm(d), nn.RMSNorm(d)
        s.conv = nn.Conv1d(d, d, 4, groups=d, dilation=3, bias=False); nn.init.zeros_(s.conv.weight)

    def read(s, e, h):
        k, v = s.wk(e), s.wv(e)
        al = torch.sigmoid((s.nh(h) * s.nk(k)).sum(-1, keepdim=True) / h.shape[-1] ** 0.5); vt = al * v
        return h + F.silu(s.conv(F.pad(s.nv(vt).transpose(1, 2), (9, 0))).transpose(1, 2)) + vt


class Engram(Read):
    """Suffix n-grams of canonical ids, K hash heads per order, prime table sizes, multiplicative-XOR hash (arXiv 2601.07372)."""
    def __init__(s, c, W, dev, orders=(2, 3), K=4):
        d = c["width"]; dh = d // (len(orders) * K); super().__init__(d, len(orders) * K * dh)
        s.ords, s.dh, s.K = orders, dh, K; g = np.random.default_rng(1); s.comp = torch.as_tensor(W.comp, device=dev)
        s.M = {n: primes_near(c["slots"] // K, K) for n in orders}; s.mult = {n: g.integers(1, 2 ** 20, (K, n)) * 2 + 1 for n in orders}
        s.tab = nn.ModuleList(nn.Embedding(m, dh, sparse=dev != "mps") for n in orders for m in s.M[n])
        for t in s.tab: nn.init.normal_(t.weight, std=dh ** -0.5)

    def keys(s, x):
        c = s.comp[x]; out = []
        for n in s.ords:
            for k, m in enumerate(s.M[n]):
                key = torch.zeros_like(c)
                for i in range(n): key = key ^ (F.pad(c, (i, 0), value=0)[:, :c.shape[1]] * int(s.mult[n][k, i]))
                out.append(key % m)
        return out

    def forward(s, x, h):
        return s.read(torch.cat([t(k) for t, k in zip(s.tab, s.keys(x))], -1), h)

    def table_stats(s, X):
        """Share of rows ever hit by the training text, and distinct n-grams per row hit."""
        hit, per = [], []; X = X[:20000]; keys = s.keys(X); j = 0
        for n in s.ords:
            c = s.comp[X].cpu().numpy(); gram = np.zeros(c.shape, np.int64)
            for i in range(n): gram = gram * 1000003 + np.pad(c, ((0, 0), (i, 0)))[:, :c.shape[1]]
            U = len(np.unique(gram))
            for m in s.M[n]:
                R = len(torch.unique(keys[j])); hit.append(R / m); per.append(U / R); j += 1
        return dict(rows_hit=round(float(np.mean(hit)), 4), grams_per_row=round(float(np.mean(per)), 3))

    def active(s):
        return len(s.ords) * s.K * s.dh


class TokenTable(Read):
    """Control: one row per canonical token, the same table weights as Engram at this tier, so any gain here needs no n-grams."""
    def __init__(s, c, W, dev):
        d = c["width"]; dh0 = d // 8; dh = max(1, 2 * 4 * (c["slots"] // 4) * dh0 // W.ncanon); super().__init__(d, dh)
        s.dh = dh; s.comp = torch.as_tensor(W.comp, device=dev)
        s.tab = nn.ModuleList([nn.Embedding(W.ncanon, dh, sparse=dev != "mps")]); nn.init.normal_(s.tab[0].weight, std=dh ** -0.5)

    def forward(s, x, h):
        return s.read(s.tab[0](s.comp[x]), h)

    def table_stats(s, X):
        return dict(rows_hit=round(len(torch.unique(s.comp[X])) / s.tab[0].num_embeddings, 4), grams_per_row=1.0)

    def active(s):
        return s.dh


class ValueTable(nn.Module):
    """Value Embedding (modded-nanogpt, nanochat): one row per token, V x H x d_value, added to the attention values."""
    def __init__(s, c, W, dev):
        super().__init__()
        s.H, s.dk = c["heads"], c["width"] // c["heads"]; s.tab = nn.Embedding(W.V, c["width"], sparse=dev != "mps"); nn.init.normal_(s.tab.weight, std=0.1)

    def forward(s, ids, h, v):
        b, t = ids.shape; return v + s.tab(ids).view(b, t, s.H, s.dk).transpose(1, 2)


class MoME(nn.Module):
    """MoME (arXiv 2609.15126): V x M x d_value memory slots; each head routes to its top 2 slots, gated by the hidden state entering the block."""
    def __init__(s, c, W, dev):
        super().__init__()
        d = c["width"]; s.H, s.M, s.dk = c["heads"], c["heads"], d // c["heads"]; s.K = min(2, s.M); s.tab = nn.Embedding(W.V, s.M * s.dk, sparse=dev != "mps"); nn.init.normal_(s.tab.weight, std=0.1)
        s.router = nn.Linear(d, s.H * s.M); s.router_gate = nn.Linear(d, s.H)

    def forward(s, ids, h, v):
        b, t = ids.shape; E = s.tab(ids).view(b, t, 1, s.M, s.dk).expand(b, t, s.H, s.M, s.dk); l = s.router(h).view(b, t, s.H, s.M)
        idx = l.topk(s.K, -1).indices; w = torch.sigmoid(l).gather(-1, idx); w = w / w.sum(-1, keepdim=True)
        m = (E.gather(3, idx[..., None].expand(b, t, s.H, s.K, s.dk)) * w[..., None]).sum(3)
        return v + (2 * torch.sigmoid(s.router_gate(h))[..., None] * m).transpose(1, 2)


class Stem(nn.Module):
    """STEM (arXiv 2601.10639, CMU + Meta): a V x d_ffn table replaces the FFN up-projection; gate and down-projection stay dense. The 1/2 setting."""
    def __init__(s, c, W, dev):
        super().__init__()
        s.tab = nn.Embedding(W.V, int(8 * c["width"] / 3), sparse=dev != "mps"); nn.init.normal_(s.tab.weight, std=3 ** -0.5)  # same scale as the dense up-projection's output


class Bigram(nn.Module):
    """Bigram (modded-nanogpt): one hashed table of (previous, current) token shared across memory layers, added through per-head gates and a learned per-layer scalar."""
    def __init__(s, c, tab, N):
        super().__init__()
        d = c["width"]; s.H, s.dk, s.N = c["heads"], d // c["heads"], N; s.tab = tab; s.nh = nn.RMSNorm(d); s.router = nn.Linear(d, s.H); s.scale = nn.Parameter(torch.ones(()))

    def forward(s, x, h):
        b, t = x.shape; k = (F.pad(x, (1, 0), value=0)[:, :t] * 1000003 ^ x) % s.N; g = 2 * torch.sigmoid(s.router(s.nh(h)))[..., None]
        return h + s.scale * (g * s.tab(k).view(b, t, s.H, s.dk)).flatten(-2)


class Broken(MoME):
    def forward(s, ids, h, v):
        return super().forward(torch.randint_like(ids, s.tab.num_embeddings), h, v)


odd = lambda c: range(1, c["layers"], 2)


def bigram(c, W, dev, rows=0):
    """Table rows = V x memory layers, MoME's iso-parameter rule (6V at their 6 memory layers)."""
    N = primes_near((rows or len(odd(c))) * W.V, 1)[0]; tab = nn.Embedding(N, c["width"], sparse=dev != "mps"); nn.init.normal_(tab.weight, std=0.1)
    return dict(mems={k: Bigram(c, tab, N) for k in odd(c)})


class TNgram(nn.Module):
    """TN-gram (arXiv 2606.08347): suffix n-grams 2..N as a tensor product of per-position token factors, no hash and no canonical ids.

    The paper has V=1024 and R=1024 with modules at layers 1 and 7 of 9. Here R is set so the factors hold Engram's table weights at this tier, one module at Engram's layer."""
    def __init__(s, c, W, dev, N=5):
        super().__init__()
        d = c["width"]; s.N = N; s.R = max(2, c["slots"] * d // (4 * N * W.V))
        s.tab = nn.ModuleList(nn.Embedding(W.V, s.R, sparse=dev != "mps") for _ in range(N))
        for t in s.tab: nn.init.normal_(t.weight, std=1.0)
        s.order = nn.Parameter(torch.ones(N - 2, s.R)); s.logs = nn.Parameter(torch.zeros(N - 1)); s.norms = nn.ModuleList(nn.RMSNorm(s.R) for _ in range(N - 1))
        s.wk, s.wv = nn.Linear((N - 1) * s.R, d, bias=False), nn.Linear((N - 1) * s.R, d, bias=False); s.nh, s.nk, s.nv = nn.RMSNorm(d), nn.RMSNorm(d), nn.RMSNorm(d)
        s.conv = nn.Conv1d(d, d, 3, groups=d, dilation=N, bias=False); nn.init.zeros_(s.conv.weight)

    def forward(s, x, h):
        t = x.shape[1]; A = [s.tab[i](F.pad(x, (s.N - 1 - i, 0), value=0)[:, :t]) for i in range(s.N)]; es = []; b = A[-1]
        for n in range(2, s.N + 1):  # order n multiplies the last n factors; the unused positions are the learned order vectors
            b = b * A[s.N - n]; o = s.order[:s.N - n].prod(0) if n < s.N else 1.0; es.append(s.logs[n - 2].exp() * s.norms[n - 2](b * o))
        e = torch.cat(es, -1); k, v = s.wk(e), s.wv(e); al = (s.nh(h) * s.nk(k)).sum(-1, keepdim=True) / h.shape[-1] ** 0.5
        vt = torch.sigmoid(al.sign() * al.abs().clamp_min(1e-6).sqrt()) * v
        return h + vt + F.silu(s.conv(F.pad(s.nv(vt).transpose(1, 2), (2 * s.N, 0))).transpose(1, 2))


class Slots(nn.Module):
    """Our test: MoME's read path, but the slots come from several tables, each keyed by the last n tokens. rows = (n, table rows, slots)."""
    def __init__(s, c, W, dev, rows):
        super().__init__()
        d = c["width"]; s.H, s.dk = c["heads"], d // c["heads"]; s.rows = [(n, primes_near(N, 1)[0] if n > 1 else W.V, m) for n, N, m in rows]; s.M = sum(m for _, _, m in rows); s.K = min(2, s.M)
        s.tab = nn.ModuleList(nn.Embedding(N, m * s.dk, sparse=dev != "mps") for _, N, m in s.rows)
        for t in s.tab: nn.init.normal_(t.weight, std=0.1)
        s.router = nn.Linear(d, s.H * s.M); s.router_gate = nn.Linear(d, s.H)

    def keys(s, x, n, N):
        k = torch.zeros_like(x)
        for i in range(n): k = k * 1000003 ^ F.pad(x, (n - 1 - i, 0), value=0)[:, :x.shape[1]]
        return k % N

    def forward(s, ids, h, v):
        b, t = ids.shape; E = torch.cat([tb(s.keys(ids, n, N)) for tb, (n, N, _) in zip(s.tab, s.rows)], -1).view(b, t, 1, s.M, s.dk).expand(b, t, s.H, s.M, s.dk); l = s.router(h).view(b, t, s.H, s.M)
        idx = l.topk(s.K, -1).indices; w = torch.sigmoid(l).gather(-1, idx); w = w / w.sum(-1, keepdim=True)
        m = (E.gather(3, idx[..., None].expand(b, t, s.H, s.K, s.dk)) * w[..., None]).sum(3)
        return v + (2 * torch.sigmoid(s.router_gate(h))[..., None] * m).transpose(1, 2)


class Reader(nn.Module):
    """Our test: MoME's table, but attention carries every slot and the reading position picks among them, so the slot depends on the question, not on the token being read."""
    def __init__(s, c, W, dev):
        super().__init__()
        d = c["width"]; s.H, s.M, s.dk = c["heads"], c["heads"], d // c["heads"]; s.K = min(2, s.M); s.tab = nn.Embedding(W.V, s.M * s.dk, sparse=dev != "mps"); nn.init.normal_(s.tab.weight, std=0.1)
        s.router = nn.Linear(d, s.H * s.M); s.router_gate = nn.Linear(d, s.H)

    def forward(s, ids, h, v):
        b, t = ids.shape; return torch.cat([v, s.tab(ids)[:, None].expand(b, s.H, t, s.M * s.dk)], -1)

    def after(s, y, h):
        b, H, t, _ = y.shape; m = y[..., s.dk:].view(b, H, t, s.M, s.dk); l = s.router(h).view(b, t, H, s.M).transpose(1, 2)
        idx = l.topk(s.K, -1).indices; w = torch.sigmoid(l).gather(-1, idx); w = w / w.sum(-1, keepdim=True)
        m = (m.gather(3, idx[..., None].expand(b, H, t, s.K, s.dk)) * w[..., None]).sum(3)
        return y[..., :s.dk] + 2 * torch.sigmoid(s.router_gate(h)).transpose(1, 2)[..., None] * m


class AttPair(nn.Module):
    """Attention's value gets one more term: a hashed table keyed by the unordered pair of tokens a head attends to, so "France" + "capital" and "France" + "language" get different rows in any wording.
    Each head reads its A most attended tokens; a token paired with itself is the row for that token alone, which is the fallback for a pair seen rarely."""
    def __init__(s, c, W, dev, A=4):
        super().__init__()
        d = c["width"]; s.H, s.dk, s.A, s.V = c["heads"], d // c["heads"], A, W.V; s.N = primes_near(W.V * s.H, 1)[0]; s.tab = nn.Embedding(s.N, s.dk, sparse=dev != "mps"); nn.init.normal_(s.tab.weight, std=0.01)
        s.gate = nn.Linear(d, s.H)
    def forward(s, ids, h, v): return v
    def pair(s, ids, h, q, k):
        b, H, t, dk = q.shape; mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), 1)
        a = (q @ k.transpose(-1, -2) / dk ** 0.5).masked_fill(mask, float("-inf")).softmax(-1); A = min(s.A, t); av, ai = a.topk(A, -1)
        tok = ids[:, None, None].expand(b, H, t, t).gather(-1, ai); i, j = torch.triu_indices(A, A, device=q.device); x, y = tok[..., i], tok[..., j]
        key = ((x.minimum(y) * s.V + x.maximum(y)) * 1000003) % s.N; w = av[..., i] * av[..., j] * (1 + (i != j).float())
        return 2 * torch.sigmoid(s.gate(h)).transpose(1, 2)[..., None] * (s.tab(key) * w[..., None]).sum(3)


class Triple(nn.Module):
    """Our test, round 6, for the relation test: one hashed table keyed by (token read, code of the reading position). Each head reads its A most attended tokens and picks 2 of C codes
    from its own hidden state, so a fact is stored under the name and the kind of question. Another question picks another code and finds another row. Paraphrases gain from sharing a code, because the row is already trained."""
    def __init__(s, c, W, dev, C=16, A=4):
        super().__init__()
        d = c["width"]; s.H, s.dk, s.C, s.A, s.K = c["heads"], d // c["heads"], C, A, 2; s.N = primes_near(W.V * s.H, 1)[0]; s.tab = nn.Embedding(s.N, s.dk, sparse=dev != "mps"); nn.init.normal_(s.tab.weight, std=0.01)
        s.router = nn.Linear(d, s.H * C); s.router_gate = nn.Linear(d, s.H)

    def forward(s, ids, h, v):
        return v

    def pair(s, ids, h, q, k):
        b, H, t, dk = q.shape; mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), 1)
        a = (q @ k.transpose(-1, -2) / dk ** 0.5).masked_fill(mask, float("-inf")).softmax(-1); av, ai = a.topk(min(s.A, t), -1)
        l = s.router(h).view(b, t, H, s.C).transpose(1, 2); ci = l.topk(s.K, -1).indices; w = torch.sigmoid(l).gather(-1, ci); w = w / w.sum(-1, keepdim=True)
        tok = ids[:, None, None].expand(b, H, t, t).gather(-1, ai); key = ((tok[..., None] * s.C + ci[..., None, :]) * 1000003) % s.N
        m = (s.tab(key) * (av[..., None] * w[..., None, :])[..., None]).sum((3, 4))
        return 2 * torch.sigmoid(s.router_gate(h)).transpose(1, 2)[..., None] * m


class Agree(Triple):
    """Triple whose gate also asks: does this reader's state already expect the kind of word the table proposes? A paraphrase of "The capital of" still expects a city and "The language of" doesn't, so the same row can stay on for the first and go off for the second. A boost that ignores the question adds the same score to both sides, which is what lowers the relation gap."""
    def __init__(s, c, W, dev, C=16, A=4, cos=True, sharp=4, free=True):
        super().__init__(c, W, dev, C, A); d = c["width"]; s.expect = nn.Linear(d, d, bias=False); s.cos, s.sharp, s.free = cos, sharp, free

    def pair(s, ids, h, q, k):
        b, H, t, dk = q.shape; mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), 1)
        a = (q @ k.transpose(-1, -2) / dk ** 0.5).masked_fill(mask, float("-inf")).softmax(-1); av, ai = a.topk(min(s.A, t), -1)
        l = s.router(h).view(b, t, H, s.C).transpose(1, 2); ci = l.topk(s.K, -1).indices; w = torch.sigmoid(l).gather(-1, ci); w = w / w.sum(-1, keepdim=True)
        tok = ids[:, None, None].expand(b, H, t, t).gather(-1, ai); key = ((tok[..., None] * s.C + ci[..., None, :]) * 1000003) % s.N
        m = (s.tab(key) * (av[..., None] * w[..., None, :])[..., None]).sum((3, 4)); e = s.expect(h).view(b, t, H, dk).transpose(1, 2)
        fit = s.sharp * F.cosine_similarity(e, m, -1, 1e-6) if s.cos else (e * m).sum(-1) / dk ** 0.5
        return 2 * torch.sigmoid((s.router_gate(h).transpose(1, 2) if s.free else s.router_gate.bias[:, None]) + fit)[..., None] * m


class AgreePair(AttPair):
    """AttPair with the same gate as Agree."""
    def __init__(s, c, W, dev, A=4, rows=1):
        super().__init__(c, W, dev, A); d = c["width"]; s.expect = nn.Linear(d, d, bias=False)
        if rows != 1: s.N = primes_near(W.V * s.H * rows, 1)[0]; s.tab = nn.Embedding(s.N, s.dk, sparse=dev != "mps"); nn.init.normal_(s.tab.weight, std=0.01)

    def pair(s, ids, h, q, k):
        b, H, t, dk = q.shape; mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), 1)
        a = (q @ k.transpose(-1, -2) / dk ** 0.5).masked_fill(mask, float("-inf")).softmax(-1); A = min(s.A, t); av, ai = a.topk(A, -1)
        tok = ids[:, None, None].expand(b, H, t, t).gather(-1, ai); i, j = torch.triu_indices(A, A, device=q.device); x, y = tok[..., i], tok[..., j]
        key = ((x.minimum(y) * s.V + x.maximum(y)) * 1000003) % s.N; w = av[..., i] * av[..., j] * (1 + (i != j).float()); m = (s.tab(key) * w[..., None]).sum(3)
        e = s.expect(h).view(b, t, H, dk).transpose(1, 2); return 2 * torch.sigmoid(s.gate(h).transpose(1, 2) + 4 * F.cosine_similarity(e, m, -1, 1e-6))[..., None] * m


class Expects:
    """The gate of Agree in output space, with both sides trained to predict the next word: one head reads the reader's state, one reads the table's proposal. Their cosine then says whether the state expects the kind of word the table proposes. The two heads add a training term and nothing at test time beyond the gate."""
    def bind(s, emb): s.emb = [emb]

    def fit_gate(s, ids, h, m, gate):
        b, H, t, dk = m.shape; e = s.expect(h); p = s.prop(m.transpose(1, 2).reshape(b, t, H * dk))
        if s.training:
            w = s.emb[0].weight.detach(); y = ids[:, 1:].flatten(); ce = lambda z: F.cross_entropy((z[:, :-1] @ w.T).flatten(0, 1), y, ignore_index=s.unk)
            s.aux = s.aux_w * (ce(e) + ce(p))
        return 2 * torch.sigmoid(gate + s.sharp * F.cosine_similarity(e, p, -1, 1e-6)[:, None])[..., None] * m


class ExpectTriple(Expects, Triple):
    def __init__(s, c, W, dev, C=16, A=4, aux_w=0.1, sharp=4):
        Triple.__init__(s, c, W, dev, C, A); d = c["width"]; s.expect, s.prop = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False); s.aux_w, s.sharp, s.unk, s.aux = aux_w, sharp, W.unk, 0.0

    def pair(s, ids, h, q, k):
        b, H, t, dk = q.shape; mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), 1)
        a = (q @ k.transpose(-1, -2) / dk ** 0.5).masked_fill(mask, float("-inf")).softmax(-1); av, ai = a.topk(min(s.A, t), -1)
        l = s.router(h).view(b, t, H, s.C).transpose(1, 2); ci = l.topk(s.K, -1).indices; w = torch.sigmoid(l).gather(-1, ci); w = w / w.sum(-1, keepdim=True)
        tok = ids[:, None, None].expand(b, H, t, t).gather(-1, ai); key = ((tok[..., None] * s.C + ci[..., None, :]) * 1000003) % s.N
        return s.fit_gate(ids, h, (s.tab(key) * (av[..., None] * w[..., None, :])[..., None]).sum((3, 4)), s.router_gate(h).transpose(1, 2))


class ExpectPair(Expects, AttPair):
    def __init__(s, c, W, dev, A=4, aux_w=0.1, sharp=4):
        AttPair.__init__(s, c, W, dev, A); d = c["width"]; s.expect, s.prop = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False); s.aux_w, s.sharp, s.unk, s.aux = aux_w, sharp, W.unk, 0.0

    def pair(s, ids, h, q, k):
        b, H, t, dk = q.shape; mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), 1)
        a = (q @ k.transpose(-1, -2) / dk ** 0.5).masked_fill(mask, float("-inf")).softmax(-1); A = min(s.A, t); av, ai = a.topk(A, -1)
        tok = ids[:, None, None].expand(b, H, t, t).gather(-1, ai); i, j = torch.triu_indices(A, A, device=q.device); x, y = tok[..., i], tok[..., j]
        key = ((x.minimum(y) * s.V + x.maximum(y)) * 1000003) % s.N; w = av[..., i] * av[..., j] * (1 + (i != j).float())
        return s.fit_gate(ids, h, (s.tab(key) * w[..., None]).sum(3), s.gate(h).transpose(1, 2))


class WordBucket(nn.Module):
    """Pair keys with sharing: a row is keyed by one attended token and the bucket of another, so "capital" and "capitals" can share rows while "capital" and "language" don't. Buckets come from fixed random hyperplanes over the word embeddings, so they can't collapse the way a trained router does, and they stop moving after the first 30% of training so rows don't go stale. The gate is Agree's."""
    def __init__(s, c, W, dev, bits=4, A=4):
        super().__init__()
        d = c["width"]; s.H, s.dk, s.A, s.C = c["heads"], d // c["heads"], A, 2 ** bits; s.N = primes_near(W.V * s.H, 1)[0]; s.tab = nn.Embedding(s.N, s.dk, sparse=dev != "mps"); nn.init.normal_(s.tab.weight, std=0.01)
        s.gate = nn.Linear(d, s.H); s.expect = nn.Linear(d, d, bias=False); s.stop = int(c["steps"] * 0.3)
        s.register_buffer("code", torch.zeros(W.V, dtype=torch.long)); s.register_buffer("t", torch.zeros(1, dtype=torch.long)); s.register_buffer("planes", torch.randn(d, bits) * 1.0); s.register_buffer("pow", 2 ** torch.arange(bits))

    def bind(s, emb): s.emb = [emb]

    def forward(s, ids, h, v): return v

    def pair(s, ids, h, q, k):
        if s.training:
            s.t += 1
            if int(s.t) <= s.stop and int(s.t) % 25 == 1:
                with torch.no_grad(): w = s.emb[0].weight; s.code[:] = (((w - w.mean(0)) @ s.planes > 0).long() * s.pow).sum(-1)
        b, H, t, dk = q.shape; mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), 1)
        a = (q @ k.transpose(-1, -2) / dk ** 0.5).masked_fill(mask, float("-inf")).softmax(-1); A = min(s.A, t); av, ai = a.topk(A, -1); tok = ids[:, None, None].expand(b, H, t, t).gather(-1, ai)
        key = ((tok[..., None] * s.C + s.code[tok][..., None, :]) * 1000003) % s.N; m = (s.tab(key) * (av[..., None] * av[..., None, :])[..., None]).sum((3, 4))
        e = s.expect(h).view(b, t, H, dk).transpose(1, 2); return 2 * torch.sigmoid(s.gate(h).transpose(1, 2) + 4 * F.cosine_similarity(e, m, -1, 1e-6))[..., None] * m


class Both(nn.Module):
    """Two readers in one attention, Agree for facts in a new wording and AgreePair for the relation."""
    def __init__(s, c, W, dev):
        super().__init__(); s.a, s.b = Agree(c, W, dev), AgreePair(c, W, dev)

    def forward(s, ids, h, v): return v

    def pair(s, ids, h, q, k): return s.a.pair(ids, h, q, k) + s.b.pair(ids, h, q, k)


class TripleBal(Triple):
    """Triple with MoE's load balancing on the codes: a per-head bias pushes down codes picked too often, so the router can't collapse to 2 codes and stays free to learn which questions share one."""
    def __init__(s, c, W, dev, C=16, A=4):
        super().__init__(c, W, dev, C, A); s.register_buffer("rbias", torch.zeros(c["heads"], C))
    def pair(s, ids, h, q, k):
        if s.training:
            with torch.no_grad():
                b, t = h.shape[:2]; idx = (s.router(h).view(b, t, s.H, s.C) - s.rbias).topk(s.K, -1).indices.flatten(0, 1); load = F.one_hot(idx, s.C).float().sum((0, 2)); s.rbias += 1e-3 * torch.sign(load - load.mean(-1, keepdim=True))
        return s.read_pair(ids, h, q, k)
    def read_pair(s, ids, h, q, k):
        b, H, t, dk = q.shape; mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), 1)
        a = (q @ k.transpose(-1, -2) / dk ** 0.5).masked_fill(mask, float("-inf")).softmax(-1); av, ai = a.topk(min(s.A, t), -1)
        l = (s.router(h).view(b, t, H, s.C) - s.rbias).transpose(1, 2); ci = l.topk(s.K, -1).indices; w = torch.sigmoid(l).gather(-1, ci); w = w / w.sum(-1, keepdim=True)
        tok = ids[:, None, None].expand(b, H, t, t).gather(-1, ai); key = ((tok[..., None] * s.C + ci[..., None, :]) * 1000003) % s.N
        m = (s.tab(key) * (av[..., None] * w[..., None, :])[..., None]).sum((3, 4))
        return 2 * torch.sigmoid(s.router_gate(h)).transpose(1, 2)[..., None] * m


class TripleHash(Triple):
    """Triple with codes nothing can train: fixed random directions over the reader's hidden state, centred by a running mean. Training shows a name with one relation, so a learned router has no reason to split relations and collapses to 2 codes."""
    def __init__(s, c, W, dev, C=16, A=4):
        super().__init__(c, W, dev, C, A); s.router.requires_grad_(False); s.register_buffer("mean", torch.zeros(c["width"]))
        with torch.no_grad(): s.router.bias.zero_()
    def pair(s, ids, h, q, k):
        if s.training:
            with torch.no_grad(): s.mean.lerp_(h.mean((0, 1)), 0.01)
        return super().pair(ids, h - s.mean, q, k)


class DropEngram(Engram):
    """Our test: Engram's output is dropped at half the positions in training, so a fact can't be stored only under its exact wording."""
    def __init__(s, c, W, dev, p=0.5):
        super().__init__(c, W, dev); s.p = p

    def forward(s, x, h):
        o = super().forward(x, h)
        if not s.training: return o
        return h + (o - h) * (torch.rand(x.shape, device=x.device) > s.p)[..., None] / (1 - s.p)


class Gated(nn.Module):
    """Our test: VE's table, but attention carries it apart from the values and the reading position multiplies what comes back, so the answer depends on the name read and on the question asking."""
    def __init__(s, c, W, dev):
        super().__init__()
        d = c["width"]; s.H, s.dk = c["heads"], d // c["heads"]; s.tab = nn.Embedding(W.V, d, sparse=dev != "mps"); nn.init.normal_(s.tab.weight, std=0.1)
        s.wg = nn.Linear(d, d, bias=False); nn.init.zeros_(s.wg.weight)

    def forward(s, ids, h, v):
        b, t = ids.shape; return torch.cat([v, s.tab(ids).view(b, t, s.H, s.dk).transpose(1, 2)], -1)

    def after(s, y, h):
        b, H, t, _ = y.shape; a, m = y.split(s.dk, -1); return a + m * (1 + s.wg(h).view(b, t, H, s.dk).transpose(1, 2))


def gram_key(x, N):
    return (F.pad(x, (1, 0), value=0)[:, :x.shape[1]] * 1000003 ^ x) % N


class QGram(ValueTable):
    """Our test: VE plus a hashed table of the reader's last 2 tokens added to its query. The phrase steers where attention looks and writes no answer itself."""
    def __init__(s, c, W, dev, rows):
        super().__init__(c, W, dev); s.N = primes_near(rows, 1)[0]; s.tabq = nn.Embedding(s.N, c["width"], sparse=dev != "mps"); nn.init.normal_(s.tabq.weight, std=0.1)

    def query(s, ids, q):
        b, t = ids.shape; return q + s.tabq(gram_key(ids, s.N)).view(b, t, s.H, s.dk).transpose(1, 2)


class ValueGram(ValueTable):
    """Our test: VE plus a hashed table of (previous, current) token, both added to the values, so a two-token name gets its own stored vector."""
    def __init__(s, c, W, dev, rows):
        super().__init__(c, W, dev); s.N = primes_near(rows, 1)[0]; s.tabg = nn.Embedding(s.N, c["width"], sparse=dev != "mps"); nn.init.normal_(s.tabg.weight, std=0.1)

    def forward(s, ids, h, v):
        b, t = ids.shape; return super().forward(ids, h, v) + s.tabg(gram_key(ids, s.N)).view(b, t, s.H, s.dk).transpose(1, 2)


class RSlot(Reader):
    """Our test: Reader, but the reader's last 2 tokens also choose the slot, so the phrase "was born in" can pick the birthplace slot of the name."""
    def __init__(s, c, W, dev, rows):
        super().__init__(c, W, dev); s.N = primes_near(rows, 1)[0]; s.tabq = nn.Embedding(s.N, c["width"], sparse=dev != "mps"); nn.init.normal_(s.tabq.weight, std=0.1)

    def forward(s, ids, h, v):
        s.phrase = s.tabq(gram_key(ids, s.N)); return super().forward(ids, h, v)

    def after(s, y, h):
        return super().after(y, h + s.phrase)


def late(make, value=None):
    """Our test, round 2: a value table inside attention plus Engram after the last block, in one table budget. Value tables get every memory layer if they fit in 3/4 of Engram's table weights,
    else only the last one, and the late Engram gets the rest. Late, the n-gram can't memorize an answer before attention reads the name, so the value table still has to learn it."""
    def build(c, W, dev):
        d = c["width"]; dh = d // 8; budget = 2 * c["slots"] * dh * (1.25 if c["layers"] <= 2 else 0.995); ks = list(odd(c)); out = {}
        if make: ks = ks if len(ks) * value * W.V * d <= budget * 3 / 4 else ks[-1:]; out["value"] = {k: make(c, W, dev) for k in ks}
        S = int(budget - (len(ks) * value * W.V * d if make else 0)) // (2 * dh); out["mems"] = {c["layers"]: Engram(dict(c, slots=S), W, dev)}; return out
    return build


class ZeroEngram(Engram):
    """Our test: Engram whose output starts at zero, so the table only learns what the model above it still gets wrong."""
    def __init__(s, c, W, dev):
        super().__init__(c, W, dev); nn.init.zeros_(s.wv.weight)


mix2 = lambda c, W, dev: Slots(c, W, dev, [(1, W.V, c["heads"] // 2), (2, W.V, c["heads"] // 2)])


def early(make, gram=Engram):
    """Our test, round 7: a value table in the last memory layer plus Engram after the first block with the rest of the budget, where Engram alone gets the best relation score. The phrase reaches every later block."""
    def build(c, W, dev):
        d = c["width"]; dh = d // 8; budget = 2 * c["slots"] * dh * 0.985; S = int(budget - W.V * d) // (2 * dh)
        return dict(mems={1: gram(dict(c, slots=S), W, dev)}, value={list(odd(c))[-1]: make(c, W, dev)})
    return build


def second(make):
    """Our test, round 8: the word table after block 1, Engram after block 2, the value table in the last memory layer. Between the late Engram of three() and the first-block Engram of early()."""
    def build(c, W, dev):
        d = c["width"]; dh = d // 8; budget = 2 * c["slots"] * dh * 0.985; S = int(budget - W.V * d) // (4 * dh)
        return dict(mems={1: TokenTable(dict(c, slots=S), W, dev), 2: Engram(dict(c, slots=S), W, dev)}, value={list(odd(c))[-1]: make(c, W, dev)})
    return build


def three(make, where=-1, tw=1, gw=1, gram=None, out=False, moe=0, every=False, rows=1):
    """Our test, round 3: three tables in one budget, each for what it did best in round 2. A word table after the first block, where the baselines put theirs, for text, a slot table in the last memory layer for facts, Engram after the last block for phrases."""
    def build(c, W, dev):
        d = c["width"]; dh = d // 8; budget = 2 * c["slots"] * dh * (1.25 if c["layers"] <= 2 else 0.98 if c["layers"] >= 6 else 0.985); hid = int(8 * d / 3)
        if moe: budget -= c["layers"] * d * (3 * (hid // 2) + 3 * moe * (hid // 4) + moe - 3 * hid)  # MoE's extra weights come out of the table budget
        S = int(budget - W.V * d * (len(odd(c)) if every else rows)) // (2 * (tw + gw) * dh); mems = {"out" if out else c["layers"]: (gram or Engram)(dict(c, slots=S * gw), W, dev)} if gw else {}
        if tw: mems[1] = TokenTable(dict(c, slots=S * tw), W, dev)
        return dict(mems=mems, value={k: make(c, W, dev) for k in (odd(c) if every else [list(odd(c))[where]])}, moe=moe)
    return build


def engmoe(c, W, dev):
    """Control: Engram with MoE, nothing of ours, with MoE's extra weights taken out of the table so the total stays at Engram's."""
    d = c["width"]; dh = d // 8; hid = int(8 * d / 3); extra = c["layers"] * d * (3 * (hid // 2) + 3 * 8 * (hid // 4) + 8 - 3 * hid)
    return dict(mems={1: Engram(dict(c, slots=int(c["slots"] * 0.99 - extra / (2 * dh))), W, dev)}, moe=8)


small = lambda c: c["slots"] // (16 * len(odd(c)))  # rows that hold about a quarter of Engram's table weights across the memory layers
quarter = lambda c: dict(c, slots=c["slots"] // 4)
slots = lambda rows: lambda c, W, dev: dict(value={k: Slots(c, W, dev, rows(c, W)) for k in odd(c)})
METHODS = {
    "plain": lambda c, W, dev: {},
    "engram": lambda c, W, dev: dict(mems={1: Engram(c, W, dev)}),
    "token": lambda c, W, dev: dict(mems={1: TokenTable(c, W, dev)}),
    "moe": lambda c, W, dev: dict(moe=8),
    "ve": lambda c, W, dev: dict(value={k: ValueTable(c, W, dev) for k in odd(c)}),
    "mome": lambda c, W, dev: dict(value={k: MoME(c, W, dev) for k in odd(c)}),
    "stem": lambda c, W, dev: dict(stem={k: Stem(c, W, dev) for k in odd(c)}),
    "bigram": bigram,
    "bigram6": lambda c, W, dev: bigram(c, W, dev, 6),
    "tngram": lambda c, W, dev: dict(mems={1: TNgram(c, W, dev)}),
    "reader": lambda c, W, dev: dict(value={k: Reader(c, W, dev) for k in odd(c)}),
    "vgram": slots(lambda c, W: [(2, W.V, c["heads"])]),
    "mix": slots(lambda c, W: [(1, W.V, c["heads"] // 2), (2, W.V, c["heads"] // 2)]),
    "mix3": slots(lambda c, W: [(1, W.V, c["heads"] // 2), (2, W.V, c["heads"] // 4), (3, W.V, c["heads"] // 4)]),
    "mixbig": slots(lambda c, W: [(1, W.V, c["heads"] // 2), (2, 6 * W.V, c["heads"] // 2)]),
    # Our tests: value table for stored facts plus a small Engram for phrases. both adds Engram at a quarter of its rows; bothlate pays for it by keeping only the last value table.
    # Controls for the environment. broken looks up a random row per position, so it can store nothing; cheat holds 4- and 5-grams at 16x rows, so it memorizes training text word for word.
    "broken": lambda c, W, dev: dict(value={k: Broken(c, W, dev) for k in odd(c)}),
    "cheat": lambda c, W, dev: dict(mems={1: Engram(dict(c, slots=16 * c["slots"]), W, dev, orders=(4, 5))}),
    "both": lambda c, W, dev: dict(mems={1: Engram(dict(c, slots=c["slots"] // 4), W, dev)}, value={k: MoME(c, W, dev) for k in odd(c)}),
    "bothlate": lambda c, W, dev: dict(mems={1: Engram(dict(c, slots=c["slots"] // 4), W, dev)}, value={list(odd(c))[-1]: MoME(c, W, dev)}),
    "bothve": lambda c, W, dev: dict(mems={1: Engram(dict(c, slots=c["slots"] // 4), W, dev)}, value={k: ValueTable(c, W, dev) for k in odd(c)}),
    "bothread": lambda c, W, dev: dict(mems={1: Engram(dict(c, slots=c["slots"] // 4), W, dev)}, value={k: Reader(c, W, dev) for k in odd(c)}),
    # Round 1 of new designs, one idea each. EXPERIMENTS.md has the reason for each one and the win rule.
    "gated": lambda c, W, dev: dict(value={k: Gated(c, W, dev) for k in odd(c)}),
    "qgram": lambda c, W, dev: dict(value={k: QGram(c, W, dev, small(c)) for k in odd(c)}),
    "valgram": lambda c, W, dev: dict(value={k: ValueGram(c, W, dev, small(c)) for k in odd(c)}),
    "rslot": lambda c, W, dev: dict(value={k: RSlot(c, W, dev, small(c)) for k in odd(c)}),
    "dropve": lambda c, W, dev: dict(mems={1: DropEngram(quarter(c), W, dev)}, value={k: ValueTable(c, W, dev) for k in odd(c)}),
    "lateve": lambda c, W, dev: dict(mems={c["layers"]: Engram(quarter(c), W, dev)}, value={k: ValueTable(c, W, dev) for k in odd(c)}),
    # Round 2: lateve was closest, so keep the late Engram and change what sits inside attention. late alone checks whether the place explains it.
    "tmome": three(MoME),
    "treader": three(Reader),
    "tmix": three(lambda c, W, dev: Slots(c, W, dev, [(1, W.V, c["heads"] // 2), (2, W.V, c["heads"] // 2)])),
    # Round 3 changes to the two closest: slot table in the first memory layer, slots keyed by up to 3 tokens, no Engram, and more budget for Engram.
    "tmix1": three(lambda c, W, dev: Slots(c, W, dev, [(1, W.V, c["heads"] // 2), (2, W.V, c["heads"] // 2)]), where=0),
    "tmix3": three(lambda c, W, dev: Slots(c, W, dev, [(1, W.V, c["heads"] // 2), (2, W.V, c["heads"] // 4), (3, W.V, c["heads"] // 4)])),
    "tmix0": three(lambda c, W, dev: Slots(c, W, dev, [(1, W.V, c["heads"] // 2), (2, W.V, c["heads"] // 2)]), gw=0),
    "tmomeg": three(MoME, tw=1, gw=3),
    # The late Engram gave the fact gains and cost web loss, so hold it back: start it at zero, or drop it at half the positions in training.
    # Round 4, for the relation test. A table read in an early layer leaves later layers to adjust the answer to the question. tsplit keeps half a slot table in each memory layer, tgz gates the table on the reading position.
    "tmixz1": three(mix2, where=0, gram=ZeroEngram),
    "tmomez1": three(MoME, where=0, gram=ZeroEngram),
    "tgz": three(Gated, gram=ZeroEngram),
    "tbiz": three(lambda c, W, dev: Slots(c, W, dev, [(2, W.V, c["heads"])]), gram=ZeroEngram),
    "tsplit": lambda c, W, dev: dict(three(lambda c, W, dev: nn.Identity(), gram=ZeroEngram)(c, W, dev), value={k: Slots(c, W, dev, [(1, W.V, c["heads"] // 4), (2, W.V, c["heads"] // 4)]) for k in odd(c)}),
    # Better stored facts came with a worse relation gap, because the model only ever sees a name with its true relation. Here the late Engram sits after the final norm, so the phrase adds its own term to the word scores.
    "tmixo": three(mix2, gram=ZeroEngram, out=True),
    "tmomeo": three(MoME, gram=ZeroEngram, out=True),
    "tgo": three(Gated, gram=ZeroEngram, out=True),
    "tmixog": three(mix2, tw=1, gw=3, gram=ZeroEngram, out=True),
    "tmixz": three(mix2, gram=ZeroEngram),
    # Round 6, for the relation test: the slot table keyed by the name and the kind of question. trip has the table alone, so its effect shows without the other two.
    "ttrip": three(Triple, gram=ZeroEngram),
    "tpair": three(AttPair, gram=ZeroEngram),
    "thash": three(TripleHash, gram=ZeroEngram),
    "mtrip": three(Triple, gram=ZeroEngram, moe=8),
    "engmoe": engmoe,
    "vemoe": lambda c, W, dev: dict(value={k: ValueTable(c, W, dev) for k in odd(c)}, moe=8),
    "momemoe": lambda c, W, dev: dict(value={k: MoME(c, W, dev) for k in odd(c)}, moe=8),
    "magree": three(Agree, gram=ZeroEngram, moe=8),
    "magreed": three(lambda c, W, dev: Agree(c, W, dev, cos=False), gram=ZeroEngram, moe=8),
    "magree1": three(Agree, gram=ZeroEngram, moe=8, where=0),
    "magree8": three(lambda c, W, dev: Agree(c, W, dev, sharp=8), gram=ZeroEngram, moe=8),
    "magreef": three(lambda c, W, dev: Agree(c, W, dev, free=False), gram=ZeroEngram, moe=8),
    "mexp": three(ExpectTriple, gram=ZeroEngram, moe=8),
    "mexp8": three(lambda c, W, dev: ExpectTriple(c, W, dev, sharp=8), gram=ZeroEngram, moe=8),
    "mexp16": three(lambda c, W, dev: ExpectTriple(c, W, dev, sharp=16), gram=ZeroEngram, moe=8),
    "mexp32": three(lambda c, W, dev: ExpectTriple(c, W, dev, sharp=32), gram=ZeroEngram, moe=8),
    "mexp8a": three(lambda c, W, dev: ExpectTriple(c, W, dev, sharp=8, aux_w=0.3), gram=ZeroEngram, moe=8),
    "mexp8b": three(lambda c, W, dev: ExpectTriple(c, W, dev, sharp=8, aux_w=0.03), gram=ZeroEngram, moe=8),
    "mexpair": three(ExpectPair, gram=ZeroEngram, moe=8),
    "apair": three(AgreePair, gram=ZeroEngram),
    "mapair": three(AgreePair, gram=ZeroEngram, moe=8),
    "mbuck16": three(lambda c, W, dev: WordBucket(c, W, dev, bits=4), gram=ZeroEngram, moe=8),
    "mbuck64": three(lambda c, W, dev: WordBucket(c, W, dev, bits=6), gram=ZeroEngram, moe=8),
    "mbuck256": three(lambda c, W, dev: WordBucket(c, W, dev, bits=8), gram=ZeroEngram, moe=8),
    "mapair2": three(lambda c, W, dev: AgreePair(c, W, dev, rows=2), gram=ZeroEngram, moe=8, rows=2),
    "mapair8": three(lambda c, W, dev: AgreePair(c, W, dev, A=8), gram=ZeroEngram, moe=8),
    "mboth": three(Both, gram=ZeroEngram, moe=8, rows=2),
    "mtok": three(Triple, gram=ZeroEngram, moe=8, gw=0),
    "mgram": three(Triple, gram=ZeroEngram, moe=8, tw=0),
    "mtrip1": three(Triple, gram=ZeroEngram, moe=8, where=0),
    "mmixz1": three(mix2, gram=ZeroEngram, moe=8, where=0),
    "mmomez1": three(lambda c, W, dev: MoME(c, W, dev), gram=ZeroEngram, moe=8, where=0),
    "m2trip": three(Triple, gram=ZeroEngram, moe=8, every=True),
    "nmtrip2": lambda c, W, dev: dict(three(Triple, gram=ZeroEngram, moe=8)(c, W, dev), neg=0.2),
    "nmtrip1": lambda c, W, dev: dict(three(Triple, gram=ZeroEngram, moe=8)(c, W, dev), neg=1.0),
    "nmtrip5": lambda c, W, dev: dict(three(Triple, gram=ZeroEngram, moe=8)(c, W, dev), neg=5.0),
    "nplain": lambda c, W, dev: dict(neg=1.0),
    "mtrip16": three(Triple, gram=ZeroEngram, moe=16),
    "btrip": three(TripleBal, gram=ZeroEngram),
    "mbtrip": three(TripleBal, gram=ZeroEngram, moe=8),
    "mbtrip4": three(lambda c, W, dev: TripleBal(c, W, dev, C=4), gram=ZeroEngram, moe=8),
    "mmomez": three(MoME, gram=ZeroEngram, moe=8),
    "mpair": three(AttPair, gram=ZeroEngram, moe=8),
    "mhash4": three(lambda c, W, dev: TripleHash(c, W, dev, C=4), gram=ZeroEngram, moe=8),
    "mmixz": three(mix2, gram=ZeroEngram, moe=8),
    "ehash4": early(lambda c, W, dev: TripleHash(c, W, dev, C=4)),
    "ehash4z": early(lambda c, W, dev: TripleHash(c, W, dev, C=4), gram=ZeroEngram),
    "epair": early(AttPair),
    "spair": second(AttPair),
    "shash4": second(lambda c, W, dev: TripleHash(c, W, dev, C=4)),
    "tpair2": three(lambda c, W, dev: AttPair(c, W, dev, A=2), gram=ZeroEngram),
    "tpair8": three(lambda c, W, dev: AttPair(c, W, dev, A=8), gram=ZeroEngram),
    "emome": early(MoME),
    "thash4": three(lambda c, W, dev: TripleHash(c, W, dev, C=4), gram=ZeroEngram),
    "trip": lambda c, W, dev: dict(value={list(odd(c))[-1]: Triple(c, W, dev)}),
    "tmixd": three(mix2, gram=DropEngram),
    "tmomez": three(MoME, gram=ZeroEngram),
    "tmomed": three(MoME, gram=DropEngram),
    "late": late(None),
    "lve": late(ValueTable, 1),
    "lmome": late(MoME, 1),
    "lreader": late(Reader, 1),
    "lmix": late(lambda c, W, dev: Slots(c, W, dev, [(1, W.V, c["heads"] // 2), (2, W.V, c["heads"] // 2)]), 1),
    "ltoken": lambda c, W, dev: dict(mems={1: TokenTable(dict(c, slots=c["slots"] * 49 // 100), W, dev), c["layers"]: Engram(dict(c, slots=c["slots"] * 49 // 100), W, dev)}),
}


def build(name, c, W, dev):
    return METHODS[name.split("@")[0]](c, W, dev)  # engram@65536 = Engram with that many table rows per order
