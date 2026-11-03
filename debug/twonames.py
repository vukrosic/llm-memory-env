import sys, os, json, collections, numpy as np
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,HERE); os.chdir(HERE)
import env, evals
for T,meths in (("5m",["plain","moe","engram","ve","mome","stem","bigram","tngram","token","ttrip","mtrip","mbtrip","tmomez"]),):
    c=dict(env.TIERS[T]); W=env.World(c,c["steps"]); ctx=evals.Ctx(None,W,c,"cpu"); stored,*_=evals.fact_sets(ctx)
    rels=collections.defaultdict(set)
    for f in W.facts: rels[f["s"]].add(f["r"])
    rng=np.random.default_rng(4); pool=[i for i in stored if i["copies"]>=8 and not i["maj"] and not i["inq"]]; two=[]; true_other=[]
    for i in pool:
        other=[j for j in pool if j["f"]["r"]==i["f"]["r"] and j["y"]!=i["y"]]; rel=[j for j in pool if j["f"]["r"]!=i["f"]["r"]]
        if not other or not rel: continue
        o,r=other[rng.integers(len(other))],rel[rng.integers(len(rel))]; rng.integers(len(r["f"]["S"])); rng.integers(len(i["f"]["S"]))
        two.append(len(rels[i["f"]["s"]])>=2); true_other.append(r["f"]["r"] in rels[i["f"]["s"]])
    two=np.array(two); true_other=np.array(true_other); print(T,"items",len(two),"names with 2+ stored relations",two.sum(),"asked with its other true relation",true_other.sum())
    best={}
    for r in map(json.loads,open("results.jsonl")):
        if r["tier"]==T and not r["tag"] and (r["method"] not in best or r["scores"]["text.tune.loss"]<=best[r["method"]]["scores"]["text.tune.loss"]): best[r["method"]]=r
    k="changed.other-relation.gap"
    print("method lr  gap one-relation names | gap two-relation names")
    for m in meths:
        g=np.load(f"runs/{T}_{m}_lr{best[m]['lr']:g}.npz")[k]; assert len(g)==len(two)
        print(m,best[m]["lr"],round(g[~two].mean(),3),round(g[~two].std()/(~two).sum()**.5,3),"|",round(g[two].mean(),3),round(g[two].std()/max(1,two.sum())**.5,3))
