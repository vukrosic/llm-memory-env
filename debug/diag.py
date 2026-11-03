import sys, os, collections, numpy as np, torch
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,HERE); os.chdir(HERE)
import env, methods, evals
T,m,lr=sys.argv[1],sys.argv[2],float(sys.argv[3])
c=dict(env.TIERS[T]); c["lr"]=lr; W=env.World(c,c["steps"]); parts=methods.build(m,c,W,"cpu"); model=env.LM(c,W.V,parts); model.load_state_dict(torch.load(f"runs/end_{T}_{m}_lr{lr:g}.pt",map_location="cpu")); model.eval()
blk=[b for b in model.blocks if b.val is not None][0]; val=blk.val; rec={}
orig=val.pair
def pair(ids,h,q,k):
    b,H,t,dk=q.shape; l=val.router(h).view(b,t,H,val.C).transpose(1,2); rec['code']=l.argmax(-1)[0,:,-1].cpu(); rec['gate']=torch.sigmoid(val.router_gate(h))[0,-1].cpu()
    a=(q@k.transpose(-1,-2)/dk**.5).masked_fill(torch.triu(torch.ones(t,t,dtype=torch.bool),1),float('-inf')).softmax(-1)[0,:,-1]; rec['top']=a.topk(min(val.A,t),-1).indices.cpu(); rec['aw']=a.topk(min(val.A,t),-1).values.cpu(); rec['t']=t
    return orig(ids,h,q,k)
val.pair=pair
rng=np.random.default_rng(0); E=lambda p,x:[env.EOT]+W.E(p[:p.find("[Y]")].replace("[X]",x))
fs=[f for f in W.facts if f["copies"]>=8]; rels=collections.defaultdict(list)
for f in fs: rels[f["r"]].append(f)
rows=[]
with torch.no_grad():
    for r,L in rels.items():
        for f in L[:20]:
            for wording in (f["S"][0],f["T"][0]):
                ids=[int(W.look[t]) for t in E(wording,f["s"])]
                model(torch.tensor([ids])); name_ids=set(int(W.look[t]) for t in W.E(f["s"])); 
                readname=(torch.tensor([[ids[i] in name_ids for i in row] for row in rec['top'].tolist()]).float()*rec['aw']).sum(-1)
                rows.append((r,f["s"],wording,rec['code'].tolist(),rec['gate'].tolist(),readname.tolist()))
code=np.array([x[3] for x in rows]); print("items",len(rows),"heads",code.shape[1])
for h in range(code.shape[1]):
    cnt=collections.Counter(code[:,h].tolist()); print("head",h,"codes used",len(cnt),"top share",round(cnt.most_common(1)[0][1]/len(code),3), "attention on name in top reads",round(np.mean([x[5][h] for x in rows]),3),"gate",round(np.mean([x[4][h] for x in rows]),3))
rel=np.array([x[0] for x in rows]); names=np.array([x[1] for x in rows]); wd=np.array([x[2] for x in rows])
same,par,diff=[],[],[]
for i in rng.integers(len(rows),size=20000):
    j=rng.integers(len(rows)); 
    if i==j: continue
    agree=(code[i]==code[j]).mean()
    if rel[i]==rel[j] and wd[i]==wd[j]: same.append(agree)
    elif rel[i]==rel[j]: par.append(agree)
    else: diff.append(agree)
print("share of heads with the same code: same relation same wording",round(np.mean(same),3),"same relation other wording",round(np.mean(par),3),"other relation",round(np.mean(diff),3))
