"""Fetch one finished Kaggle job and add its scores here: python3 kaggle/merge.py NAME TIER. Each download try stops after 240 s, because a download can hang for hours.
Only rows of that tier with an empty tag and not yet present are added, and the per-item score files come with them."""
import glob, json, os, shutil, subprocess, sys, time
name, tier = sys.argv[1], sys.argv[2]; HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); out = os.path.join(HERE, name, "output"); os.makedirs(out, exist_ok=True)
cfg = os.path.expanduser("~/.kaggle/kaggle.json"); USER = os.environ.get("KAGGLE_USERNAME") or json.load(open(cfg))["username"]
for i in range(6):
    try:
        r = subprocess.run(["kaggle", "kernels", "output", f"{USER}/memory-env-{name}", "-p", out], capture_output=True, text=True, timeout=240)
        print("try", i, "code", r.returncode, flush=True)
        if r.returncode == 0 and os.path.exists(os.path.join(out, "me", "results.jsonl")): break
    except subprocess.TimeoutExpired:
        print("try", i, "timed out", flush=True)
    time.sleep(8)
have = {(r["method"], r["tier"], r["lr"], r["tag"]) for r in map(json.loads, open(os.path.join(ROOT, "results.jsonl")))}; added = 0
with open(os.path.join(ROOT, "results.jsonl"), "a") as f:
    for l in open(os.path.join(out, "me", "results.jsonl")):
        r = json.loads(l); k = (r["method"], r["tier"], r["lr"], r["tag"])
        if r["tier"] == tier and not r["tag"] and k not in have: f.write(l); added += 1; print("add", r["method"], r["lr"]); have.add(k)
if os.path.exists(os.path.join(out, "me", "final_results.jsonl")):
    with open(os.path.join(ROOT, "final_results.jsonl"), "a") as f: f.write(open(os.path.join(out, "me", "final_results.jsonl")).read())
for p in glob.glob(os.path.join(out, "me", "runs", "*.npz")):
    if os.path.basename(p).split("_")[0] == tier or os.path.basename(p).startswith(f"final_{tier}_"): shutil.copy(p, os.path.join(ROOT, "runs"))
print(added, "rows added")
