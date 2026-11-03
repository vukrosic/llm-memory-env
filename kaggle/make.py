"""Packs one Kaggle job: python3 kaggle/make.py NAME method:lr method:lr ... [--tier 1h] writes kaggle/NAME/job.py and kernel-metadata.json. Push with kaggle kernels push -p kaggle/NAME.
python3 kaggle/make.py --data copies data/ into kaggle/data-pack/ for one upload: kaggle datasets create -p kaggle/data-pack. Your Kaggle name comes from KAGGLE_USERNAME or ~/.kaggle/kaggle.json."""
import base64, json, os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(HERE, ".."); args = sys.argv[1:]
cfg = os.path.expanduser("~/.kaggle/kaggle.json")
USER = os.environ.get("KAGGLE_USERNAME") or (json.load(open(cfg))["username"] if os.path.exists(cfg) else None)
if not USER: raise SystemExit("set KAGGLE_USERNAME or run kaggle's API token setup first")
if args[0] == "--data":
    pack = os.path.join(HERE, "data-pack"); os.makedirs(pack, exist_ok=True)
    for n in ("facts.json", "counts.npy"): shutil.copy(os.path.join(SRC, "data", n), pack)
    shutil.copytree(os.path.join(SRC, "data", "bench"), os.path.join(pack, "bench"), dirs_exist_ok=True)
    json.dump({"title": "memory-env-data", "id": f"{USER}/memory-env-data", "licenses": [{"name": "other"}]}, open(os.path.join(pack, "dataset-metadata.json"), "w")); print("wrote", pack); sys.exit()
tier = args[args.index("--tier") + 1] if "--tier" in args else "1h"; args = [a for i, a in enumerate(args) if a != "--tier" and (i == 0 or args[i - 1] != "--tier")]
name = args[0]; job = os.path.join(HERE, name); os.makedirs(job, exist_ok=True)
F = {n: base64.b64encode(open(os.path.join(SRC, n), "rb").read()).decode() for n in ("env.py", "evals.py", "methods.py")}
R = [(m, float(l)) for m, l in (x.split(":") for x in args[1:])]
open(os.path.join(job, "job.py"), "w").write("FILES = " + repr(F) + "\nRUNS = " + repr(R) + "\nTIER = " + repr(tier) + "\n" + open(os.path.join(HERE, "steps.py")).read())
json.dump({"id": f"{USER}/memory-env-{name}", "title": f"memory-env-{name}", "code_file": "job.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [f"{USER}/memory-env-data"], "competition_sources": [], "kernel_sources": [], "model_sources": []}, open(os.path.join(job, "kernel-metadata.json"), "w"))
print("wrote", job, len(R), "runs at tier", tier)
