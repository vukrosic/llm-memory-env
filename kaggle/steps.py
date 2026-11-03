# Memory env on Kaggle: unpack the code, fetch FineWeb, check every method on a tiny run, then train the listed runs, 2 per GPU.
import base64, json, os, shutil, subprocess, sys, time, threading, queue
W = "/kaggle/working"; SRC = W + "/me"; FW = "/tmp/fw"; os.makedirs(SRC, exist_ok=True); os.makedirs(FW, exist_ok=True)
for n, b in FILES.items(): open(os.path.join(SRC, n), "wb").write(base64.b64decode(b))
T0 = time.time(); log = open(W + "/log.txt", "a")
def say(*a):
    s = "[%6.0fs] " % (time.time() - T0) + " ".join(map(str, a)); print(s, flush=True); log.write(s + "\n"); log.flush()
say(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv; free -g; nproc; ls -R /kaggle/input | head -20", shell=True, capture_output=True, text=True).stdout)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"])
inp = subprocess.run("find /kaggle/input -name facts.json", shell=True, capture_output=True, text=True).stdout.split()[0]
shutil.copytree(os.path.dirname(inp), SRC + "/data", dirs_exist_ok=True)
HF = "https://huggingface.co/datasets/kjj0/fineweb100B-gpt2/resolve/main/"
for n in ("fineweb_val_000000.bin", "fineweb_train_000001.bin"): subprocess.run(["curl", "-sSL", "--retry", "5", "-o", os.path.join(FW, n), HF + n])
say("downloaded", os.listdir(FW))
import torch
NG = max(1, torch.cuda.device_count()); say("gpus", NG)
ENV = dict(os.environ, FWDIR=FW, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
def run(tier, m, lr, gpu, extra=()):
    t = time.time(); name = "%s_%s_%g%s" % (tier, m, lr, "".join(extra))
    p = subprocess.run([sys.executable, SRC + "/env.py", "--tier", tier, "--method", m, "--lr", str(lr), *extra], env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)), capture_output=True, text=True, cwd=SRC)
    open(W + "/out_" + name + ".txt", "w").write(p.stdout + "\n--stderr--\n" + p.stderr)
    say(name, "code", p.returncode, "sec %.0f" % (time.time() - t), p.stdout[-200:] if p.returncode == 0 else p.stderr[-1500:])
def pool(jobs, per=2):
    q = queue.Queue(); [q.put(j) for j in jobs]
    def work(g):
        while time.time() - T0 < 10.5 * 3600:
            try: j = q.get_nowait()
            except queue.Empty: return
            run(*j[:3], g, *j[3:])
    ts = [threading.Thread(target=work, args=(g % NG,)) for g in range(NG * per)]; [x.start() for x in ts]; [x.join() for x in ts]
pool([("5s", m, 0.02, ("--steps", "20", "--tag", "_smoke")) for m in sorted({m for m, _ in RUNS})])
pool([(TIER, m, lr, ("--final",)) for m, lr in RUNS])
say("done")
