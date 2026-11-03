# Setup

You need Python 3.10 or newer, about 500 MB of disk, and `curl`. A CPU is enough up to tier 5m. Tier 1h needs a GPU.

## Read the results without training

`results.jsonl` and `runs/` are in the repo, and these commands need only `numpy`:

```bash
python3 tables.py 5m
```

```bash
python3 beat.py 5m magree
```

```bash
python3 compare.py 5m engram
```

## Download the data

```bash
pip install -r requirements.txt
```

```bash
python3 setup_data.py
```

This fetches 382 MB of web text, 4 MB of facts and 6 MB of benchmark portions into `data/`, and it took 6 minutes in a test from a clean copy. `facts.json` and `counts.npy` are built the first time `env.py` runs.

| Data | Source | Licence, check before reuse |
|---|---|---|
| Web text | FineWeb, tokenized with GPT-2, [kjj0/fineweb100B-gpt2](https://huggingface.co/datasets/kjj0/fineweb100B-gpt2) | Listed as MIT |
| Facts | [ParaRel](https://github.com/yanaiela/pararel), Elazar et al. 2021 | MIT |
| Benchmarks | BLiMP, HellaSwag, WinoGrande, ARC, MMLU and 4 text kinds, through `fetch.py` | Each has its own |

The web text file can live elsewhere if you set `FWDIR`.

## Train one setting

```bash
python3 env.py --method engram --tier 5s --lr 0.02
```

This appends one row to `results.jsonl` and writes one per-item score file to `runs/`.

| Flag | Meaning |
|---|---|
| `--method` | Any name in `METHODS` in [methods.py](methods.py). |
| `--tier` | Model size and steps, see below. |
| `--lr` | Learning rate. Run .01 and .02 for every method you compare. |
| `--dev` | Device, for example `cuda` or `mps`. |
| `--threads` | CPU threads. |
| `--steps` | Stop early. A row with changed steps is a test, so give it a `--tag`. |
| `--tag` | Rows with a tag are left out of the tables. |
| `--keep` | Save the trained model, which the scripts in `debug/` need. |
| `--max-seconds` | Save and stop after this time. The same command continues the run. |
| `--final` | Score the hidden items. Read the warning below. |

## Pick a tier

| Tier | Width | Layers | Vocabulary | Steps | Approximate training time |
|---|---|---|---|---|---|
| `5s` | 48 | 2 | 4,000 | 250 | 5 s on one CPU core, plus evaluation |
| `1m` | 64 | 4 | 8,000 | 600 | 2 min on one CPU core |
| `1mv` | 64 | 4 | 1,000 | 600 | 2 min on one CPU core, no fact scores |
| `5m` | 96 | 4 | 12,000 | 1500 | 5 min on one CPU core |
| `1h` | 256 | 6 | 20,000 | 10,000 | 40 to 90 min on a Kaggle T4, sharing it with another run |
| `5h` | 512 | 8 | 50,257 | 60,000 | Never run |

Designs with MoE take its weights from the table budget and need tier `1m` or larger, because at `5s` no table budget is left.

`5h` will stop with "add more shards". It reads 983M tokens and `World` reads one file of about 95M, so it needs a loader for about 11 files. A rough guess is 13 hours or more on a T4, which means two Kaggle jobs that resume with `--max-seconds`.

## Add your own module

Write a module, then add one entry to `METHODS` in [methods.py](methods.py). The docstring at the top of that file gives the interface.

```python
"mine": lambda c, W, dev: dict(mems={1: MyTable(c, W, dev)}),
```

Tables must be sparse `nn.Embedding` under `.tab`, and projections must be named `.w*`, so the optimizers find them. Then:

```bash
python3 check.py 5s
```

This asks whether a module can see future tokens, whether every weight gets a gradient, and whether the table changes the output. Use `check.py 1m` for designs with MoE.

```bash
python3 env.py --method mine --tier 5m --lr 0.01
```

```bash
python3 env.py --method mine --tier 5m --lr 0.02
```

```bash
python3 beat.py 5m mine
```

```bash
python3 compare.py 5m engram mine
```

`beat.py` applies the win rule, which needs both 1m and 5m. `compare.py` prints the scores in `judges.json`. `tables.py` prints the six headline scores.

> [!WARNING]
> **Use `--final` once per design, at the very end.** It scores the items no design choice should see, and it writes to `final_results.jsonl`. If you pick designs by these scores, the test stops being a test.

> [!NOTE]
> `audit.py` overwrites `judges.json` with the scores that pass its checks on your current `results.jsonl`.

## Run on a free Kaggle GPU

These jobs are private, and they use a T4. Kaggle allows 12 hours per job and, at the time of these runs, 2 batch GPU sessions at once. Set up the [Kaggle API token](https://www.kaggle.com/docs/api) first.

Upload the benchmark and fact files once. The job downloads the web text itself:

```bash
pip install kaggle
```

```bash
python3 kaggle/make.py --data
```

```bash
kaggle datasets create -p kaggle/data-pack
```

Make a job and start it. Each `method:lr` is one run, and 2 runs share each GPU:

```bash
python3 kaggle/make.py myjob engram:0.01 mome:0.01 --tier 1h
```

```bash
kaggle kernels push -p kaggle/myjob
```

Check on it and add its scores when it finishes:

```bash
kaggle kernels status YOUR_KAGGLE_NAME/memory-env-myjob
```

```bash
python3 kaggle/merge.py myjob 1h
```

`merge.py` adds rows of that tier with an empty tag that are not present yet, and it brings the per-item score files. Each download try stops after 240 s, because one download here hung for hours.

## Run the debugging experiments

Train with `--keep` first, then run a script from `debug/`.

```bash
python3 env.py --method magree --tier 5m --lr 0.01 --keep
```

```bash
python3 debug/ablate.py 5m magree 0.01
```

```bash
python3 debug/speed.py 5m engram magree
```

[EXPERIMENTS.md](EXPERIMENTS.md) lists what each script asks and found.

## Check that your setup matches

A clean copy of this repo gave the recorded scores exactly for `plain` at 5s, `engram` at 5s and `magree` at 1m on a CPU. Other machines and GPUs can give slightly different numbers.

```bash
python3 env.py --method plain --tier 5s --lr 0.02 --tag mycheck
```

Compare the web loss with the `plain` row at lr .02 in `results.jsonl`.
