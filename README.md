# An open test environment for LLM memory modules

> [!IMPORTANT]
> Memory modules such as Engram, MoME and value embeddings give an LLM a learned table to look things up in. Papers usually compare a few of them at a time, on validation loss and some benchmarks. This repo trains 9 baselines on the same text at several sizes and gives each one 74 scores. It also holds about 90 new designs and every failed experiment. **None of the new designs passes the win rule that was written before they ran.** The open problem and the unfinished paper are free for anyone to take: [PAPER-IDEA.md](PAPER-IDEA.md).

![Relation gap against new-wording loss at 5m](charts/tradeoff-5m.png)

---

## Memory modules add a lookup table to the model

A language model reads a sentence and passes a hidden state from block to block. Whatever it remembers has to be stored in its weights and reached through that state.

A **memory module** adds a learned table. The model finds a row using the current token or the last few tokens, then adds that row to the computation.

Say the model has read "The capital of France is" many times. A table keyed on "France" can learn a row that pushes up "Paris".

Now ask "The currency of France is". A row keyed on "France" still pushes up "Paris". The **relation gap** measures this: how much more likely is the stored answer under its own question than under another relation's question?

The question this repo leaves open: **can one table recall a fact in a new wording and also hold it back when the question changes?**

---

## Results

### Read each score separately, because no published module wins all of them

At 5m (1.6M to 7.7M weights, 1500 steps), value embeddings and MoME lead on facts, and Engram leads on the relation gap and common phrases. MoME has the lowest web loss.

### Do not trust the order at small sizes

A plain table with one row per token beats Engram on web loss at 1m and 5m. Engram beats it at 1h. Run this control at more than one size.

### Do not credit a new design until you have added MoE to a published module

Adding MoE to MoME, with no new part, already beats every baseline on new wording and far names at 5m. So part of the gain in the designs here comes from a known combination.

### No design passes: the best one recalls well and leaks

`magree` is a table inside attention whose output is gated by how well the row fits the reader's state. At 5m it beats every baseline on web loss and three fact scores, and it loses the relation gap to Engram by 8.4 standard errors. Its table raises the wrong answer more than the right one. At 1m it has no lead on web loss.

`mbuck64` comes closest on the relation gap, and it loses new wording to value embeddings by 6.0 standard errors.

> **Engram protects the relation gap and recalls poorly in new wording. The tables that recall well leak. No run here matches the best baseline on both.**

---

## Where it stops working

> [!WARNING]
> - **Small models.** Every result comes from models of about 50M weights or fewer. The 5h tier (512 wide, 8 layers) has settings in `env.py` and has never run.
> - **One training run per setting.** Standard errors cover the test items and do not cover the change between training runs.
> - **Hidden test items are mostly unused.** Only baselines, `tmomez` and `tmixz` have been scored on them. `magree` has not.
> - **Published gaps are out of reach.** This test can judge large gaps between modules. It can't judge published gaps of about .001 between good modules, and published benchmarks sit near chance at these sizes.
> - **Fact scores depend on lr.** At 5m their module rankings flip between lr .01 and .02.
> - **Nothing shows that these small models predict large ones.**

An AI coding agent wrote the code, ran the experiments and wrote these documents. They have had no peer review. The ports follow the papers' text and have not been compared with the authors' code.

---

## Is this new?

Three extended searches found no shared test with these scores. That does not settle it, and [PAPER-IDEA.md](PAPER-IDEA.md) lists what to read next. The `magree` gate is close to Engram's published gate, and MoE + Engram is published in Lngram (2605.24869). This repo makes no claim of a new module.

---

## Setup

```bash
pip install -r requirements.txt
```

```bash
python3 setup_data.py
```

```bash
python3 env.py --method engram --tier 5s --lr 0.02
```

The download is 392 MB and takes a few minutes. The last command trains in about a minute on one CPU core and prints its scores. To read the existing results, you need neither the data nor training: `python3 tables.py 5m`.

## More

- [SETUP.md](SETUP.md): install, tiers, free Kaggle GPU, adding your own module.
- [RESULTS.md](RESULTS.md): every score, the win rule, controls and speed.
- [EXPERIMENTS.md](EXPERIMENTS.md): the full log with each prediction and result, small debugging experiments, and the bugs found.
- [PAPER-IDEA.md](PAPER-IDEA.md): the unfinished paper and what is missing.

MIT licence. The downloaded data has its own licences, see [SETUP.md](SETUP.md).
