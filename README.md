# Free ICML research: LLM memory modules and a test to compare them

> [!IMPORTANT]
> **This is giveaway research for ICML 2027.** It was made for students at an AI research school, and anyone can use it. Take the code, the results or the ideas and publish your own paper. You need no permission and you do not need to add anyone as an author.
>
> A memory module gives an LLM a learned table to look things up in. This repo tests 9 published modules and about 90 new designs on the same text at several sizes. **At 5m, the new design below beats all 9 published modules on web text and on all three fact recall scores, with no more weights than Engram. It loses one score, the relation gap.**

![Architecture of magree: a word table, a gated table inside attention, an n-gram table and MoE](charts/architecture.png)

**`magree`, the best design at 5m.** Left: the whole model. Right: attention in its last block. The words are an example. Coral parts are new in this repo, and the rest follows published modules.

---

## Memory modules add a lookup table to the model

A language model reads a sentence and passes a hidden state from block to block. Whatever it remembers has to be stored in its weights and reached through that state.

A **memory module** adds a learned table. The model finds a row using the current token or the last few tokens, then adds that row to the computation.

Say the model has read "The capital of France is" many times. A table keyed on "France" can learn a row that pushes up "Paris".

Now ask "The currency of France is". A row keyed on "France" still pushes up "Paris". The **relation gap** measures this: how much more likely is the stored answer under its own question than under another relation's question?

---

## Which modules work, and why

### Try `magree` for fact recall, because three tables each do a different job

Tier 5m is a model 96 wide with 4 layers, trained for 1500 steps. Lower is better, except for the relation gap. The bracket says how many standard errors `magree` is from the best published module on that score, and positive means better.

| Score | Engram | Best published module | `magree` |
|---|---|---|---|
| Web loss | 5.408 | 5.386 MoME | **5.365 (+4.7)** |
| Stored facts | .633 | .224 value embeddings | **.098 (+5.0)** |
| Facts in new wording | 2.553 | .962 value embeddings | **.724 (+2.3)** |
| Name far from the answer | 3.324 | 1.784 value embeddings | **1.068 (+8.3)** |
| Relation gap | 5.368 | **5.368 Engram** | 4.082 (-8.4) |
| Usual phrase | 1.595 | 1.595 Engram | 1.567 (+1.0) |

Each part was switched off in a trained model to see what it does. Scores are log-probabilities of the stored answer, and all tables on gives -1.40 under its own question and -5.48 under another relation's question.

1. **The word table gives the most recall.** It has one row per token and sits after block 1. Without it, the right answer falls by 3.21 to -4.61.
2. **The gated table adds recall, and it also leaks.** Each attention head takes the 4 tokens it attended to most and uses its own state to pick 2 of 16 codes. A row is found by token and code. Without this table, the right answer falls by 2.32 and the wrong one falls by 3.42.
3. **The n-gram table protects the relation gap.** It is Engram's table after the last block, with its output starting at zero. Without it, the gap falls from 4.08 to 2.56.
4. **MoE runs in every block.** Its extra weights come out of the table budget, so the total stays at Engram's.

> **The word table and the gated table raise the right answer, and they also raise that answer under the wrong question. Only the n-gram table holds it back.**

The cosine in the gate helps only at the larger size. Against `mtrip`, the same design without the cosine, `magree` gains 3.8 standard errors on new wording at 5m and loses 2.5 at 1m.

### Use a different module for other jobs

| Job | Try | Result, in standard errors from the best published module | Cost |
|---|---|---|---|
| Lowest web loss at 1h | `mmomez`: `magree` with MoME in place of the gated table | Web 4.550 (+14.3), usual phrase (+4.8), relation gap (-14.4) | 43.30M weights |
| Fact recall at 1m | `mtrip`: `magree` without the cosine | Stored facts .294 (+5.2), relation gap (-2.4) | 1.37x Engram's step |
| A lead checked on hidden items | `tmomez`: three tables, no MoE | At 5m: web (+4.2), stored facts (+3.5), relation gap (-7.1) | 4.73M weights |
| Highest relation gap with a table | `mbuck64`: rows keyed by token and a bucket of another token | Gap 5.598 (+1.6), new wording (-6.0) | 4.74M weights |
| A strong result with no new part | `momemoe`: MoME + MoE | Name far (+4.3), new wording (+2.2), relation gap (-13.5) | 1.21x Engram's step |
| Cheap fact recall, published | Value embeddings | Lead all three fact scores at 5m | .93x Engram's step |
| Low web loss, published | MoME | Lowest published web loss at 5m and 1h | 1.06x Engram's step |
| Relation gap and common phrases, published | Engram | Leads both at 5m, and usual phrase at 1h | 1.00x |

More relation gap tends to come with worse new-wording loss, and no run here has both:

![Relation gap against new-wording loss at 5m](charts/tradeoff-5m.png)

### Test at more than one size, because the order changes

A plain table with one row per token beats Engram on web loss at 1m and 5m, and Engram beats it at 1h. `magree` has no web lead at 1m (-.1).

### Read the win rule as a target, because no module passes it

`beat.py` applies a rule that was written before the new designs ran. At both 1m and 5m, a design must have the lowest web loss by 2 standard errors, lose none of the other 5 scores to a published module by 2 standard errors, and use no more weights than Engram.

None of the 9 published modules passes it against the other 8, at 1m, 5m or 1h. They lead on different scores, and the new designs fail it the same way. So a module can be useful and still fail, and [RESULTS.md](RESULTS.md) gives every score for you to judge.

---

## What is new here, and what comes from papers

> **You can claim a new combination and two small changes to known parts. You cannot yet claim that nobody did this first.** Three extended searches found no earlier version, and most close papers have not been read in full.

| Part | Where it comes from | Status here |
|---|---|---|
| MoE | Standard in current models | Used as published |
| Table with a gate, added to the hidden state | Engram, 2601.07372 | Word table and n-gram table reuse its read path |
| Table in attention, router, top 2 slots | MoME, 2609.15126 | `magree` adds its row after attention instead of to the values. Memory Attention 2609.28399 tests both and finds values better |
| MoE with Engram, MoME with Bigram | Lngram 2605.24869, MoME Table 4 | Published pairs of parts |
| **Router code inside the table key, picked by the reading head** | Not found | New here, as far as the searches show |
| **Gate from a cosine between the hidden state and the row** | Close to Engram's gate section 2.3 | Small change: per head, on attention output, cosine |
| **Word table, gated table, n-gram table and MoE, at these depths in one budget** | Not found | New combination |
| **One test with fact, wording, relation and text scores** | Not found. ROME and CounterFact (2202.05262) may be close, from training data and not checked | New test, to be checked |

The ports follow the papers' text, and they have not been compared with the authors' code.

---

## Pick any of these research directions

There is room for many separate papers. [PAPER-IDEA.md](PAPER-IDEA.md) has more directions, a draft abstract and what a reviewer will ask.

1. **Fix the leak.** Can a table recall a fact in new wording and hold it back when the question changes?
2. **Change the key.** `magree` recalls and `mbuck64` protects the gap. Can a key do both?
3. **Change the gate.** What should a gate read so that it closes under the wrong relation?
4. **Move the tables.** Which block should each table sit in, and does that change with size?
5. **Split the budget.** How should weights divide between the tables and MoE as the model grows?
6. **Make it cheap.** Can the gated table keep its gain below 1.40x Engram's step time?
7. **Explain the size flips.** Why does module order change with size, and does tokens per table row predict it?
8. **Test bigger models.** Do scores at small sizes predict the order at 5h and beyond?

---

## Where it stops working

> [!WARNING]
> - **Small models.** Every result comes from models of about 50M weights or fewer. The 5h tier (512 wide, 8 layers) has settings in `env.py` and has never run.
> - **`magree` has no result at 1h yet.** That run is still training.
> - **It costs more compute.** `magree` takes 1.40x Engram's time per step, and a run that gives Engram the same time has not been done.
> - **One training run per setting.** Standard errors cover the test items and do not cover the change between training runs.
> - **Hidden test items are mostly unused.** Only baselines, `tmomez` and `tmixz` have been scored on them. `magree` has not.
> - **Even the plain model beats it on the relation gap.** `magree` scores 4.082 and a model with no memory scores 4.512.
> - **Published gaps are out of reach.** This test can judge large gaps between modules. It can't judge published gaps of about .001, and published benchmarks sit near chance at these sizes.
> - **Fact scores depend on lr.** At 5m their module rankings flip between lr .01 and .02.
> - **Nothing shows that these small models predict large ones.**

An AI coding agent wrote the code, ran the experiments and wrote these documents. They have had no peer review, so check everything before you rely on it.

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
- [PAPER-IDEA.md](PAPER-IDEA.md): research directions, a draft abstract and what is missing.

MIT licence. The downloaded data has its own licences, see [SETUP.md](SETUP.md).
