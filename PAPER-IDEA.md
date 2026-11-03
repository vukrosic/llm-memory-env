# Free research directions for ICML 2027: LLM memory modules

This is giveaway research. It was made for students at [AI Research School](https://www.skool.com/become-ai-researcher-2669), and anyone can use it. The code is MIT licensed, no permission is needed, and you do not need to add anyone as an author. Citing this repo is welcome if it helped.

> [!IMPORTANT]
> **There is room for many papers here, so pick the one you like.** The strongest result so far is `mmomez` at 1h, about 43M weights: it has the lowest web loss, ties the best published modules on fact recall, and loses the relation gap. The largest tier has never run, and nothing yet shows the small models predict large ones. Check all of it before you submit.

An AI coding agent did this work and wrote this page. No person or reviewer has checked it.

---

## Choose a direction

### Papers about a new module

1. **Fix the leak.** Can a table recall a fact in new wording and hold it back when the question changes? No run here does both. `magree` has new wording .724 and gap 4.082, and Engram has 2.553 and 5.368.
2. **Change the key.** `magree` keys rows by token and code. `mbuck64` keys them by token and a bucket of another token, and it has the only gap above Engram's. Can one key do both jobs?
3. **Change the gate.** The cosine gate helped new wording at 5m and hurt at 1m, and it did not close under the wrong relation. What should a gate read instead? `mexp8` teaches it with a next-word head during training, at 2.25x the cost.
4. **Put the row in the values.** `magree` adds its row after attention, and Memory Attention 2609.28399 finds values better. `mmomez`, with MoME in the values, has the lowest web loss at 1h.
5. **Move the tables.** Which block should each table sit in? The n-gram table protects the gap from the last block, and Qwen3.8-Next found no clearly best depth for one n-gram layer.
6. **Split the budget.** How should weights divide between tables and MoE as the model grows? MoE alone has the best relation gap at 1h, and adding MoE changed designs differently at 5m and 1h.
7. **Share or drop tables.** Is one table enough once MoE is present? `momemoe` reaches part of the gain with no new parts.
8. **Make it cheap.** `magree` costs 1.40x Engram's step, and each head reads 8 rows per position (4 tokens x 2 codes). Can it keep the gain with fewer reads?

### Papers about the test

9. **Publish the test itself.** Claim that memory modules trade recall against question sensitivity, and that a shared multi-score test finds this. The draft abstract below is for this one.
10. **Explain why the order changes with size.** A single-token table beats Engram on web loss at 1m and 5m, and loses at 1h. Tokens per table row explains one mismatch with published rankings.
11. **Show whether small models predict large ones.** Run the 5h tier and measure the rank correlation. This could also come out against small tests.
12. **Build a better fact score.** At 5m the fact rankings flip between lr .01 and .02. Find out why, or find a score that does not flip.
13. **Use natural facts.** Natural facts first move at 1h here. Find the size and training length where they rank modules.

### Papers about training

14. **Tune tables separately.** Do tables want their own lr or schedule, and does that remove the flip in fact scores?
15. **Give each module equal compute.** Compare modules at equal time per step instead of equal weights.

---

## Draft abstract for the test paper

*Bracketed parts are not done. Replace or remove them.*

> Memory modules add a learned lookup table to a language model. Published comparisons usually cover a few modules on validation loss and benchmarks, which leaves open what each module is good at. We train 9 modules on the same fresh web text with planted facts, at [5] sizes from 0.25M to [more than 100M] weights, and score each on recall, recall in new wording, sensitivity to the question, and text prediction. No module leads on all scores. Modules that recall facts in new wording best lose the most when the question changes: [the relation gap falls from 5.37 to 3.57 at 5m]. A single-token table with the same weights beats Engram on web text at small sizes and loses at [1h], so module order does not carry across sizes. [Scores at the small sizes predict the order at the largest size with rank correlation X.] We release the test and every run.

## Draft abstract for a module paper

> We add three lookup tables to a transformer with mixture-of-experts layers, at different depths and in one weight budget: a word table after the first block, a gated table inside attention in the last block, and an n-gram table after the last block. At equal weights this beats [9] published modules on text prediction and fact recall at [sizes]. Switching each table off at test time shows that the n-gram table protects question sensitivity, the word table adds the most recall, and the gated table adds recall and lowers question sensitivity. [A new key or gate closes this gap by X.]

---

## What is done

| Part | State |
|---|---|
| Test environment at 5 tiers, 74 scores | Done, reproduced from a clean copy |
| 9 baselines up to 1h, at 2 or more lrs (`moe` at 5m at one) | Done |
| Controls: broken table, memorizer, published module + MoE | Done at 1m and 5m, MoE controls still running at 1h |
| Audit of which scores can judge | Done: 12 of 74 pass on the fixed test |
| Check against published rankings | Done: about half hold, small gaps out of reach |
| Win rule, written before new designs ran | Done, and no published or new module passes it |
| About 90 new designs, with predictions written first | Done, see [EXPERIMENTS.md](EXPERIMENTS.md) |
| Why tables lose the relation gap, switch-off tests | Done at 5m |
| Time per step | Done at 5m on one CPU thread |
| Hidden items | Baselines, `tmomez` and `tmixz` only |

---

## Do these before you submit any of them

- **Train each setting more than once.** Each setting has one training run, so standard errors cover items only. Some z scores here will shrink.
- **Score your final design on hidden items, once, at the end.** `magree`, `mexp8`, `mapair`, `mbuck64` and the controls have never used them. The hidden scores of `tmomez` and `tmixz` were read before `magree` was built, so consider holding out new items.
- **Tune all designs equally.** 43 of 70 methods at 5m ran at one lr, and baselines ran at two.
- **Compare the ports with the authors' code.** They follow the papers' text, and the docstrings in `methods.py` say what was adapted. Any error here changes the results for that module.
- **Finish the literature check**, see below.
- **Run a larger tier if your claim needs it.** The 5h tier needs a loader for about 11 web text files, since `World` reads one. A rough guess is 13 hours or more on a T4, so two Kaggle jobs with `--max-seconds`.

---

## Expect these objections from a reviewer

- **"These are toy models."** Yes, about 50M weights at most so far. Direction 11 is the answer, and it may come out against the test. HellaSwag, ARC, MMLU and WinoGrande sit near chance through 1h, and BLiMP is the only benchmark that moves clearly.
- **"Planted facts are not real knowledge."** Natural facts do not rank modules at 1m or 5m. Planted-fact scores could measure only memorization of the planted text.
- **"Your small tiers disagree with published rankings."** They do. About half of published rankings hold, and published gaps of about .001 can't be seen here. Say this openly and use tokens per table row, which explains one mismatch.
- **"You used more compute."** `magree` costs 1.40x Engram's step and `mexp8` 2.25x, and the win rule caps only weights. Direction 15 answers this.

---

## Read this work first

Three extended searches found no shared multi-score test. Those searches do not settle it, and only two papers below were read closely.

| Work | Why it matters here | Read? |
|---|---|---|
| Engram, 2601.07372 | Baseline. Its gate compares the hidden state with the retrieved row, close to `magree`'s gate. | Read |
| Lngram, 2605.24869 | Compares MoE, MoE + Engram and MoE + Lngram at 2B, with gains on MMLU, SciQ and PIQA. So MoE + Engram is published. | Read |
| Lngram v2, 2609.03426 | Scales to a 30B model with parameter-matched baselines. | Not read in full |
| MoME, 2609.15126 | Baseline. Its tables compare MoME, VE, Bigram, STEM and Engram on validation loss, and Table 4 scales MoME with Bigram. | Tables used in `claims.py` and the port |
| STEM, 2601.10639 | Baseline. | Port follows its text |
| TN-gram, 2606.08347 | Baseline. | Port follows its text |
| Memory Attention, 2609.28399 | Finds memory in attention values better than in attention outputs. | Not read in full |
| Qwen3.8-Next, 2608.30320 | One n-gram layer tested at layers 1 to 25, no depth clearly best. | Not read in full |
| Memory Is Not Always Needed, 2608.23982 | Router against static or random routing on scientific reasoning. No named memory comparison found. | Not read in full |
| NGM 2605.16893, TF-Engram 2607.07388, collision-free hot tier 2601.16531, BadEngram 2609.13478, ARM 2609.24417, MoM 2502.13685 | Other close modules and attacks. | Not read in full |
| ROME and CounterFact, 2202.05262 | From training data, not checked here: it scores paraphrase generalization and neighbourhood specificity, which resembles new wording and relation gap. Check this first, because it may already cover the test. | Not read |

---

## Dates

ICML has not published its 2027 dates at the time of writing. Estimates from past years are an abstract around January 16 and a full paper around January 22, 2027. Check the [official site](https://icml.cc/).

Any other venue works too. A workshop or a preprint needs less of the list above.
