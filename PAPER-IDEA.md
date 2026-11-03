# A paper you can take: testing LLM memory modules side by side

This is an unfinished paper with its code and results. Anyone can take it, change it and submit it. The code is MIT licensed, no permission is needed, and you do not need to add anyone as an author. Citing this repo is welcome if it helped.

> [!IMPORTANT]
> **It is not ready to submit.** The strongest claim that the data supports today is about the test and not about a new module. No new design passes the win rule, the largest tier has never run, and nothing yet shows the small models predict large ones.

An AI coding agent did this work and wrote this page. No person or reviewer has checked it, so check everything before you rely on it.

---

## Choose the shape of the paper

**Shape 1: a test environment and what it shows.** This one is closest. It would claim that memory modules trade recall against question sensitivity, that module order changes with size, and that a shared multi-score test finds this. It needs the size work below.

**Shape 2: a new module that closes the gap.** This one is open research. It needs a table that recalls a fact in new wording and holds it back under another relation. No run here does. Shape 1 can be written first and gives Shape 2 its test.

---

## Draft abstract for Shape 1

*Bracketed parts are not done. Replace or remove them.*

> Memory modules add a learned lookup table to a language model. Published comparisons usually cover a few modules on validation loss and benchmarks, which leaves open what each module is good at. We train 9 modules on the same fresh web text with planted facts, at [5] sizes from 0.25M to [more than 100M] weights, and score each on recall, recall in new wording, sensitivity to the question, and text prediction. No module leads on all scores. Modules that recall facts in new wording best lose the most when the question changes: [the relation gap falls from 5.37 to 3.57 at 5m]. A single-token table with the same weights beats Engram on web text at small sizes and loses at [1h], so module order does not carry across sizes. [Scores at the small sizes predict the order at the largest size with rank correlation X.] We release the test, every run and a win rule for new designs.

---

## What is done

| Part | State |
|---|---|
| Test environment at 5 tiers, 74 scores | Done, reproduced from a clean copy |
| 9 baselines up to 1h, at 2 or more lrs (`moe` at 5m at one) | Done |
| Controls: broken table, memorizer, published module + MoE | Done at 1m and 5m, MoE controls still running at 1h |
| Audit of which scores can judge | Done: 12 of 74 pass on the fixed test |
| Check against published rankings | Done: about half hold, small gaps out of reach |
| Win rule, written before new designs ran | Done, nobody passes |
| About 90 new designs, with predictions written first | Done, see [EXPERIMENTS.md](EXPERIMENTS.md) |
| Why tables lose the relation gap, switch-off tests | Done at 5m |
| Time per step | Done at 5m on one CPU thread |
| Hidden items | Baselines, `tmomez` and `tmixz` only |

---

## What is missing, in the order to do it

1. **Run the 5h tier.** It needs a loader for about 11 web text files, since `World` reads one. A rough guess is 13 hours or more on a T4, so two Kaggle jobs with `--max-seconds`.
2. **Show that the test predicts bigger models.** This is the biggest risk. HellaSwag, ARC, MMLU and WinoGrande sit near chance through 1h, and BLiMP is the only benchmark that moves clearly. Natural facts first move at 1h. Without evidence here, a reviewer can dismiss all of it as toy results.
3. **Train each setting more than once.** Each setting has one training run, so standard errors cover items only. Some z scores here will shrink.
4. **Score the remaining designs on hidden items.** `magree`, `mexp8`, `mapair`, `mbuck64` and the controls have never used them. Do this once, at the end. The hidden scores of `tmomez` and `tmixz` were read before `magree` was built, so consider holding out new items.
5. **Fix the fact scores.** At 5m their module rankings flip between lr .01 and .02. They fail the audit, and three of them are in the win rule. Try more lrs, or find out why they flip.
6. **Tune all designs equally.** 43 of 70 methods at 5m ran at one lr, and baselines ran at two.
7. **Add a compute cap to the win rule.** `magree` costs 1.40x Engram's step and `mexp8` 2.25x. Only weights are capped today.
8. **Compare the ports with the authors' code.** They follow the papers' text, and the docstrings in `methods.py` say what was adapted. Any error here changes the results for that module.
9. **Finish the literature check**, see below.
10. **Solve the open problem**, for Shape 2: a table whose boost depends on the question.

---

## Expect these objections from a reviewer

- **"These are toy models."** Yes, about 50M weights at most so far. Item 2 is the answer, and it may come out against the test.
- **"Planted facts are not real knowledge."** Natural facts do not rank modules at 1m or 5m. Planted-fact scores could measure only memorization of the planted text.
- **"Your small tiers disagree with published rankings."** They do. About half of published rankings hold, and published gaps of about .001 can't be seen here. Say this openly and use tokens per table row, which explains one mismatch.
- **"The win rule is arbitrary."** It was fixed before the designs ran, and [RESULTS.md](RESULTS.md) shows all scores, so readers can apply another rule.

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
| ROME and CounterFact, 2202.05262 | From memory, not checked here: it scores paraphrase generalization and neighbourhood specificity, which resembles new wording and relation gap. Check this first, because it may already cover the idea. | Not read |

---

## Dates

ICML has not published its 2027 dates at the time of writing. Estimates from past years are an abstract around January 16 and a full paper around January 22, 2027. Check the [official site](https://icml.cc/).

Any other venue works too. A workshop or a preprint needs less of the list above.
