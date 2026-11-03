# Experiment log

This is the full working log, kept as it was written, with small edits so that file paths work here. It holds every round, the prediction written before each round, and the result, including the predictions that failed. An AI coding agent wrote it while running the experiments, so "I" in it means the agent.

Cleaner tables are in [RESULTS.md](RESULTS.md). Read this file to see how the work went and which ideas did not work.

- Fact scores from before the two test fixes (see [the bug list](#bugs-found-and-fixed)) do not compare with later ones.
- Some sections report facts as accuracy (higher is better) and later ones as loss (lower is better). Check the row labels.
- Each setting has one training run. Standard errors in this log compare items, so they do not include the change between two training runs.

## Method names

There are 108 entries in `METHODS` in [methods.py](methods.py). The names were working labels and follow a pattern. "Three tables" means a word table after block 1, a third table in the last memory layer, and an Engram after the last block, sharing Engram's weight budget.

| Name | What it is |
|---|---|
| `plain` | No memory module. |
| `token` | One table row per token, added to the hidden state, with Engram's table weights. It is the control for n-grams. |
| `engram`, `moe`, `ve`, `mome`, `stem`, `bigram`, `tngram` | The published modules. [README.md](README.md) links the papers. |
| `bigram6` | Bigram at the wrong size, kept to show the size bug. |
| `broken`, `cheat` | Test controls. `broken` reads a random table row each time, and `cheat` memorizes text with 16x the rows. |
| `engmoe`, `vemoe`, `momemoe` | Fairness controls: a published module plus MoE, nothing else. |
| `reader`, `vgram`, `mix`, `mix3`, `mixbig` | Value tables whose slots are chosen by the reading position or by hashed bigrams. |
| `both`, `bothlate`, `bothve`, `bothread` | A value table plus a quarter-size Engram. |
| `gated`, `qgram`, `valgram`, `rslot`, `dropve`, `lateve` | Round 1, one idea each. |
| `late`, `lve`, `lmome`, `lreader`, `lmix`, `ltoken` | Round 2: `l` puts Engram after the last block. |
| `t...` | Three tables. The rest of the name says what sits in the last memory layer: `tmome` MoME, `tmix` token and bigram slots, `ttrip` Triple, `tpair` attended-pair keys, `thash` fixed hashed codes. A `z` ending starts Engram's output at zero, a `d` ending drops it at half the positions, and an `o` ending places it after the final norm. |
| `m...` | The same with MoE in every block, for example `mtrip`, `mmomez`, `mpair`. |
| `magree...` | `mtrip` whose gate also reads whether the reader's state expects what the table proposes. `magree` is the design described in [RESULTS.md](RESULTS.md). |
| `mexp...` | `magree` with a training-only next-word head that teaches the expectation. Numbers set the gate's sharpness; `a` and `b` endings change the head's weight. |
| `mapair...`, `mexpair`, `apair` | The same gate on rows keyed by an exact pair of tokens. |
| `mbuck16`, `mbuck64`, `mbuck256` | Rows keyed by a token and a bucket of another token, with buckets from fixed random hyperplanes. |
| `mtok`, `mgram` | `mtrip` with the n-gram table or the word table removed. |
| `mtrip1`, `mmomez1`, `mmixz1`, `m2trip` | `mtrip` and relatives with the third table in the first memory layer, or in every memory layer. |
| `nmtrip2`, `nmtrip1`, `nmtrip5`, `nplain` | A training penalty for giving a rare answer after a swapped phrase, at weight .2, 1 and 5. |
| `ehash4`, `epair`, `emome`, `spair`, `shash4` | `e`: Engram after block 1. `s`: Engram after block 2. |

## Bugs found and fixed

Each of these changed scores or would have.

1. **Fact questions lacked the start token.** Training sentences start with the end-of-text token and fact questions did not. Fact scores changed and web scores did not.
2. **Some answers sat in the question.** Items such as "art school works in the area of" -> "art" were scored. They were removed.
3. **Bigram was too large.** It copied 6V rows without the 6 layers that rule assumes, so it had 8.5M weights and not 3.9M. Those runs stay as `bigram6`.
4. **Natural facts ran out at 1h.** The pool of unplanted facts was used up, so `env.py` now reserves 400.
5. **Speeds from parallel runs are wrong.** `train_s` in `results.jsonl` depends on what else was running, up to 4x. Use `debug/speed.py`.
6. **A list of judging scores went out of date.** The list of 31 was made before the fact test fix. `audit.py` on the fixed test gives 12, because fact rankings at 5m flip with lr.
7. **Tables cannot fit at small tiers.** Designs with MoE take its weights from the table budget, and at 5s nothing is left. `check.py` now prints a skip row for them.

## Small debugging experiments

These scripts answer one question each about a trained model. Train with `--keep` first, so the model is saved.

| Script | Question | What it found |
|---|---|---|
| [debug/ablate.py](debug/ablate.py) | Which table raises the wrong answer? Switch each one off at test time. | The third table leaks, against the prediction that the word table would. See rounds 13 and 20. |
| [debug/diag.py](debug/diag.py) | Do learned codes separate relations? Does attention read the name? | `ttrip` at 1m uses 2 of 16 codes per head. Two relations share a code on 80% of heads, and 65-72% of top reads go to the name. |
| [debug/twonames.py](debug/twonames.py) | How many test names have a second stored relation? | 17 of 442 items, standard error about .7, so this test can't tell. |
| [debug/speed.py](debug/speed.py) | What does one training step cost, alone on one thread? | `magree` costs 1.40x Engram's step and `mexp8` 2.25x. The win rule caps only weights. |
| [debug/ckbench.py](debug/ckbench.py) | Do benchmark scores move partway through a 1h run? | Used on a `plain` checkpoint from an early 1h run: BLiMP moves, and most others stay at chance. |
| [check.py](check.py) | Can a module see future tokens? Does each weight get a gradient? Does the table matter? Do test rows appear in training? | No future leaks. No test rows in training. |

[experiments/job-lists/](experiments/job-lists/) holds the job list of each round, in the order and with the settings they ran. [experiments/tables/](experiments/tables/) holds printed tables from along the way, and [experiments/older-results/](experiments/older-results/) holds rows from before each test fix.

---

## Can this test be trusted (2026-11-02)

**It can judge large gaps between modules on web text, frequent phrases, and facts written 32 times. It cannot judge the published gaps between good modules, published benchmarks, or real-world knowledge at these sizes.**

**Published rankings (`claims.py`, web loss, 5m).** All six modules beat plain in direction, and five reach |z| ≥ 2 (Engram -.026, Bigram -.024, VE -.041, MoME -.048, STEM -.033). TN-gram doesn't (-.006, z -1.5). MoME ties VE and beats STEM and Bigram as published. Published gaps between good modules are .0012-.0015 of .86, too small to expect here. Raw counts (8 of 15 at 5m) are not a percent of truth, because some claims contradict each other across backbones.

Mismatches and what explains them:

| Mismatch at 5m | Check | Result |
|---|---|---|
| Bigram trails VE (+.017, z 3.4), published tie | Tier `1mv`: 8x tokens per table row, V=1000, 600 steps | Tie, +.003 (z .6). Tokens per row probably explain it. 256 here, about 100,000 published. |
| TN-gram loses to Engram (+.020, z 4.5), published tie | `1mv` with both tables at 320,000 rows, 5.9M and 5.4M weights | TN-gram wins, -.017 (z -2.85). At 400k weights it loses by +.030. It needs table size, so the port is not obviously broken. |
| Bigram ties Engram (+.002), published Bigram wins at 1.4B | none | Not explained. |

**Controls (1m, lr .01).** `broken` is MoME reading a random table row each time. `cheat` is Engram with 16x the rows at orders 4-5, built to memorize the training text. Both get no credit:

| Module | Weights | Web loss | Stored | New wording | Relation gap | Usual phrase |
|---|---|---|---|---|---|---|
| Plain | .71M | 5.530 | .216 | .020 | 2.639 | 1.910 |
| Broken | 1.74M | 5.536 | .216 | .059 | 2.527 | 1.843 |
| Cheat | 17.5M | 5.538 | .529 | .118 | 2.814 | 1.862 |
| Engram | 1.77M | 5.523 | .392 | .078 | 3.467 | 1.775 |
| MoME | 1.74M | 5.471 | .765 | .333 | 2.948 | 1.731 |

Fact items n=51 here, standard error about .05. Cheat raises stored facts with 10x MoME's weights, while web loss gets no better than plain and new wording stays far below MoME, so a score on stored facts must never be read without new wording and web loss.

**Scores (`audit.py`).** Later correction: this count is from before the fact test was fixed. On the fixed test 12 pass, see [RESULTS.md](RESULTS.md). 31 of 74 pass at both 5m and 1m, using published modules only. Cut-offs for moved, spread, lr rho and tier rho were set before reading the table. The hidden rho cut-off of .4 was set after reading it, and it removes two scores at 1m (new-wording pick with name near .10, name far .17).

- Can judge: web loss and accuracy, frequent and mid-frequency tokens, phrase rarity, usual continuation, stored and new-wording facts at 32 copies, answer loss at 8 copies, relation and name gaps, unknown-entity entropy, copy-text gain, memorizing training rows, fiction, forgetting.
- Cannot judge at these sizes: all 5 benchmarks, natural facts, tracking, typos and lower case, facts at 2 copies, exact answers at 8 copies, page gain (lr rho .12 at 1m), memory-drop, recall gains, math, code, German, rare tokens.
- Natural facts don't move, so there is no evidence yet that planted-fact scores reflect real knowledge.

**Hidden items.** Scored once at 1m for the 8 published modules. Ranks match public items: web 1.00, stored 32x .83, new wording 32x .95, relation gap .83. The 5m hidden run was dropped at this point, because 10 parallel jobs were too slow on one laptop (about 40 steps per 400 s); it ran later, see "Hidden items, 1m and 5m". Our own designs had not touched hidden items at this point.

**Open for bigger compute:** benchmarks and natural facts moving, tokens per row near published, module rankings at larger scale.

## Every published module side by side (2026-11-02)

Tier `1m` (width 64, 4 layers, 600 steps, about 2 min alone) picks the same winner as 5m on 23 of 64 scores against 6.4 by chance, rank agreement .507 (counted before the Bigram fix). 5s against 5m was .218 and 18 of 55 against about 14, so iterate at 1m and check at 5m.

5m, best lr on tuning rows, every module tried lr .01 and .02. Fact items n=181, standard error about .03. Bigram follows MoME's size rule, table rows = V x memory layers. An earlier version copied 6V without the 6 layers that rule assumes, so it had 8.5M weights; those runs are kept as `bigram6`.

| Score | Plain | Token | VE | STEM | MoME | Bigram | Engram | TN-gram | Mix | Reader |
|---|---|---|---|---|---|---|---|---|---|---|
| Weights | 1.6M | 4.8M | 3.9M | 7.7M | 3.9M | 3.9M | 4.8M | 4.8M | 3.9M | 3.9M |
| Web loss | 5.434 | 5.394 | 5.393 | 5.401 | 5.386 | 5.410 | 5.408 | 5.428 | **5.366** | 5.386 |
| Fact 32x, stored wording, exact | .398 | .619 | .807 | .580 | **.823** | .442 | .536 | .497 | .790 | .801 |
| Fact 32x, new wording, exact | .243 | .381 | **.602** | .420 | .569 | .166 | .243 | .276 | .475 | .569 |
| Name far from question, stored, pick | .343 | .460 | .625 | .460 | **.632** | .321 | .371 | .394 | .629 | .629 |
| Other-relation gap | 3.965 | 3.462 | 3.646 | 3.912 | 3.237 | 4.283 | **4.345** | 4.014 | 3.586 | 3.413 |
| Usual continuation of ambiguous phrase, loss | 1.773 | 1.694 | 1.689 | 1.705 | 1.693 | 1.615 | **1.595** | 1.695 | 1.657 | 1.682 |

- Tables feeding attention values (VE, MoME) store facts far better than Engram with 18% fewer weights (.82 vs .54 stored, .57-.60 vs .24 new wording).
- The same tables are worse at telling relations apart (MoME 3.24, Engram 4.35) and at frequent phrases (1.69 vs 1.60). Engram keeps that lead at both lrs (4.27-4.34, 1.59-1.60).
- Mix (token and hashed bigram slots in MoME's table) has the best web loss, 5.366, but new-wording facts fall to .475. MoME's paper already tests a bigram compound, so Mix is not clearly new.
- Reader (the reading position picks the slot) ties MoME at 5m on web and facts. Relation gap 3.41 vs 3.24 (z=2.1), page gain .603 vs .550 (z=1.9), other-name gap 2.39 vs 2.60 (z=-2.6, worse). Page gain later failed `audit.py`, so don't rely on that part.
- `track.*` never moves at 1m or 5m. Hellaswag, ARC and MMLU accuracy and WinoGrande margin don't move at 5m.
- Fact winners flip with lr: MoME stored is .586 at .02 and .823 at .01, the 8.5M Bigram is .564 and .365. A method needs both lrs before a claim.

### Value table plus a quarter-size Engram (bothve, bothread)

`bothve` is VE in odd layers plus Engram at layer 1 with a quarter of its rows. `bothread` is the same with Reader in place of VE. Both have about 4.7M weights, 1% fewer than Engram's 4.76M.

| 5m, lr | Web | Stored | New wording | Other-name gap | Other-relation gap | Usual phrase loss |
|---|---|---|---|---|---|---|
| Engram .02 | 5.408 | .536 | .243 | 1.424 | 4.345 | 1.595 |
| MoME .01 | 5.386 | .823 | .569 | 2.598 | 3.237 | 1.693 |
| MoME .02 | 5.394 | .586 | .348 | 1.575 | 3.675 | 1.682 |
| Mix .02 (3.9M) | 5.375 | .801 | .519 | 2.549 | 4.097 | 1.649 |
| bothve .01 | 5.380 | .901 | .470 | 3.139 | 3.747 | 1.575 |
| bothve .02 | 5.381 | .812 | .436 | 2.392 | 4.358 | 1.597 |
| bothread .01 | 5.383 | .912 | .530 | 3.679 | 3.491 | 1.591 |
| bothread .02 | 5.380 | .801 | .436 | 2.334 | 4.181 | 1.601 |

- bothve at .02 against Engram at .02: stored +.276 (z=7.2), new wording +.193 (z=6.2), other-name gap +.968 (z=11.1), web -.027 (z=-6.2); relation gap and usual phrase loss tie (z=.1, .1).
- bothread at .01 against MoME at .01: other-name gap +1.081 (z=10.1), usual phrase loss -.102 (z=-4.0), stored +.088 (z=3.0), relation gap +.254 (z=2.4), but page gain -.082 (z=-2.5) and copy gain -.084 (z=-5.2). It has 21% more weights than MoME.
- Lr trades these scores: going from .01 to .02 raises relation gap and lowers stored facts in every value-table method. Tuning rows pick .01 for bothve (5.339 vs 5.345) and .02 for bothread (5.331 vs 5.333), so bothve's Engram-level relation gap needs an lr the tuning rule doesn't pick.
- Mix at .02 has 17% fewer weights and beats bothve .02 on new wording (.519 vs .436, z=3.2). bothve .02 beats it on relation gap (z=2.4) and usual phrases (z=2.2). Neither wins outright.
- Not checked: the scale rule. Only two sizes ran, 1.7M and 4.7M weights.

Is it new: MoME's paper (2609.15126, Table 4) already scales MoME with a Bigram table, and modded-nanogpt uses value embeddings with Bigram, so a value table plus an n-gram table is not a new method. Three extended searches on 2026-11-02 found no paper with Engram plus a value table, none with slots chosen by the reading position, and none comparing these modules score by score on facts. ARM (2609.24417) and MoM (2502.13685) route by the query but before attention, over a different kind of memory. Not read in full, so originality is not settled.

Limits: small models, facts planted in training text, published benchmark accuracy still at chance, one lr pair. Three extended searches found no head-to-head of Engram, VE and STEM on factual recall. Reader had no literature check of its own at this point.

## Existing benchmarks (2026-11-02)

Three extended searches found no side-by-side test of memory modules across tasks and budgets. Closest: "Memory Is Not Always Needed" (2608.23982, scientific reasoning only, not read in full) and a paper library of Engram follow-ups. Originality is not settled until more of these are read.

## Results on the improved evals (2026-11-02)

Evals now include 5 published benchmarks (half hidden, `--final` scores that half into `final_results.jsonl`), 4 text kinds, natural facts, forgetting, and repaired copy, recall, conflict and typo tests. Older runs: `experiments/older-results/results_first_rows.jsonl`, `results_second.jsonl`.

| Tier | Steps | Width | Lr tried | Plain | Token | Engram | MoE |
|---|---|---|---|---|---|---|---|
| 5s web loss | 250 | 48 | .01 .02 .04 | 5.636 | **5.542** | 5.613 | 5.638 |
| 5m web loss | 1500 | 96 | .01 .02 (.04 for Token, Engram) | 5.430 | **5.382** | 5.411 | 5.416 |

Engram table size at 5m (same text, lr chosen on tuning rows; .02 won for all three tables that tried .01, .02, .04, and the 16k table tried .01 and .02):

| Score | Plain | Engram 16k | Engram 131k | Engram 1M | Token (4.8M weights) |
|---|---|---|---|---|---|
| Weights | 1.6M | 2.0M | 4.8M | 26.8M | 4.8M |
| Web loss | 5.430 | 5.423 | 5.411 | 5.412 | **5.382** |
| Web next-token accuracy | .187 | .189 | .191 | .191 | .191 |
| Fact written 32x, stored wording, exact answer | .415 | .509 | .536 | **.675** | .600 |
| Fact written 32x, new wording, exact answer | .249 | .234 | .245 | .298 | **.358** |
| Fact written only in the first third, answer loss | 4.644 | 4.549 | 4.179 | **4.123** | 4.436 |
| Stated code a few tokens back, recall gain | .334 | .284 | .240 | .206 | .276 |

- Web loss and accuracy stop improving past 131k rows, while stored-fact recall keeps rising. Web loss alone would hide that. Lr moves these fact scores a lot: across .01, .02, .04, stored exact is .536-.577 for 131k, .600-.675 for 1M, .453-.604 for Token; new wording is .245-.306 for 131k, .219-.298 for 1M, .260-.366 for Token (265 items, standard error about .03).
- At equal weights Token beats Engram on web loss and new-wording facts; Engram beats Token on phrases seen 10+ times (2.231 vs 2.287) and early facts (4.179 vs 4.436).
- Every table lowers the recall gain from a code stated on the page, so tables trade page use for stored answers; more rows lower it further.

**What the evals can judge at each tier:**

- 5s: web, rarity and answer-loss scores move; most exact-answer, recall and benchmark scores don't.
- 5m: all own tests move for at least 2 of 4 methods; recall-far is the weakest. Published benchmarks are at chance in accuracy (ARC .20, MMLU .19, WinoGrande .52, HellaSwag .30, chance .25/.25/.50/.25) and BLiMP is .58; only some margins move.
- Larger tier: `1h` is width 256, 6 layers, 9.8M weights, 10000 steps, 41M tokens. Plain at step 2500: web loss 5.09, copy accuracy .29 (.17 at 5m), recall gain 1.68 (.33 at 5m), BLiMP .62 (start .47, 500 items). HellaSwag, WinoGrande, ARC and MMLU accuracy are still at chance. Stopped at step 3344: under load it needs over an hour more per method, about 6 hours for four, so this tier belongs on a GPU.

`compare.py` hides scores no method moved at that tier or that `audit.py` rejected, and lists them under the table.

**Limits:**

- Token's lead may be early-training speed from a second embedding at 5x lr. Unknown until longer runs.
- Lr is chosen on web loss, so a fact score at the chosen lr isn't that method's best fact score.
- Hidden benchmark halves are checked to be separate (ARC at 5s: 60 and 60 items, 0 shared).
- 8 parallel runs plus another heavy job make `train_s` here 4x the alone time; don't read speed from these runs.
- Benchmarks score a model trained only on web text, so accuracy needs a much larger budget than the Mac gives.
- The 1h run above had no natural facts because the pool ran out; `env.py` now reserves 400, which changes the 1h world only.

## Not built yet

- Eval group: exact-answer reasoning (GSM8K, TriviaQA and others have no options to score).
- The 5h tier has settings but needs a GPU and more data shards than the one shard here.
- The AI design search.

## New designs: win rule and reasons (2026-11-02)

The win rule is in `beat.py` and was written before any new design ran. Web loss must beat every baseline by 2 standard errors, no baseline may beat the design by 2 standard errors on the 5 other headline scores, and total weights must stay within Engram's at 1m and 5m. A claim needs 1m and 5m.

Two test bugs were fixed before these runs. Fact questions did not start with the end-of-text token that training sentences start with. Items whose answer already sits in the question ("art school works in the area of" -> "art") were scored. Web scores did not change. Fact scores did, so old fact numbers don't compare with new ones. The old rows are in `experiments/older-results/results_before_fact_fix.jsonl`.

Bug check results at 5s: no module sees a future token, no test row is in training, tables change the output in every table module. `nv.weight` gets no gradient on the first step only, because Engram's conv starts at zero. Two runs of one setting give identical scores.

Designs tried: 9 earlier (reader, vgram, mix, mix3, mixbig, both, bothlate, bothve, bothread), then 6 in round 1 (gated, qgram, valgram, rslot, dropve, lateve), 6 in round 2 (late, lve, lmome, lreader, lmix, ltoken), 11 in round 3 (tmome, treader, tmix, tmix1, tmix3, tmix0, tmomeg, tmixz, tmixd, tmomez, tmomed). That is 32 designs, so a win on the public items needs the hidden items too.

Reasons:
- Engram before attention memorizes the answer under the exact wording, so the value table learns less. Adding it to a value table raised stored facts and lowered new wording. Moving it after the last block keeps the gain without that cost: lateve had the first no-loss row at 5s.
- At 1m lmome and lreader beat every baseline on stored facts (.324, .285 vs .635).
- A word table early, after the first block (token), wins web text; ltoken also does. Slots keyed by one and two tokens also help web (tmix0, 5.449).
- So round 3 gives each table one job: word table after the first block for text, slot table in the last memory layer for facts, Engram after the last block for phrases.
- The late Engram costs web loss. Starting its output at zero (tmomez, tmixz) lowered that cost. Dropping it at half the positions cost stored facts.

Noise warning: tmix lr .01 had web 5.453 and lr .02 had 5.462, while tuning rows chose .02. Gaps of .01 at 1m are the size of the change between two runs, and row-by-row standard errors don't include that.

## New designs: result (2026-11-02)

No design meets the win rule. 41 designs tried: the 32 above plus round 4 (tmixz1, tmomez1, tgz, tbiz, tsplit) and round 5 (tmixo, tmomeo, tgo, tmixog). Only tmomez and tmixz ran at 5m, chosen from the 1m results.

5m, lr chosen on tuning rows, z against the best baseline on each score. All 9 baselines were rerun with the fixed test.

| score | best baseline | tmomez | tmixz |
|---|---|---|---|
| web loss, 500 rows | 5.386 MoME | 5.368 (+3.9) | 5.366 (+4.6) |
| stored, 32 copies, loss | .224 VE | .111 (+3.9) | .104 (+5.3) |
| new wording, 32 copies, loss | .962 VE | .838 (+1.8) | 1.161 (-2.0) |
| new wording, far names, loss | 1.784 VE | 1.480 (+3.7) | 1.623 (+1.7) |
| other relation, gap | 5.368 Engram | 3.665 (-12.1) | 3.867 (-10.3) |
| usual answer, loss | 1.595 Engram | 1.544 (+2.4) | 1.534 (+2.3) |
| stored, 8 copies, accuracy | .564 VE | .761 | .797 |
| other name, gap | 3.220 VE | 4.416 | 4.798 |

Weights: tmomez 4.733M, tmixz 4.734M, Engram 4.761M, MoME 3.903M. On all judging scores against the best baseline on each: tmomez 15 better, 3 worse; tmixz 16 better, 6 worse. tmomez is above MoME on the relation gap (3.665 vs 3.573) and below the n-gram baselines and plain.

At 1m the gains are smaller: tmomez web +2.4, stored +.6, new wording -1.5, relation -2.7; tmixz web +.3, relation -1.2.

Stored facts and relation gap trade against each other at 1m across rounds 4 and 5. Moving the slot table to the first layer raised the gap to 4.33 and raised stored loss to .90. Gating on the reading position lowered stored loss to .35 and the gap to 3.29. Putting Engram after the final norm raised the gap to 4.42 and stored loss to 1.32. A reason, not tested: training shows each name only with its true relation, so nothing teaches the model to withhold the answer under another relation.

Not done at this point: hidden items for tmomez and tmixz (done in the next section), more than one training run per setting (still not done).

Literature, 3 extended searches and 2 papers read: Qwen3.8-Next (2608.30320) tested one n-gram layer at layers 1 to 25 and found no depth clearly best, so late placement alone is not new. Memory Attention (2609.28399) found memory added to values better than memory added to attention outputs. Neither has a word table, slot table and late n-gram table sharing one budget. Nothing found with that combination; this is not settled.

## Hidden items, 1m and 5m (2026-11-02)

Hidden rows, facts and benchmark halves, lr chosen on tuning rows. All 9 baselines and tmomez, tmixz were trained again with `--final`. Each setting has one training run.

| score | 5m best baseline | 5m tmomez | 5m tmixz | 1m best baseline | 1m tmomez | 1m tmixz |
|---|---|---|---|---|---|---|
| web loss | 5.365 MoME | 5.344 (+4.2) | 5.345 (+4.0) | 5.577 token | 5.587 (-1.5) | 5.588 (-1.7) |
| stored, 32 copies, loss | .214 MoME | .109 (+3.5) | .100 (+4.1) | .878 VE | .365 (+3.3) | .583 (+3.0) |
| new wording, 32 copies, loss | 1.083 VE | 1.017 (+.5) | 1.209 (-.8) | 2.172 VE | 1.536 (+2.1) | 1.948 (+.8) |
| new wording, far names, loss | 1.928 VE | 1.643 (+2.0) | 1.887 (+.3) | 3.357 MoME | 2.622 (+2.6) | 3.314 (+.2) |
| other relation, gap | 4.806 Engram | 3.307 (-7.1) | 3.141 (-8.1) | 5.145 Engram | 3.702 (-4.1) | 4.206 (-2.7) |
| usual answer, loss | 1.649 bigram | 1.618 (+1.2) | 1.591 (+2.2) | 1.531 Engram | 1.563 (-.7) | 1.545 (-.3) |

Neither design meets the win rule at either tier. The 5m lead on web loss and stored facts held on hidden items; at 1m there is no web lead. The relation gap is the blocker, and at 5m it loses to plain as well.

## Why token-keyed tables lose the relation score (2026-11-02)

Learned codes collapse: `ttrip` at 1m uses 2 of 16 codes per head, and two questions of different relations share a code on 80% of heads, against 82% for two wordings of one relation. Attention does read the name (65-72% of the top reads).
Codes nothing can train separate relations: `thash` shares a code on 52% of heads across relations, 66% across wordings and 80% within one wording.

| 1m, lr .01 | web | stored | new wording | far name | relation gap | usual |
|---|---|---|---|---|---|---|
| baseline best | 5.466 | .635 | 2.091 | 3.011 | 4.395 | 1.715 |
| ttrip | 5.459 | .424 | 2.429 | 2.925 | 4.021 | 1.651 |
| thash, 16 codes | 5.469 | 1.160 | 3.307 | 3.912 | 4.457 | 1.685 |
| thash4, 4 codes | 5.465 | 1.028 | 2.939 | 3.646 | 3.859 | 1.671 |
| tpair, 0.02, attended-pair key | 5.465 | 1.252 | 3.774 | 4.231 | 4.465 | 1.678 |

Splitting rows by relation fixes the gap and costs stored facts at 1m. Open at this point: do facts recover with more training while the gap holds? `thash`, `thash4` and `tpair` were then run at 5m; `python3 tables.py 5m thash thash4 tpair` prints them.

Round 7 prediction, written before running: Engram after block 1 instead of after the last block raises the relation gap of `thash4` from 4.60 to at least 5.0 at 5m, and web loss moves by less than .01.

Round 7 result at 5m: Engram after block 1 raised the gap as predicted (`ehash4` 5.07, `epair` 5.24) and new-wording facts fell far (2.72, 1.92 against VE's .96). Every design so far lies on one line: more relation gap, worse new wording, because a model this small can't tell a paraphrase from another relation.
Round 8 prediction, written before running: Engram after block 2 with the word table after block 1 lands between (`spair` gap about 4.6, new wording about 1.4), so it stays on the line. A point at gap 5.1 with new wording 1.15 would be off the line and worth a 1m check.

Round 9. 1h baselines: the relation gap leader is MoE (6.04), then plain (5.41), and the strong fact memorizers are lowest (MoME 3.99, bigram 4.32). MoE beats plain on facts and on the gap at once, the only part that moves both the same way.
Prediction, written before running: adding MoE (8 experts, its extra weights taken from the table budget) to a three-table design raises the relation gap by at least .4 at 5m and more at 1h, with web loss moving by less than .01.

Round 9 result at 5m: MoE raised the gap by 1.09 on `ttrip` (3.19 to 4.28 as `mtrip`), above the prediction, with web loss unchanged. `mtrip` is 12 better and 1 worse on judging scores; new-wording facts moved from .79 to 1.04.
Round 10 prediction, written before running: MoE's load balancing on Triple's codes stops the collapse (at least 6 of 16 codes in use per head), raises the gap of `mtrip` to at least 4.7, and keeps new-wording facts below 1.3.

Two-relation names, 5m: only 17 of 442 relation items have a name with a second stored relation, so the standard error is about .7 and the test can't tell. It says nothing about the training-pressure reason.
Round 11 prediction, written before running: a penalty that makes the model not give a rare answer after a swapped phrase raises `mtrip`'s gap by at least .5 at weight 1, new-wording facts worsen by less than .2, and web loss by less than .01. `nplain` measures the penalty alone.

Round 11 result, 5m, lr .01. The penalty failed the prediction on cost. `mtrip` without it: web 5.371, new wording 1.037, gap 4.279. Weight .2: web 5.419, new wording 1.643, gap 4.806. Weight 1: web 5.613, new wording 5.813, gap 5.627. Weight 5: web 6.116, new wording 6.760, gap 4.019. It removes facts faster than it adds gap, and it hurts web text as well, because a rare next word is often predicted by the topic of the page and not by the last 3 tokens. Dropped.

Round 12 prediction, written before running: at 1h a table in the first memory layer raised the gap from 4.30 to 5.14, because later layers can overrule what it recalled. At 5m, `mtrip1` (Triple in the first memory layer) raises `mtrip`'s gap by at least .4 and worsens new-wording facts by less than .2. `m2trip` (Triple in every memory layer) keeps the facts of `mtrip` and raises the gap by at least .2.

Round 12 result, 5m. `mtrip1` .02: new wording 1.547, gap 4.919. `mmomez1` .01: 1.330, 4.577. `mmixz1` .01: 1.081, 4.025. `m2trip` .01: 1.071, 4.061. `mtrip` .01: 1.037, 4.279. The placement prediction held on gap and failed on facts, and `m2trip` lowered the gap. All of them sit on the same line as before.

Round 13, a test of which table leaks. `mtrip` has three tables: a word table after block 1, Triple in the last memory layer, and an n-gram table after the last block. Prediction, written before running: the word table leaks most, because it never sees the question, so switching it off at test time raises the wrong-relation gap by the most. `mgram` (no word table) then gains at least .4 of gap over `mtrip`.

Round 13 result, 5m, `mtrip` .01, memories switched off at test time, on stored facts asked with their own wording and with another relation's wording:

| switched off | own question | wrong relation | gap |
|---|---|---|---|
| nothing | -1.372 | -5.651 | 4.279 |
| n-gram table after the last block | -1.772 | -4.637 | 2.865 |
| word table after block 1 | -6.147 | -8.420 | 2.273 |
| Triple in block 3 | -4.063 | -9.179 | 5.116 |
| all three | -8.655 | -8.818 | 0.162 |

The prediction was wrong: Triple leaks, not the word table. Triple raises the wrong-relation answer by 3.5 and the right one by 2.7. A boost that ignores the question adds the same score to both sides, and the right answer can't go above log-prob 0, so better recall lowers the gap by arithmetic. That explains the line between new-wording facts and gap. The word table, which later blocks can overrule, adds 4.8 and 2.8, and the n-gram table adds .4 and takes away 1.0. Removing a table in training lowers the gap too (`mtok` 3.846, `mgram` 3.971).
A way off the line: a paraphrase and another relation differ in the kind of answer they expect. "The capital of" expects a city under any wording, and "The language of" doesn't. So the boost should depend on whether the model's own state already expects the kind of word the table proposes.

Round 14 prediction, written before running: `magree` (`mtrip` with Triple's gate also reading whether the reader's state expects what the table proposes) raises the gap by at least .5 over `mtrip` at 5m, with new-wording facts below 1.2 and web loss within .005. In the switch-off test Triple's boost to the wrong-relation answer falls below its boost to the right one. If training gives the gate no reason to close, the gap stays within .2 of 4.279 and the idea is dropped.

Round 14 result, 5m. `magree` .01 (tuning picked .01 over .02, 5.3235 vs 5.3262): web 5.365 (+4.7), stored .098 (+5.0), new wording .724 (+2.3), name far 1.068 (+8.3), gap 4.082 (-8.4), usual 1.567 (+1.0), 14 better and 3 worse on all judging scores. The gap prediction failed: it fell by .2. Facts moved by far more than I predicted: new-wording facts improved from 1.037 to .724, and this is the first design to beat every baseline on them. In the switch-off test Triple's boost is now +2.3 for the right answer and +3.4 for the wrong one, so it still leaks. The gate helps the reader use a row in a new wording and doesn't close on a wrong relation.
`magree1` .01: new wording 1.398, gap 4.686. `magreed` .01: .807, 3.884.
Round 15 prediction, written before running: new-wording facts now have room (limit about 1.2). `mapair` (pair keys with this gate) keeps the gap of `mpair` (5.04) within .2 and improves its new-wording facts from 1.95 to below 1.5. `magreef` (the gate reads only the fit, not the hidden state) has a higher gap than `magree`. `magree` holds up at 1m: web and new-wording facts both beat the baselines.

Round 15 result, 5m. `mapair` .01: web 5.367 (+4.3), stored .261 (-1.1), new wording 1.430 (-5.1), name far 2.433 (-7.1), gap 5.137 (-1.7), usual 1.583 (+.5). Both predictions on `mapair` held: the same gate on pair keys raised the gap over `mpair` (5.04) and new-wording facts improved from 1.95 to 1.43. This is the first design that doesn't lose the relation score while beating every baseline on web text, and it now loses only on new-wording facts. `magreef` .01: .772, 3.843, so the prediction that a fit-only gate raises the gap failed. `magree8` .01: .866, 4.158.
At 1m the prediction on `magree` failed: web 5.466 (-.1), new wording 2.582 (-1.9), gap 3.627 (-3.5), worse than `mtrip`.
Round 16 prediction, written before running: `mapair2` (twice the rows, less word table) and `mapair8` (8 tokens read) both bring new-wording facts below 1.3 and keep the gap above 5.0. `mboth` lands between `magree` and `mapair` on both scores, about 1.0 and 4.6.

Round 16 result, 5m. `mapair2` .01: web 5.384, new wording 1.599, gap 4.521. `mapair8` .01: 1.691, 5.040. `mboth` .01: .772, 3.702. All three predictions failed: more rows and more reads didn't help new-wording facts, and `mboth` takes the gap of Triple, so Triple's leak wins when the two readers are added. At 1m `mapair` .01: web 5.459 (+1.1), stored 1.371 (-5.7), new wording 3.705 (-4.9), gap 4.223 (-.9): pair rows don't get enough updates in 600 steps.
Round 17 prediction, written before running: the gate fails to close because nothing teaches `expect` what kind of word comes next. `mexp` gives both the state and the proposal a next-word head during training (weight .1 each), so their cosine means agreement in output space. Its gap beats `magree` by at least .5, new-wording facts stay below 1.2, web within .005. In the switch-off test the boost to the wrong answer falls below the boost to the right one.

Round 17 result, 5m. `mexp` .01: web 5.366, stored .196, new wording 1.395, far 2.005, gap 4.502. `mexp8` (sharper gate) .01: web 5.365 (+4.9), stored .169 (+1.7), new wording 1.026 (-.8), far 1.630 (+1.8), gap 4.595 (-5.3), usual 1.591 (+.1). `mtrip` had 1.037 and 4.279, so the gate adds .32 of gap at the same new-wording facts. This is the first step off the line that cost no facts. Headline losses are now only the relation score, to engram, bigram and tngram. The switch-off test on `mexp`: Triple adds +.96 to the right answer and +1.22 to the wrong one, down from +2.7 and +3.5. `mexpair` .01: 1.631, 5.248 (-.9): pair rows add only +.33 and +.19, so the gate almost closes them. `apair` .01: 1.793, 5.127; at 1m web 5.460, stored 1.199, gap 4.228.
The prediction held in part: gap rose over `magree` by .42 to .51, new wording stayed below 1.2 at sharpness 8, and the wrong answer still gets a larger boost than the right one. The second half failed, though the leak shrank by two thirds.
Round 18 prediction, written before running: sharper still is better up to a point. `mexp16` reaches a gap of at least 4.8 with new wording below 1.1. Aux weights .03 and .3 tell whether .1 is near the best.

1h public, with both lrs for every baseline, weight limit 43.54M. Baseline best: web 4.591 mome, stored .084 bigram, new wording 1.417 stem, far 1.825 stem, gap 6.041 moe, usual 1.456 engram.
| design | web | stored | new wording | far | gap | usual |
|---|---|---|---|---|---|---|
| mmomez .01 | 4.550 (+14.3) | .091 (-.4) | 1.370 (+.5) | 1.706 (+1.4) | 4.383 (-14.4) | 1.410 (+4.8) |
| mmixz .01 | 4.553 (+13.2) | .065 (+1.7) | 1.422 (0) | 1.695 (+1.2) | 3.830 (-17.8) | 1.413 (+5.0) |
| mtrip .01 | 4.565 (+9.3) | .082 (+.2) | 1.541 (-1.2) | 1.790 (+.4) | 4.166 (-16.0) | 1.413 (+4.4) |
| mpair .01 | 4.581 (+3.4) | .079 (+.4) | 1.363 (+.5) | 1.803 (+.3) | 4.356 (-14.7) | 1.426 (+3.4) |
| ttrip .01 | 4.580 (+3.9) | .082 (+.1) | 1.519 (-1.0) | 1.765 (+.7) | 4.416 (-13.2) | 1.429 (+2.7) |
At 1h every design wins or ties on every headline score except the gap. The gap is the only loss, and it is largest here, 1.7 behind moe. MoE on these designs doesn't raise the gap at 1h: `mtrip` 4.166 against `ttrip` 4.416. The designs in the table have no gate from rounds 14 to 18.

Round 18 result, 5m, new wording and gap. Sharpness 8: 1.026, 4.595. Sharpness 16: 1.536, 4.606. Sharpness 32: 1.345, 4.448. Aux weight .03 at sharpness 8: .815, 3.755. Aux .3: 2.028, 4.873. The prediction on `mexp16` failed. `mexp8` stays the best of these, though the jumps between sharpness settings suggest that one run per setting leaves real noise in new wording.
Round 19 prediction, written before running: `mapair` loses new-wording facts and starves at 1m because rows are keyed by exact pairs. `mbuck64` keys rows by (token, bucket of another token), with buckets from fixed random hyperplanes over the word embeddings. At 5m it reaches new wording below 1.2 with a gap above 5.0. At 1m stored facts improve on `mapair`'s 1.371 to below .7.

Round 19 result, 5m. `mbuck64` .01: web 5.378 (+1.8), stored .279 (-1.7), new wording 1.644 (-6.0), far 2.377 (-6.3), gap 5.598 (+1.6), usual 1.578 (+.7). `mbuck256` .01: 5.374, .177, 1.210 (-2.9), 1.927 (-1.6), 4.603 (-5.6). `mbuck16` .01: 5.369, .274, 1.566, 2.190, 4.812. The prediction failed on new-wording facts. `mbuck64` has the highest gap of any run that trains normally, above Engram's, but its web lead is under 2 standard errors. At 1m `mbuck64` .01: web 5.481 (-2.3), stored 1.395, new wording 3.258, gap 4.571 (+1.0), so the stored-fact prediction failed too. No design meets the win rule yet.

Round 20, fairness, prediction written before running. The win rule caps weights but not compute, and the designs carry MoE while most baselines don't, so a win could come from MoE alone. Controls made only of published parts: `engmoe` (Engram + MoE at Engram's weights), `vemoe`, `momemoe`. Prediction at 5m: their web loss is within .01 of `magree`, because MoE drives the web gain, and `magree` still leads on stored, new-wording and far-name facts by at least 2 standard errors. Relation gap of `engmoe` stays above 5.0. Speed prediction: `magree` is 1.5x to 2.5x slower per step than Engram, mostly from MoE. Switch-off on `magree`: as on `mtrip`, turning off the gated table raises the gap above 5, so the gate doesn't stop the leak. 1h prediction: `magree` and `mexp8` lead on web and lose only the gap, at least 1.0 behind moe.

Round 20 result, 5m. Speed per training step, one at a time on one thread: Engram 1.00x, `engmoe` 1.17x, `momemoe` 1.21x, `magree` 1.40x, `mexp8` 2.25x. The win rule caps weights only, so compute isn't equal. Weights: `ve`, `mome` and `bigram` use 3.90M, `stem` 7.69M, against the 4.76M limit. Controls (web, stored, new wording, far, gap, usual): `engmoe` .02: 5.407, .607, 2.310, 3.123, 5.193, 1.606. `vemoe` .02: 5.389, .467, 1.415, 2.530, 4.056, 1.690. `momemoe` .01 at 4.35M: 5.379, .179, .786, 1.450, 3.385, 1.665. `momemoe` reaches most of the facts gain with published parts alone, and the web prediction failed narrowly (.014 behind `magree`). Rerun of `magree` .01 gives identical numbers. `magree` .02 has gap 4.858, web 5.369, stored .135, but the tuning rows pick .01. Switch-off on `magree` (own, wrong, gap): all on -1.400, -5.482, 4.082; n-gram off -2.009, -4.569, 2.560; word table off -4.605, -7.860, 3.255; gated table off -3.717, -8.898, 5.181; all off -8.827, -9.220, .393. The gated table raises the right answer by 2.32 and the wrong one by 3.42, so the gate doesn't stop the leak; the switch-off prediction held.

1h result for the gated designs and the controls: see the 1h section of [RESULTS.md](RESULTS.md).
