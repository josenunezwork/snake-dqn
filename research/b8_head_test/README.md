# B8 mass-advantage head: saved-data exploration

This package implements the owner's combined Astra/Fable cheapest test. It reuses
only the 2,048 original C/T H5000 training roots from 524 parent worlds. It never
fits or predicts assessment roots and never opens the separate assessment artifact.
The original source torch containers mix partitions and both suffix families:
loading their bytes is unavoidable. Numerical views immediately retain training
rows; scoring fields are accessed only after the complete OOF action seal.

`run.py` checks the original bank, B8 and eight chunk SHA256 pins, extracts frozen
B8 256-D post-fusion features (Q parity <=1e-3), freezes five parent-world folds
(seed 343013), and fits a fresh 256→128→6 ReLU head per fold. Each head uses its
training fold's feature normalization, legal-action MSE on four-selection-suffix
mean mass advantage over B8, and the doom smoke's 3,000-update final-only schedule:
AdamW lr 3e-4, weight decay 1e-4, batch 32, gradient norm clip 5; initialization
seed 343010 and sampling seed 343011. Targets stay in raw mass units. No sweep.

The action is the largest positive predicted advantage among legal actions,
otherwise B8; B8 wins exact ties. B8's predicted component competes just like the
other predictions (its regression target is identically zero). The diagnostic
teacher maximizes selection mean mass, with B8 winning exact ties. OOF predictions,
chosen actions, teacher actions and all final checkpoints are create-only files.
Only after the full OOF seal do the four independent scoring suffixes enter.

Report chosen/teacher-minus-B8 H300 full-horizon mass and terminal survival, with
root-weighted means and 2,000 whole-parent-world bootstrap draws (seed 343012).
LB90 is the 10th percentile; CI95 is the central 95% percentile interval. CIs
condition on the fitted OOF ensemble, without refitting. Late means length >=600;
ordinary means non-lookback roots of all lengths. Also report ordinary_late and
death_lookback strata. Require >=32 parent worlds per reported stratum.

GO: late mass LB90>0, >=50% of positive independently scored teacher gain retained
(point ratio), ordinary relative mass LB90>=-5%, ordinary terminal-survival point
loss <=1pp. KILL takes precedence when late teacher gain <=0 or late rescue <20%.
Otherwise failure to satisfy GO is GRAY. Unsupported rescue is INCOMPLETE_SUPPORT;
provenance failures/unfinished fits cannot produce a science verdict. This is
exploration; no registration, fresh gameplay, promotion or serving change.

Fable rescue: B8 dies in >=3/4 scoring suffixes and chosen survives in >=3/4.
Non-death roots are the complement (including 2/4 mixed roots). Report late,
ordinary and all-root rates/CIs; the requested Fable bar is late rescue >=35%
and late non-death changes <=15%. Also report paired-suffix rescue: among suffix
events where B8 dies, chosen survives in the same suffix. These are separate
metrics; neither is used for a margin sweep or refit.

Run from this worktree, choosing a fresh output path:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 SNAKE_DQN_DEVICE=cpu nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python research/perf_sim_20261007/with_slots.py --slots 1 --timeout 60 -- /usr/bin/caffeinate -dims /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.b8_head_test.run --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/b8-head-test/run-01
```

Run targeted tests with the same CPU/thread environment:
`python -m pytest -q research/b8_head_test/test_head.py` (use the absolute venv
interpreter above). Results and checkpoints stay outside Git. The runner refuses
an existing output directory and checks a one-hour deadline in fitting. Argon is
the required independent reviewer; quota unavailability must be disclosed.

## Completed run (2026-10-30)

Run pin `61fb8b00359c159720f24cfc011b7ec2f538a205`; artifacts:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/b8-head-test/`.
See `RESULT.md`, `run-01/result.json`, `verification.json`, and `pytest-final.log`.

Exploratory mass-rule **GO**: late mass +30.089, LB90 +23.962, CI95
[20.958, 39.943]; independently scored teacher +50.682; retention 59.37%.
Ordinary relative mass LB90 −0.756% (guard −5%); ordinary survival point delta
−0.391pp (guard −1pp). Fable bar **FAIL**: late rescue 26.42% vs 35%,
non-death late action changes 97.60% vs 15%. No post-look tuning.

Five final heads, 3,000 updates each; main-to-scoring wall time 5.023s,
CPU process time 4.978s (imports/slot wait excluded). A replay of all checkpoint
predictions was bitwise equal to the OOF seal; seven targeted tests pass.
Argon independent review is NOT RUN because live Gemini quota is 0%; see
`review/REVIEW_STATUS.json` and the retained review request. No main merge,
release or serving change.

## Closed-loop exploration

`closedloop_prepare inputs` fits exactly one fresh head on all 2,048 original
training roots with the same 3,000-update recipe. Only selection suffixes supply
training targets. A full-history Mac seed audit runs before gameplay. The new
namespace is `t34-b8-head-closedloop-20261030/v1`, with separate main, canary and
preflight domains and independent world/opponent keys.

`closedloop.py` compares bare B8 with fixed margins **0, 10, 25, 50** on 192
fresh paired v1 terminal-hero H5000 worlds, 64 per scripted/frozen/mixed roster.
The proposed legal action replaces B8 only when its predicted advantage minus
the predicted component for B8's action is strictly greater than the margin.
B8 wins equality. No margin or checkpoint is selected from the results.

The GPU runner reuses the pilot CT simulator, canonical B8 observation/mask and
batched frozen/scripted opponents. It duplicates the world and opponent RNG keys
across all five arms, checks complete initial-state digests, excludes dead hero
frames from mass, and divides mass/alive frames by H5000. Only living decisions
enter override denominators. Opponents respawn; the hero does not. World batches
are immutable, namespace/allocation/head-bound units. Partial batches are never
published. A verified ledger skips exactly complete units; the Mac receipt-bound
ledger builder refuses duplicate units and computes no outcomes.

`closedloop_readout` opens the verified complete three-batch union only on the
Mac. It reports every margin plus B8, each mix and equal-weight overall results:
paired mass delta, one-sided LB90, central CI95, deaths, alive fraction and
override rate. Percentile bootstrap uses 10,000 whole paired-world draws within
mix, seed 343014; intervals condition on the single fitted head.

Prepare inputs, run the tiny real Mac entrypoint, commit/publish the source, then
seal a canary packet with `python -m research.b8_head_test.closedloop_prepare
packet --input-dir INPUTS --output FRESH_PACKET --commit FULL_SHA`. Run its
`DISPATCH.txt` dry-run and enqueue through the live `tools/ops/ready_queue.py`.
The A100 canary uses three excluded full-arena H64 worlds and all five arms. A
main packet uses the same source and inputs and binds its genuine stopped canary
receipt via `--receipt CANARY_OUTPUT/receipt.json`. The ready queue may prepare
that main only after the canary passes. The main forecast is 90 minutes; the
whole-session runaway timeout is 180 minutes (base 90, doubled), with informational
A100 cap 24 units and reserve 4 units. These are forecasts, not measured A100
performance for this head.

Watch6-v2 is omitted from this packet: the reused simulator's contract is v1
terminal-hero skill, while product Watch6-v2 requires v2 mechanics, continuous
hero respawn and a different roster. Adding that adapter is separate work.
