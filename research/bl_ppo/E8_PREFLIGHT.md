# E8 implementation and E9 blocked freeze — Mac preflight 2026-10-21

Status: IMPLEMENTED_AND_LOCALLY_VERIFIED; E2_ADMISSION_BLOCKED; NOT_DISPATCHED.
Published executable source pin: `0b53afdf094fb689878d10cc8a9c405ae623b682`. Worktree: `/Users/josenunez/Projects/ml/snake-dqn-bl-e2`,
branch `bl-e2-ppo-pilot-20261021`. No Colab calls, scientific E2 collection,
strict gate, serving qualification or release occurred.

Authority: owner request and [E8_AMENDMENT.md](E8_AMENDMENT.md), committed
as `243c6978` before the amended probes. [E9_RULE.md](E9_RULE.md) freezes the
resulting rule. Original INTENT.md, PREFLIGHT.md, BANK.json and SEED_AUDIT.json
remain byte-identical. This is development preparation, excluded from registered verdicts.

## Final real Mac preflight

Primary Python 3.12 venv, CPU, nice 10, all math/Numba thread caps of 1; two worlds,
one collector, at most two numerical threads. Other owner activity remained running.
These are contended operational timings, not serving latency or L4 forecasts.

Each R/F run completed 400 total hero transitions: 320 critic-only collected rows
(160 fit-world rows) plus 80 actor rows, four critic updates plus one actor update.
The tiny override is `BYPASSED_PREFLIGHT_ONLY`, scientific=false, qualified=false.
It exercises execution/export/recovery and cannot certify the required 2M fit rows,
held-out EV or any efficacy decision.

| Probe | Run wall seconds | Collection seconds | Learner seconds |
|---|---:|---:|---:|
| R | 11.315095584 | 1.601626251 | 2.590705250 |
| F | 10.706465250 | 1.555261333 | 1.806246917 |
| R resume | 3.003741959 | 1.601626251 | 2.590705250 |

R explicit actor-phase resume preserved counters, optimizer/PopArt/recurrent state
and reproduced the endpoint SHA256 exactly:
`a7d620b961d8053a58f172aabb72aa15762be0245998b794e4e11fd6ecd2e75f`.
The referenced immutable bank payload was required and hash-verified on recovery.

Long-body/native-v8 ruler: 24 measured hero decisions /0.094311375s =
254.476196545 active decisions/s.
Startup/calibration/warmup/close-inclusive allocation scope:
24 /3.486789083s = 6.883123535/s.
This scientific=false 2-world/1-process probe cannot admit E2; E4 remains 128 worlds,
6 versus 12 processes with exactly the prior manifest semantics.

Actual isolated stochastic/greedy probes completed 240 simulator transitions per
arm on the three excluded H40 fixtures. The 9-controller H40 parity smoke also
completed. Those fixtures never contribute to a screen estimate. H40 cannot
replace the full-H5000 Mac/runtime qualification required before scientific screening.
A cached-H40 linear extrapolation for 224 worlds × 3 controllers is about
2.327 Mac CPU-hours; it excludes long-body
cost changes and is not a qualified upper bound. Reprice using excluded H5000
operational timings before admitting the 672-episode screen.

Exact reproduction commands, status receipts, snapshots, probes and parity:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e2/e8-preflight-04/execute_preflight.py` and `summary.json` (immutable sidecars).
An earlier provisional preflight exposed a missing survived_horizon batch field;
`db102929` fixed it and the final pinned preflight passed. Failed/provisional
attempts remain preserved, never presented as scientific data or dispatch inputs.

## E9 result and admission

20,000 replicates/scenario; seed 20261021; paired world as unit; equal-mix
stratified variance, measured T23 paired SDs 364.683/262.732/212.360.
E2 alpha .05 and E6 alpha .05, Holm R/F inside each route; shared-B8 rho 0/.5/.9,
true effect 0/20/40/60/80, survival delta -.05/0/.03/.06, full survival/no-death
and ordinary-root guards. Both-arm and joint GO/KILL/inconclusive outcomes are exported.

| Screen | Maximum family null-pass | MC95 upper | Minimum joint funding at+40, nonnegative survival | MC95 lower |
|---|---:|---:|---:|---:|
|96/32/32|.05050|.053622|.27045|.264338|
|128/48/48|.06160|.065017|.30700|.300645|

The permitted escalation is frozen prospectively, but the>=.60 requirement fails.
E2 materialization and scientific runtime fail closed on this report. No thresholds,
guards or N beyond 128/48/48 were changed. Null <= .10 passes only within these models.
S1 final variance data was unavailable in the inspected approved locations; E6
variance uses a labelled historical T23 proxy. Broader round-05 E16/E17/32M OC is
explicitly incomplete without a registered joint forecast model. No fabricated
S1 SDs, future outcome probabilities or units replace missing inputs.

Final report: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e2/e9-oc-04/report.json`; SHA256 `9b8ae3eb5ba02f85ed979de936191b9f5c3d3ceb270784cce444c14f9f26530e`.

## Bank and prepared manifests

Prospective bank: 128 scripted /48 frozen /48 mixed plus 6 excluded fixtures.
`screen-bank-v3/bank.json` SHA256:
`40729f13e879a366179cd54df283df594ccb4faf4327d53f962092ce5b4a6f49`.
Audit SHA256: `607d57c9b409ed6db1de8c85c9a3c87947a6f79b4d10f58afc5970a7471111dc`.
Zero collisions; 115,450 approved text files, 28,256 reconstructed historical seeds,
20 published inventories. The scanner prunes the prohibited live-study directory
before traversal. A same-directory S1 alias is skipped only because its regular
SEED_AUDIT.json target is separately audited. External links are refused.
No sealed live-study file was opened in this task. Legacy v1 allocation reproduces
its immutable prior bytes; the new v2 namespace is disjoint.

All three local supervisor validations passed; NONE dispatched. Final directory:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e2/dispatch-e8-v2/`.
- `e4-manifest.json`: `124b10dc3ca084e06462d46b11c08510136841e30aa8d49afbf644a4f4bfdbea`
- `e2-r-template.json`: `0e4967e134c8ce766158cd8979fec48bbd5e56395d4a000e240f1c9171bfa506`
- `e2-f-template.json`: `367350aec4b765885275fe637338e2c48a9003832e8598d03a58b485bb02bc9e`

E4 semantics compared equal to dispatch-v1 after excluding source/output identity
repins. R/F are deliberately blocked templates with no fabricated E4 input.
Sparse fallback source: 732 files, 2,409,749 compressed bytes. Exact source install,
isolated imports and an actual native-v8 step are verified under source-e8-v2.
Root DISPATCH.txt is the authoritative mutable pointer; dispatch-e8-v1 and earlier
preparations are superseded.

## Forecasts and recovery bounds

E4 remains L4, 15 min estimate / .5 u,30 min wall cap and 2 u registered cap; at 2 u/hour,
the wall limit costs <= 1 u. E2 remains two 9 u / 270 min arm caps plus 2 u / 1 h unallocated
reserve: 20 u / 10 allocation-hours,48 elapsed hours, first cap wins. No retry.

An arm target counts actor rows; the new forecast additionally charges 3.17M
critic fit/holdout/completion collection rows (at least 2M actual fit rows),
10k bank allowance, 25% learner/export overhead and 15 min setup. Each 1M report plus
endpoint runs three paired H5000 probes; serial probes are charged at 12× the
aggregate E4-rate duration until measured. The generator refuses an estimate > 135 min
because the unchanged supervisor requires a 2× timeout margin inside 270 min.

| Hypothetical E4 rate | Actor target | Per-arm estimate | Per-arm units at2u/h | Forecast admission |
|---:|---:|---:|---:|---|
|999/s|4M|213min|7.10|REFUSED: timeout margin|
|2,000/s|8M|168min|5.60|REFUSED: timeout margin|
|3,000/s|8M|117min|3.90|Fits forecast only|
|4,000/s|20M|172min|5.73|REFUSED: timeout margin|
|6,000/s|20M|120min|4.00|Fits forecast only|

These are conditional planning numbers, not measured L4 learner/probe rates.
Every row still fails scientific admission while E9 is BLOCKED.20M additionally
requires the explicit pre-data switch and genuine E4 >= 4k/s; the default is 8M.

E2 alone changes snapshot publication to >= 1,800 s and aggregate exports to 2 GiB.
Each dynamic snapshot is <= 160 MiB; immutable ordinary/refit banks are exported once
and bounded at 192/64 MiB. At most 10 snapshots including terminal within 270 min use
<= 1,600 MiB, plus <= 256 MiB static state, leaving 192 MiB for endpoints/reports.
Byte-bound violations produce INCOMPLETE. Snapshot spacing plus a current rollout
is the recovery-loss bound; transfer latency is additional and remains unmeasured
for full-size state. All payloads use .sha256 and .complete, completion last.
Explicit recovery requires a fresh output directory and the original snapshot
plus every referenced immutable sibling payload/sidecar. It preserves the same
target, source, E4/E9 identities, seed/topology, original caps and elapsed window.

## Targeted verification and protected scope

Primary venv, CPU/thread caps as above:
- `python -m pytest -q research/bl_ppo --ignore=research/bl_ppo/test_collector.py`: 96 passed.
- Separate process: `python -m pytest -q research/bl_ppo/test_collector.py`: 5 passed.
- Black 100, isort profile black, flake8, Bash syntax and git diff whitespace passed.
- Actual native snapshot replay/partition identity, sidecar discovery/tamper/size
  rejection, frozen-bank sibling recovery, heldout fit masks, PopArt preservation,
  rollout-cut bootstrap, life diagnostics and endpoint ledgers are covered.

Only research/bl_ppo authored. No protected T34 package or research/colab_cli edited;
no released checkpoint, serving pin, tmux service or other owner's process changed.
No CUDA execution or scientific critic preparation occurred; those claims remain
unverified and cannot be inferred from the tiny preflight.
