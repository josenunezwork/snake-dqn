# T4c same-init drift diagnostic — 2026-10-16

Diagnostic only. This evidence does not authorize training, promotion, or a
change to the registered T4 decision.

Five paired finals were compared on each CTRL final's own scripted H5000
states. Round 62 is the original T4a `(c)` source. Round 65 uses the separate
registered scripted rows; frozen and mixed round-65 rows are deliberately not
combined because that would no longer be the `(c)` state-set comparison.

All action values are raw six-action first argmaxes. `(c)` excludes exact v2
bank observation fingerprints and uses pooled decisive-plus-counter bank
stratum weights. Round 62 reproduces the original pooled five-seed CTRL
cutpoints; round 65 recomputes pooled cutpoints on its own CTRL cohort under
the same matching method. A
consequential change is a changed argmax with `CTRL Q(best) - CTRL Q(other
model's selected action) > 0.3`; its denominator is every retained `(c)` row.

| Round | Comparison | Raw action change | Consequential change |
| --- | --- | ---: | ---: |
| 62 | B8 same-init drift | 9.0032% | 0.9662% |
| 62 | TWO_SIDED | 53.2161% | 10.6993% |
| 62 | LABELS | 92.7372% | 14.4861% |
| 62 | LABELS_KEEPV8 | 86.6763% | 12.8244% |
| 65 | B8 same-init drift | 8.9263% | 0.8798% |

The round-62 raw rows reproduce the complete historical raw `(c)` result
dictionaries exactly at full precision. Rounded values are:
TWO_SIDED 53.22%, LABELS 92.74%, and LABELS_KEEPV8 86.68%. The relevant
pre-registration reference is the round-62 same-init consequential baseline,
0.9662%, with a declared tolerance of ten percentage points.

Planner overlap note: on round 65, TWO_SIDED versus CTRL deaths overlap in 96
worlds; 106 are CTRL-only and 119 TWO_SIDED-only of 480. That is close to the
independence expectation (91/111/124), while TWO_SIDED deaths occur about 300
frames later. World-level overlap therefore cannot test whack-a-mole.

Evidence is in
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t4c-drift-20261016/`:
`captures/`, each capture receipt, and `drift-summary-verified.json`. The initial
`drift-summary.json` remains preserved as the earlier unvalidated aggregation.
The verified report is also checked in under
`research/t4c_20261016/results/drift_summary_verified.json`. Replays are
CPU-only, one thread, MPS disabled, and validate mass, deaths, alive frames,
terminal frame, and terminal length against their historical CTRL rows.

The final report independently validates original receipt schemas, seed/round
identities, every originally bound input hash and every capture payload hash.
For round 62 it verifies the historical three-arm receipts and exact length,
fingerprint, CTRL-Q and candidate-Q joins. Its evidence map binds the original
receipts without rewriting them. Six diagnostic tests passed, including strict
selected-action gap semantics, changed input/payload rejection, wrong-seed
rejection, fixed pooled weights, missing support and exact historical regression.

Every arm has five per-seed raw/consequential rates and a separately identified
arithmetic mean in the verified JSON. Per-seed rates use the same pooled
cutpoints and bank weights as their round's headline; missing support would
produce null without extrapolation. All five seeds have full support here.
The headline reproduces T4a's pooled standardization and is not the arithmetic
mean of those five rates. Cutpoints use all captured living CTRL states, then
the common bank support restricts the reported comparison to L>=700. Round 65
uses its own pooled CTRL cohort to apply the same matching procedure.

Source provenance limitation: original runtime capture receipts did not bind
the diagnostic Python source. Archived source snapshots were taken after
capture and before cleanup; they are contextual source evidence, not runtime
attestations. The final report's hash/input validation is explicitly retrospective.
Future replay code refuses existing captures and rechecks bound inputs after
parity validation before publication. No original receipt was upgraded to claim
checks that had not run at capture time.
