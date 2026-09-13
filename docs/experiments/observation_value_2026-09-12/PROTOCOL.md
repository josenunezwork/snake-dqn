# H1-A remote-heading decision-relevance protocol

Status: **frozen design; source and checkpoint closure pins pending.** This
protocol tests a narrow question: under matched, finite counterfactual rollouts,
can rotating one eligible enemy's heading change the near-best hero action? It
does not measure natural alias prevalence, learned-policy quality, or an optimal
action-value function. The legacy/default runtime, incumbent, `main`, and the
terminal B5 record remain unchanged.

The durable evidence root is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/observation-value-20260912`.
G0, E1, and P2 are accepted prerequisites; the root must still pin the final
integrated source and checkpoint closure before construction.

## Why this test is narrower than “aliasing exists”

Perceptual aliasing matters for control when indistinguishable percepts require
different responses; it is not established merely by finding different hidden
states behind one observation. Chrisman uses that control-relevance distinction
to motivate missing state information in partially observed learning
([Chrisman, 1992](https://cdn.aaai.org/AAAI/1992/AAAI92-029.pdf)). Likewise,
state-abstraction theory distinguishes preservation of action/value properties
from coarser aggregation ([Li, Walsh, and Littman, 2006](https://thomasjwalsh.net/pub/aima06Towards.pdf)).

This probe therefore compares actions under one controlled perturbation. Its
finite, tape-controlled returns are not optimal Q-values: they do not optimize
the continuation policy, model arbitrary natural action probabilities, or
bootstrap a learned value. Recurrence or explicit history remain possible later
candidates only; recurrent Q-networks are a relevant POMDP technique, not
evidence that this game needs one ([Hausknecht and Stone, 2015](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/SDMIA15-Hausknecht.pdf)).

## Frozen corpus and collection

Before constructing a world, the runner creates an immutable manifest that pins
the final source closure, configuration, seed derivation/version, policy class
identities, trusted-local-pickle hashes, and artifact schema. The final source
and checkpoint hashes are intentionally blank at this design stage. The manifest
and result tree are create-only; all attempts, including rejected, missing, and
partial cases, are retained.

| Item | Frozen value |
| --- | --- |
| World split | 24 distinct derived world seeds: 8 development and 16 untouched holdout. Holdout runs once with no result-driven tuning. |
| Environment | `BatchSimConfig`, `E=1`, `S=6`, mechanics v2, `train_mode=True`, `allow_respawn=False`; defaults: 1450×830, segment 10, food 250/max 300, gamma 0.99, max capacity 400. |
| Probe limits | `max_frames=5000`, hunger 500, max length 400; at most 256 natural steps per world; stop at population floor or frame cap with no reset/replacement. |
| Roster | Slots 0–5: `random_safe`, `greedy_food`, `random_safe`, `greedy_food`, `random_safe`, `greedy_food`, using public source classes and independently derived per-world policy seeds. |
| Candidate frames | Only natural pre-step frames 16, 80, and 160. A dead slot-0 hero rejects the attempt. |

The collection ceiling is 24 × 256 × 6 = **36,864 natural agent-slots**.

## Candidate and counterfactual construction

For an eligible frame, choose the lowest-ID alive single-segment enemy. Clone the
complete simulator state and RNG state; hash the trusted local pickle before
mutating it. The variant rotates that enemy's heading by `+1 mod 4`. There is no
alternate-enemy search, fill-in candidate, or replacement attempt.

Retain a pair only when the two variants have byte-identical canonical
`raster31v3` observations and resolved masks, and each is geometrically
admissible. This is a conditional geometric witness, not proof that either state
is reachable from a different natural history.

For every retained pair, derive one 32-row joint relative-action tape from the
pair seed, with left/straight/right probabilities 0.1/0.8/0.1. Use the same tape
for both variants and every initially resolved hero action; overwrite row 0 with
that action.
For unavailable boost actions record `None`; do not manufacture duplicate rows.
Run prefixes 1, 8, 16, and 32 total steps, including row 0. H16 is primary;
the other horizons are descriptive.

Each branch stops on death, population floor, or frame cap. Its remaining finite
horizon rewards are zero. There is no learned bootstrap and no optimal-Q claim.
Each return is the finite discounted sum using the frozen gamma of 0.99.
The counterfactual ceiling is 72 pairs × 2 variants × 6 initial actions × 32
frames × 6 snakes = **165,888 agent-slots**. The complete bound is **202,752**,
below 250,000. One tape is used, never a separate tape per variant.

## Metrics and decision rule

Let `L_a` and `R_a` be the H16 finite controlled returns for initially resolved
hero action `a` in the original and rotated-heading variants. Use tie margin
0.01 to form each near-best set. “Disjoint” means no action appears in both
near-best sets. For each eligible pair, calculate:

```text
best_common_regret = (max(L) + max(R)) / 2 - max_a((L_a + R_a) / 2)
```

This deliberately gives the hidden-state variants equal weight; it is not a
natural-state probability estimate. Aggregate equal-weight world means, not
rows, tapes, or actions. Use 2,000 fixed-seed bootstrap resamples of holdout
worlds and a one-sided fifth-percentile lower confidence bound (LCB) for the
world mean best-common regret. A world with no accepted pair is uncertain, not a
zero-regret observation.

Advance only when all conditions hold on holdout data:

1. at least 8 eligible holdout worlds;
2. at least 3 holdout worlds with disjoint near-best sets; and
3. strict LCB > 0.01.

Then emit `ADVANCE_INFORMATION_DIAGNOSTIC`, authorizing only a separately frozen
heading-versus-history diagnostic. Otherwise emit
`INCONCLUSIVE_NOT_ADVANCED`. Neither outcome establishes global observation
adequacy, natural alias prevalence, or a representation/model change.

## Execution, limits, and handoff

Run one CPU process with one numerical thread. The worker has an 840-second wall
limit, sampled RSS cap of 2 GiB, and minimum available memory of 6 GiB; the root
holds the exclusive CPU lane for at most 900 seconds. Do not run MPS work or
tests concurrently. Stop rather than extend, replace, or complete a partial
corpus.

| Wave | Owner and gate |
| --- | --- |
| Implementation | Core and statistics builders work independently; the CLI consumes their frozen API. |
| Review | Focused union/oracle tests, independent review, and verifier checks run before execution. No generic full suite is required for this pure diagnostic. |
| Execution | The root serially pins source/inputs, runs the one CPU corpus, audits raw artifacts, and preserves every terminal status. |
| Reporting | This file stays frozen. `REPORT.md` is created only from the final receipts and must repeat the conditional-geometric boundary. |

H1-A is an amendment that answers decision relevance before fitting histories.
Dynamic pools, PPO, history predictors, recurrence, quality evaluation, and
promotion are outside this protocol.
