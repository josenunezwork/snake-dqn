# RL research portfolio and execution waves — 2026-09-12

## Purpose and current decision

This note keeps the reinforcement-learning work legible: it records what was
implemented and observed, what research says is plausible, and which next
experiments would separate those possibilities. It does **not** select a new
algorithm, promote a checkpoint, or change the incumbent Apex path.

The working conclusion is that the project is on a reasonable path, but the
new raster/PQN stack is still an experimental learning system rather than an
established replacement. Once the decision-phase qualification is complete,
the most useful next fact is elementary learnability under a fixed opponent
distribution. That fact comes before PPO, a replay system, reward redesign, or
a self-play curriculum, because each of those would otherwise add a new
explanation for a weak result.

The proposed small study was deliberately limited to two independently
initialized corrected-PQN arms, each with 50,000 useful hero transitions
against five stationary `random_safe` scripted opponents. It uses exact
one-epoch coverage, one MPS learner at a time, and matched initial/final
evaluation on four fresh worlds at the required 5,000-frame v3 horizon. It is
entirely unrun and deferred after the phase finding; it is a diagnostic, not a
method comparison, and cannot establish that PQN is faithful to the published
recipe, that PQN beats PPO or Apex, or that a model is promotable.

Before interpreting a policy comparison from that diagnostic, qualify its
decision phase against serving and evaluation. The source audit and fixed-seed
witness found that the legacy trainer selects from a pre-maintenance state,
whereas live play and strict evaluation select after frame/food preparation.
An opt-in Watch-aligned implementation is now committed with focused contract
tests and a passing CPU non-slow suite. Five-update CPU and MPS runs completed
with exact sampled-coverage counters, changed weights, and serialized strict
profile checkpoint validation; the independent audit passed engineering
qualification. This remains a contract repair, not evidence that action choice
is harmed or that a model comparison is invalid.

## Repository facts that constrain the work

Two stacks coexist and answer different questions:

| Stack | Current role | Key contract | Decision boundary |
| --- | --- | --- | --- |
| Apex DQN | Incumbent champion | `vector61`, distributed prioritized replay, target network, n-step returns | Remains the baseline and production-quality candidate. No change is proposed here. |
| Raster PQN | Redesign experiment | `raster31v3`, `BatchSim`, synchronous Q(lambda), no replay/target/PER | Must first demonstrate bounded learning and then pass the shared promotion gate. |

`PQNTrainer` implements rollout collection, Q(lambda) target construction, and
SGD in one synchronous loop. It uses an online network whose rollout-time
values are held fixed for the update; this is update-local target lag, not a
DQN target network. The corrected recipe requires v3 observations, mechanics
and reward version 2, rectangular geometry, and disabled legacy flip
augmentation. See [the trainer](../../../src/training/pqn_trainer.py) and
[the CLI resolver](../../../src/scripts/train_pqn.py).

The exact-coverage repair is opt-in. `sgd_epochs=None` retains the historical
four independently shuffled minibatch prefixes; a positive `sgd_epochs` visits
each eligible hero transition once per epoch. This distinction is material: a
claim about full-rollout reuse requires `sgd_epochs=1` (or another explicit
positive epoch count), not merely `corrected-v3`.

The shared authority for promotion remains
[`tournament_eval.py`](../../../src/scripts/tournament_eval.py). Its headline
metric is total-horizon mass integral, so dead frames contribute zero. A short
learning diagnostic uses the same strict v3 evaluation profile, but is not a
promotion gate and must not be described as one.

## What has been tried and what it established

| Work item | Result | What it establishes | What it does not establish | Durable record |
| --- | --- | --- | --- | --- |
| Watch decision-phase repair | Implementation committed; focused contracts and CPU non-slow suite pass; CPU and MPS five-update executions and independent audit complete | Prepared Watch decisions, target/provenance boundaries, bounded device training, and serialized strict-profile checkpoint loading work for the smoke configuration | A full episode/autoreset demonstration in the smoke, policy quality, an algorithm comparison, or promotion | [decision-phase contract](../../pqn_decision_phase_2026-09-12.md) and [independent audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/decision-phase-20260912/independent-audit/phase-qualification-v1.md>) |
| RL contract repair / G0 | Completed source, simulator, evaluator, provenance, and serving qualification work | The repaired paths have focused and integrated test evidence; v3 contracts and evaluation semantics are explicit | Model quality or a winning algorithm | [execution record](../rl_repair_execution_2026-09-08.md) |
| Perenvironment autoreset B5 | `INCONCLUSIVE_STOP` after one completed barrier arm and one worker-complete/parent-unconfirmed arm | Lifecycle engineering, per-env reset oracles, CPU/MPS smoke paths, and exact within-arm sampling counters | A throughput gain, policy-quality gain, or accepted matched comparison | [B5 follow-up](../pqn_followup_2026-09-12.md) |
| H1-A remote-heading probe | `INCONCLUSIVE_NOT_ADVANCED` | The controlled perturbation can change later observations in 51 of 113 matched-action pairs; the finite H16 test did not show disjoint near-best actions | Global representation adequacy, natural alias prevalence, or that recurrence is needed | [H1-A report](../observation_value_2026-09-12/REPORT.md) |
| Visible-cue VC1 | `RESOURCE_CAP / INCONCLUSIVE` | The first seed fit training food labels but held-out food accuracy/macro-F1 was only 0.593/0.530; the second stopped at update 56 under the cap | A broad representation conclusion, an RL improvement, or a model change | [experiment ledger](EXPERIMENT_LEDGER.md) and external `research-portfolio-20260912/visible-cues/execution-v1/` receipts |
| LP1 fixed-opponent learning diagnostic | `UNRUN_DEFERRED` (zero arms, zero MPS runs, zero evaluations) | The originally frozen protocol is retained as a deferral record | A quality result, a PQN/PPO comparison, Apex parity, or promotion | External artifact root: `research-portfolio-20260912/learning-probe/` |

The B5 parent-monitor race was repaired for future supervision, but it does not
retroactively turn B5 into an accepted throughput comparison. The H1-A result
is equally easy to overread: its controlled current-observation aliases did
not produce a holdout decision-relevance signal under the finite rollout
oracle, and its authors explicitly recorded that limitation.

Every numerical attempt belongs in an append-only execution folder with a
manifest, source/config/seed closure, initial and final checkpoint hashes,
telemetry, raw evaluation rows, terminal receipts, resource observations, and a
plain-language verdict. A failed cap, tripwire, missing receipt, or unconfirmed
process exit stays in the ledger; it is not silently retried, replaced, or
collapsed into a favourable aggregate. This is how we retain both successful
and negative evidence without turning exploratory results into false
conclusions.

## Research synthesis

### 0. Decision-phase parity is a precondition for interpreting learning results

At reset, the legacy PQN rollout obtains `_current_obs()` and selects its first
action against frame zero with the initial 250 ambient food pellets. Its
subsequent `BatchSim.step()` increments the frame and tops food up to the
configured 300 before executing the selected action. The live `GameState` and
the strict SIMD evaluation path instead perform frame increment and food
maintenance before policy selection. The actor can therefore be trained on a
pre-preparation observation/action boundary while it is evaluated and served
on a prepared boundary.

This is a concrete phase-contract discrepancy. The completed fixed-RNG witness
showed food count 1 to 2, frame 0 to 1, and normalized episode progress 0 to
0.0002 across the preparation boundary; it deliberately did not establish a
different chosen or optimal action, return, or ranking. The opt-in
`watch_pre_move_v1` repair now captures its rollout-edge bootstrap state from a
discarded prepared clone and binds its changed target contract in provenance.
The default stays legacy-compatible. The completed device qualifier supports a
new study's engineering launch, not a policy-quality claim from an
initial/final delta.

### 1. Corrected PQN is a defensible experiment, not the published recipe

The primary PQN source is [Gallici et al., 2025](https://arxiv.org/abs/2407.04811),
with the authors' [PureJaxQL implementation](https://github.com/mttga/purejaxql/blob/47af6d7b35c89ddfe633aaf7341bdb8964cb7cce/purejaxql/pqn_gymnax.py)
and [reference configuration](https://github.com/mttga/purejaxql/blob/47af6d7b35c89ddfe633aaf7341bdb8964cb7cce/purejaxql/config/alg/pqn_cartpole.yaml).
Those references use complete rollout reuse for configured epochs (the shown
configuration uses four), RAdam, squared TD loss, and optional learning-rate
decay. The Snake diagnostic freezes a different, intentionally smaller recipe:
Adam, Huber, fixed learning rate, and exactly one complete SGD epoch. It tests
the repository's corrected Q(lambda) mechanics, not an exact replication of
published PQN. A negative result therefore does not refute PQN; a positive
result does not validate all recipe choices.

This first comparison is within a learner: final policy versus the same
learner's saved initialization. Its purpose is to catch a basic absence of
learning before a much more expensive control implementation. The independent
unit is the training seed, not each evaluation world. This follows the warning
from [Henderson et al.](https://ojs.aaai.org/index.php/AAAI/article/view/11694)
and [Agarwal et al.](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)
that RL results are highly sensitive to run and implementation variation.

### 2. Stationary opponents isolate a real self-play confound

`--no-self-play` does not produce a stationary baseline: it collapses the pool
and makes all six slots live heroes, so the population still co-adapts. The
current diagnostic instead injects a `FixedPolicySource` around the repository's
deterministic `ScriptedAnchor("random_safe")`; slot 0 is the forced hero and
the other five slots are fixed opponents. The policy identity and realized
exposure must be recorded. This uses existing `PQNTrainer` fixed-policy
support rather than altering training or serving code.

That sequencing is consistent with [fictitious self-play](https://proceedings.mlr.press/v37/heinrich15.html),
[PSRO](https://arxiv.org/abs/1711.00832), [OpenAI Five](https://openai.com/index/openai-five/),
and [AlphaStar's PFSP league](https://www.nature.com/articles/s41586-019-1724-z):
historical populations and measured opponent selection can improve robustness,
but they make a weak learner harder to diagnose. A uniform FIFO snapshot pool
should only be reintroduced after the stationary baseline has a measured,
repeatable signal. The existing Apex `CurriculumManager` is not a PQN
curriculum; adding it now would move board/population/food variables together
and obscure attribution.

### 3. Reward shaping may optimize a different quantity from the gate

The learner's version-2 shaping is potential based:
`gamma * Phi(s') - Phi(s)` with a length potential, plus death and kill terms.
The PQN learner uses `gamma=0.997`, while the promotion score is undiscounted
full-horizon mass integral over 5,000 frames. Potential-based shaping can
preserve an appropriately discounted control objective, but it does not prove
alignment with this finite undiscounted evaluation target. At this gamma, a
reward 5,000 frames away has negligible direct discounted weight.

This is an objective-alignment hypothesis, not a discovered defect. The next
reward study must use new, powered, paired seeds and report the raw gate metric
alongside reward-term accounting, deaths, survival, kills, and mass. It should
change one objective component at a time. The conceptual grounding is
[potential-based shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf);
the current terms and constants live in
[`reward_events.py`](../../../src/core/reward_events.py).

### 4. State representation remains open, with bounded probes before a new model

The H1-A study shows a genuine limited observability boundary: hiding remote
enemy heading can alter later observations. It did not find a controlled
decision-relevance effect on holdout finite returns, and explicitly does not
measure natural alias frequency or representation sufficiency. It is therefore
evidence to keep testing spatial state, not evidence to add a GRU or discard
the raster encoder.

The immediate visible-cue probe is intentionally lower-level: use frozen,
reachable H1 snapshots and no RL updates to test whether the tactical convolution
path can represent visible food direction and local obstacle labels. It must
hold out entire worlds and persist labels, splits, predictions, gradients, and
dead-ReLU statistics. It is a lower bound on encoder accessibility, not a
success criterion for the full control problem. Later candidates are a
decision-relevance corpus targeted at natural ambiguous states, action-value
consistency checks under enemy-slot permutations, and only then memory or
equivariant-encoder ablations. Useful theoretical references are
[DeepMDP](https://proceedings.mlr.press/v97/gelada19a.html),
[bisimulation metrics](https://proceedings.mlr.press/v97/fu19a.html), and
[group-equivariant convolution](https://proceedings.mlr.press/v48/cohenc16.html).

### 4B. Other audited transition aliases need action-relevance witnesses

The observation-contract audit found several specific omissions that are real
transition distinctions but are not yet reasons to add features. Remote food
class is painted differently only in the tactical footprint; strategic maps and
nearest-food scalars merge ambient and corpse food even though their later
maintenance behavior differs. Enemy boost burn phase is not fully observed:
the hero exposes its own phase, while opponents expose only a boosting bit and
prediction; an identical current hero observation can therefore diverge after
an enemy boost burn. Body TTL is segment rank rather than absolute vacancy time
under fill-in and enemy boosting. The resolved action mask is a safety-shielded
legal/intersection-advisory contract, not a prediction of all simultaneous
enemy actions.

Treat these as hypotheses with separate fixed-RNG, same-observation witnesses:
a distant ambient/corpse-food toggle; a tail-crossing enemy boost-phase pair;
and a k-step vacancy pair. For masks, measure false-positive and false-negative
rates against taped opponent actions instead of calling the advisory component
exact legality. Each witness must first show a relevant action or return
difference before an observation channel, history window, or recurrent model
is proposed. World size is omitted from tensors but bound by the v3
effective-world digest, so defer a size scalar unless cross-world training is
introduced.

### 4A. Exploration diagnostics should separate masks from learned behavior

The fixed-opponent probe uses epsilon from 1.0 to 0.02 over 30,000 useful hero
steps. This produces substantial intended exploration during the 50,000-step
budget, but action draws are uniform over a row's *valid* actions, not over the
six global actions. Existing action entropy combines greedy and exploratory
choices across different mask cardinalities; it can therefore report apparent
collapse when only one action is legal, or hide a greedy-policy collapse behind
epsilon noise. Augmentation can further alter the sampled-action measurement.

After a successor diagnostic, a no-SGD CPU audit can replay frozen
initial/final checkpoints on the same states and record legal-action counts,
exploration opportunities and random decisions, greedy argmaxes, legal top-two
Q margins, greedy-action churn, and the opportunity-weighted random-action
expectation. That distinguishes mask-forced concentration, epsilon-hidden
greedy concentration, and an RNG/coverage issue without claiming deep
exploration. The finite-horizon/discount implication is also worth retaining:
with gamma .997 the effective horizon is about 333 frames, while the reported
score is undiscounted 5,000-frame mass integral. This is a real mismatch to
measure, not proof that gamma is wrong. [Pardo et al.](https://proceedings.mlr.press/v80/pardo18a.html)
and [Tang et al.](https://proceedings.mlr.press/v139/tang21b.html) motivate
time-awareness and explicit treatment of training/evaluation discount choices.

NoisyNet, bootstrapped heads, RND, or a horizon curriculum remain hypotheses,
not current recommendations. Each changes exploration, model/checkpoint
contracts, or the effective world. In particular, shortening `max_frames`
changes the episode-progress feature and evaluation-world digest, so it cannot
be treated as a harmless short-horizon substitute for the strict 5,000-frame
profile.

### 5. Replay is a later, disjoint algorithm branch

The raster PQN contract is explicitly replay-free. Adding replay to it is not a
sampler optimisation: it changes the algorithm to an off-policy Q learner and
needs new target semantics, n-step/terminal/mask metadata, checkpoint and
evaluation classification, and provenance. The safe first replay experiment is
a separately labelled raster Double-DQN implementation with a compact typed
CPU ring buffer, uniform sampling, explicit n-step fields, and a target
network. It must never serialize itself as a PQN checkpoint.

The first evidence-backed ablation should compare one use versus several uses
of fresh data only after a powered stationary baseline exists. [Double
DQN](https://arxiv.org/abs/1509.06461) is the cleanest source for the target
correction; [Fedus et al.](https://proceedings.mlr.press/v119/fedus20a.html)
motivates treating replay ratio as a measured design variable. Rainbow, REM,
and high-update methods such as DroQ bundle enough additional choices that they
are poor first explanations for an unproven raster learner.

### 6. PPO is a useful independent control, not a default replacement

[PPO](https://arxiv.org/abs/1707.06347) is a credible future control because it
uses an on-policy categorical actor and critic rather than Q(lambda) targets.
For this action space it must retain the exact resolved action mask both at
sampling and optimization, store masked old log probabilities, and test GAE at
death, truncation, reset, and support-boundary cases. This is supported by the
analysis and reference implementation for [invalid-action masking](https://arxiv.org/abs/2006.14171).

It is not a small trainer substitution. A qualified PPO branch requires a
separate actor-critic head, trainer, checkpoint head contract, CLI, evaluator
and serving dispatch. It must not call logits `q_values`, and its candidate
loader needs the same hash-before-load and semantic-contract protections as
PQN. There is no current evidence that PPO wins this game.

### 7. Evaluation design must match claim strength

The current two-seed study permits descriptive within-seed paired deltas only.
Four worlds reduce measurement noise but do not create four independent learned
policies. The only legal v3 fast evaluation profile is
`promotion_v2_watch_rect` at 5,000 frames; a 400- or 1,000-frame shortcut is
not equivalent. `run_simd_eval` evaluates one hero specification across its
given worlds, so each independently initialized arm must retain its own
initial baseline and use serial calls.

For an advancement study, freeze a minimum effect, new training seeds,
opponent mixes, primary metric, and analysis before training. Use pilot variance
only to plan a later budget; do not turn a two-run variance estimate into a
significance result. Relevant guidance includes [Colas et al.](https://arxiv.org/abs/1806.08295)
and [Patterson et al.](https://www.jmlr.org/papers/v25/23-0183.html).

### 7A. Apex comparison is feasible only as a labelled diagnostic today

The live tournament path can load a raster hero and a `vector61` Apex baseline
under one named profile and identical per-seed rosters. The fast SIMD evaluator
cannot evaluate `vector61`, so it is not a cross-stack engine. The narrow
three-frame raster v3 live/SIMD trace parity and checkpoint attach paths are
already covered, but the direct raster-candidate/vector-baseline CLI fixture is
v2. Add a v3-plus-vector named-profile fixture before relying on a full live
diagnostic.

For a first cross-stack diagnostic, additionally require no advisory-empty
action decisions. Apex's legacy safe-action fallback and v3's resolved-mask
fallback otherwise have different support semantics around that edge case. The
result would assess a deployed policy interaction, not isolate algorithms:
Apex and PQN differ in representation, simulator/trainer, reward history, and
training distribution. A fair algorithm claim needs matched fresh runs under
one mechanics/reward/opponent/environment-interaction contract, with wall and
energy reporting; a 2-by-2 algorithm-by-representation isolation does not yet
exist. Apex remains unchanged unless the later shared live promotion authority
is satisfied.

### 8. Measure the CPU-to-MPS pipeline before optimizing it

A source-level pipeline audit found three plausible costs but performed no
benchmark: `_current_obs()` rebuilds NumPy observation inputs and rasters at
each rollout step; corrected-v3 resolved-mask construction performs substantial
Python set/list work; and the MPS acting path transfers actions and masks back
to NumPy. The exact-coverage permutation itself is likely minor by comparison,
but numeric-recovery snapshots also copy model and optimizer state to CPU and
must be measured before being relaxed.

The next performance work is a profiling-only, fixed-seed harness around one
rollout and update. It should separately record simulator step, resolved-mask
construction, input extraction, raster featurization, host/device transfer,
acting forward, target computation, SGD, recovery snapshot, peak RSS, and MPS
driver memory. Every timing change needs a matched action/mask/raster-digest
parity check. This is a throughput investigation, not evidence that an
optimization is correct or that per-environment autoreset improves learning.

## Execution waves and parallel ownership

The numerical lane remains serialized to protect unified memory. Research,
fixture, documentation, review, and raw-artifact audits can proceed in
parallel because they do not run learners.

| Wave | Parallel roles and sole ownership | Gate before the next wave | Outcome that is allowed |
| --- | --- | --- | --- |
| 0: evidence and contracts — complete | Research owners maintain this portfolio and source notes; reviewers inspect the phase implementation and runner; resource owner enforces one MPS learner, CPU 2 threads, RSS/MPS/available-memory caps | Fixed witness, committed Watch contract, focused tests, CPU non-slow regression, CPU/MPS five-update receipts, and independent audit all complete | A qualified experiment launch with an explicit phase boundary, never a model claim |
| 1A: representation accessibility | Probe owner: artifact-local visible-cue driver and frozen H1 snapshots; verifier: labels/splits/oracle audit | Attempted: one seed completed and one hit its cap; held-out food accuracy/macro-F1 was 0.593/0.530 for the completed seed | `RESOURCE_CAP / INCONCLUSIVE`, not an encoder conclusion |
| 1B: elementary PQN learnability | Experiment owner: artifact-local fixed-anchor driver; verifier: terminal receipts, coverage, numerical checks, raw evaluation analysis | Freeze a successor protocol after Wave 0; then two valid 50k arms, each with its own initialization and complete 5k-frame final/initial rows | `DIAGNOSTIC_ADVANCE_ONLY` or `INCONCLUSIVE_NOT_ADVANCED` |
| 2: powered stationary baseline | Statistician freezes a larger new-seed study; execution owner serializes runs; evaluator owner preserves profiles/mixes; reviewer audits before analysis | Predeclared effect, adequate new seeds, all data retained, no adaptive seed substitution | Evidence about one frozen corrected-PQN recipe |
| 2P: pipeline measurement | Performance owner owns a profiling-only artifact; parity verifier owns fixed-state digest/action comparisons | Complete per-phase timings and exact parity before changing a hot path | A prioritised optimization proposal, never a learning result |
| 2E: exploration attribution | Analysis owner owns a no-SGD frozen-checkpoint audit; mask verifier owns support/cardinality checks | Legal opportunities, epsilon decisions, greedy margins, and action churn reported before a new exploration mechanism | A local action-coverage diagnosis |
| 2X: cross-stack diagnostic | Live-evaluator owner adds a v3-plus-vector fixture; comparison owner runs a labelled no-gate diagnostic; reviewer checks advisory-empty rows | Profile and roster parity, checkpoint/serving compatibility, no advisory-empty decision | A deployment-interaction observation, never a fair algorithm comparison |
| 3A: reward objective ablation | Reward owner changes exactly one term/discount/objective element; evaluation owner measures reward accounting and gate metrics | Powered stationary baseline first; paired new seeds; identical world and opponent contracts | Evidence about objective alignment, not a general algorithm winner |
| 3B: raster replay Double-DQN | Replay owner gets separate files and checkpoint algorithm id; buffer owner owns typed storage; verifier owns target/n-step/mask/terminal tests | Explicit off-policy contract, no PQN loader confusion, baseline held fixed | A distinct replay-Q experiment |
| 4: masked PPO control | PPO trainer/head owner; checkpoint/CLI owner; serving/evaluator dispatch owner; verifier owns mask/logprob/GAE parity | All contracts, independent tests, and a frozen control recipe | A controlled algorithm comparison |
| 5: self-play and curriculum | League owner reintroduces snapshot pool with exposure telemetry; curriculum owner uses one measured difficulty axis at a time | Stationary learner signal replicated first; opponents and curriculum schedule frozen | Robustness/league evidence, not an explanation for a baseline failure |
| 6: promotion | Gate owner runs the shared paired multi-mix authority; serving owner performs live-path spot checks; independent reviewer audits rows and artifacts | Candidate passes repaired gate, anchors, behavioral probes, and serving parity | Promotion decision only through the shared gate |

Waves 3A and 3B are deliberately disjoint code packages and can be developed
in parallel after Wave 2 design is accepted; their numerical evaluations still
run one at a time. PPO overlaps checkpoint, evaluator, and serving contracts,
so its implementation should follow the replay branch rather than be developed
against moving loaders. Self-play is last because it changes the data
distribution even when code is correct.

## Superseded or unexecuted proposals

Several early research notes proposed 100,000–200,000 all-hero runs, common
initial weights, `--no-self-play`, three training seeds, or 1,000-frame
evaluation. They are planning material, not the active protocol and have not
been run as current evidence. In particular:

- all-hero `--no-self-play` co-adapts six live policies and is not the present
  stationary-opponent baseline;
- shared initialization would reduce the independence of a later two-arm
  diagnostic, so successor arms should be independently initialized;
- 100k/150k/200k budgets exceed the deferred LP1 diagnostic, whose successor
  target is 50k useful hero transitions per arm;
- v3 evaluation requires the strict 5,000-frame profile, not a shorter
  proxy; and
- an official-PQN or PPO result has not been produced.

Preserving these abandoned proposals is useful because it explains why a later
artifact differs. They must be labelled superseded rather than rewritten into
the record as executed work.

## Practical stop conditions

Stop a future diagnostic and publish its terminal artifact if a numeric
tripwire fires, an initial/final checkpoint is missing, full-coverage counters
disagree, the fixed-policy identity/exposure differs, source/config/profile
closure fails, resource limits are crossed, or the parent cannot confirm child
exit. Do not replace a failed seed or tune after reading partial results.

If two successor arms are both healthy and their paired final-minus-initial
means are positive without a negative greedy-food mean, the sole permitted
conclusion is that a larger stationary baseline is worth designing. Any other
result is inconclusive and directs the next decision toward the recorded
failure mode—implementation, objective, representation, or exploration—rather
than toward an unsupported algorithm switch.
