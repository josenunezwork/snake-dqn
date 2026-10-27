# TRAIN-01 — GPU-resident long-horizon PPO development package

## Outcome and boundary

TRAIN-01 is the first real **simulator-D development** trajectory built on the
`sandbox_a_resident` A-3 lineage.  Its purpose is to produce development
checkpoints from a fully trainable B8-initialized actor under a task and opponent
distribution shaped like the eventual Mac evaluation.  It is not a gate,
registration, release, serving change, F5 input, or an edit to
`research/colab_cli`.

The intended operator already knows the adopted A100 sandbox rule, the qualified
dispatch path, B8/ego2s checkpoints, and the difference between a development
proxy and a Mac strict gate.  The successful outcome is one finite, resumable
simulator-D run with immutable exports and enough provenance to nominate at most
one frozen checkpoint for a separately authorized Mac crossing.  It does not
establish that any checkpoint is eligible for release.

This guide implements the engineering prerequisites in Astra's
`GPU_TRAINING_PLAN_20261027.md` under the authority in
`docs/research/governance_a100_sandbox_20261026/ADOPTED.md`.  Astra is the
primary recipe source.  Argon's 50/25/25 roster proposal is recorded as a useful
alternative, but TRAIN-01 follows Astra's balanced three-mix design: the fixed
E1024 allocation is divided as evenly as possible among `scripted`, `frozen`,
and `mixed` worlds.  Equal-mix learning accounting prevents a longer-lived mix
from silently becoming the objective.  A future run that selects 50/25/25 is a
different frozen recipe and needs its own controls, manifest, and evidence.

The adopted sandbox still requires registry-checked, domain-separated
`a100-dev/` inputs, a finite endpoint and a 2x runaway timeout.  It permits
development work without per-job ratification, while retaining the one-candidate
crossing, Mac strict gate, serving qualification, and owner release rules.  The
package reads no gate bank or saved gate trajectory and uses sandbox-local
adapters rather than changing certified R2/T4A engine bytes.

## What the package contains

The entrypoint is `research.sandbox_a_resident.train`.  Its adjacent modules
separate the contracts so an operator can audit them without treating A-3's
five-`greedy_food` timing assay as a training result.

| Area | Module(s) | Required behavior |
| --- | --- | --- |
| Gate-shaped opponents | `training_rosters.py`, `training_cpu_reference.py` | A local, fresh-seed descriptor for the three mix names, five ordered opponent slots, FRP3-s12 and champion identities, scripted `greedy_food`/`random_safe`, and the v8 wrapper (`lambda=8`).  It does not load a gate bank. |
| Natural training task | `training_task.py`, `training_sim.py` | Hero death is terminal for that life; opponents retain normal pressure/respawn behavior.  A live hero bootstraps at a rollout cut, while hero death and H5000 end the trace.  Resets are fresh sampled histories, never a short body with a fabricated nonzero clock. |
| Actor and critic | `training_model.py`, `training_learner.py`, `training_prep.py` | A B8-s3 initialized, fully trainable actor; frozen B8 reference logits for a light KL anchor; and a separate E8-style clocked PopArt critic.  The critic is training-only. |
| Durable operation | `train.py`, `training_units.py`, `training_resume.py` | Full-state immutable snapshots, inference exports, a verified resume ledger, telemetry, and manually constructed outcome-blind completion attempts. |
| Development monitoring | `training_proxy.py` | Deterministic masked-`argmax` candidate and cached-B8 H5000 runs on a fresh simulator-D proxy bank, reporting each mix and pooled paired deltas. |
| Admission and fixtures | `training_preflight.py`, `training_acceptance.py`, `test_training_*.py` | Offline packet/registry/control validation, CPU fixtures, roster checks, export/round-trip checks, and a small full-loop smoke path. |

`training_rosters.py` deliberately copies only the source-level roster descriptor
needed for new development roots.  The canonical gate source remains the
authority for member order, checkpoint hashes, wrappers, and any future gate
manifest.  Before dispatch, resolve those identities against the source at the
packet pin; do not substitute a historical roster merely because its mix names
match.

## Frozen TRAIN-01 recipe

The production control uses the values below.  Test-only controls may shrink
counts, but cannot qualify a real trajectory.

| Contract | Value |
| --- | --- |
| Actor initialization | B8 s3 SHA-256 `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`; copy the full actor/trunk and advantage head. |
| Opponents | E1024 permanent resident worlds, approximately one third per `scripted`, `frozen`, and `mixed` mix.  Each world has hero slot 0 plus five fixed opponents.  Learned slots alternate FRP3-s12 SHA-256 `eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723` and champion SHA-256 `43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93`, with the v8 wrapper. |
| Horizon and objective | Natural H5000 lives, no artificial truncation.  Reward is living logical hero mass divided by 1000; dead frames contribute zero.  `gamma=1` and GAE `lambda=.995`. |
| Capacity | `max_capacity=10400` and matching body storage.  Overflow is an error, not an implicit body-length cap. |
| PPO row shape | 256 ticks per update: `1024 × 256 = 262,144` hero rows.  The finite actor endpoint is 384 updates, or exactly `100,663,296` actor-learning hero transitions.  Preparation, proxy, and opponent actions do not inflate that counter. |
| Critic | Separate E8-style encoder receives the actor observation plus remaining-horizon input and uses PopArt.  The actor is frozen during critic preparation, then remains fully trainable during PPO. |
| Actor regularization | A frozen B8 reference supplies a light masked KL anchor.  The export is one actor only; it never invokes the critic or reference network at inference. |
| Snapshot/export cadence | Every 16 PPO updates (`4,194,304` actor rows), at the calibrated/pre-actor boundary, and at the final boundary.  Each snapshot includes complete resume state and has a small inference export. |
| Proxy cadence | At update 0 and every 64 updates, run H5000 deterministic `argmax` proxy evaluation with 96 worlds for each mix, paired against cached B8 from the identical proxy engine/configuration/roster identity. |

The actor begins as a policy by calibrating B8 advantage logits before training.
Calibration uses a fresh 10,000-state B8 development bank with five equal
length-bin quotas and may inspect no more than 5,242,880 rows to fill it.  The
chosen temperature is then frozen into the initial actor head.  It is not a
claim that PPO logits are calibrated Q values.

Critic preparation uses complete natural lives from a smaller E24 collection
cohort, independent of PPO's E1024 resident allocation.  It may use no more
than 8,388,608 rows and must have at least 2,000,000 fit rows, 40 complete
held-out lives, 160 held-out rows at length at least 800, and eight completed
held-out lives per mix.  The preparation report must meet its recorded
late-body support and quality gates before actor updates begin.  Historical E8
weights or an earlier explained-variance result do not waive this requirement.

## Task semantics and CPU evidence

The frozen task is rectangular **mechanics v1**, as explicitly requested by
Astra P0 and by `t34_round0/bank.py`'s profile descriptor. The historical
`promotion-v2-watch-rect` base label in that descriptor does not make this the
immutable production evaluation profile: `src/evaluation/protocol.py` defines
that profile with mechanics v2. TRAIN-01 follows the primary plan's v1 choice;
no production-profile identity is claimed. Opponent Vector61 normalization is
explicitly frozen to the gate's length normalizer 150, 16 sectors, danger
distance 30, boundary danger enabled and 61 inputs, without ambient globals.

The local task adapter exists because A-3 was a bounded resident assay, not a
long-horizon learning environment.  It builds actual worlds through the engine's
sandbox-local independent-slot sampling path, keeps an asynchronous per-world life age, and
resets only the completed hero world.  Opponents may respawn after their own
deaths so arena pressure remains present.  H5000 is a real terminal cutoff:
there is zero bootstrap at hero death or at frame 5000, and a live collector cut
alone uses the critic bootstrap.

The adapter uses a mixed counter stream for food and independent spawn slots;
CPU MT reset rejection and device sampling are different distributions. Before
substantial training, the runtime records an E32 B8 256-frame early-life prefix
per mix on each engine, including spawn spacing, food correlation, mass, death
causes, food consumption, respawns and length occupancy. Natural tasks remain
H5000; a complete first-life screen is separately available with horizon=5000.
As a dev-sandbox engineering-screen change requested on 2026-10-27 after Argon's
review, terminal_rate_per_1k_alive_frames, starvation_share and collision_share
are compared only when **both arms have at least 16 terminal events for that
mix**. Otherwise each comparison records `insufficient_support` and cannot fail
the screen. This support requirement applies to either supported screen horizon.
The 25% relative-gap threshold remains unchanged for all supported comparisons
and for spawn spacing, initial food, mass integral and alive fraction regardless
of death counts. Death rate uses recorded nonterminal alive frames as exposure;
collision share includes wall/self/head/body codes 1-4. These v1 engines have no
starvation terminal mechanism (starvation affects rewards only), so the recorded
starvation share is zero. Relative differences above 25% are descriptive
material-gap flags. They are explained as simulator-D pretraining
with reduced transfer confidence, never as a Mac gate or equivalence result.
Natural resets retain timed host work for canonical SHA roots and CPython-
compatible `random_safe` tape regeneration; policy actions and learning remain
on the device. A-3 throughput does not price this changed path.

The E24 critic-preparation cohort bounds complete-life CPU staging to 120,000
open rows. Every generated world row, including unused fit/holdout slots, counts
toward its cap. E24's last full tick is 8,388,600 rows; the remaining eight rows
are not collected. All completed fit-life rows are trained in minibatches.

The long-body configuration must survive bodies beyond 1024, ring wraparound,
and late-game H5000 states.  The capacity tests are therefore safety checks for
the adapter, not evidence that an arbitrary real-game length distribution has
been reproduced.

CPU checks are intentionally narrower than a closed-loop identity claim.  The
acceptance fixtures replay forced actions from an explicit common state and
compare discrete mechanics, vector carry, advisory masks, and v8 actions.  The
roster/reference fixtures use short horizons to confirm the CPU construction
against the local descriptor.  Before a substantial run, conduct the separate
fresh-root B8 distribution screen described by `training_acceptance.py`: at
least 32 worlds per mix on each engine over a 256-frame prefix of H5000, with spawn spacing,
food/refill, collisions, respawns, mass, and occupancy reported. The reference
keeps CPU world mechanics and feature construction; in production its B8 and
frozen-opponent neural forwards use CUDA (batched opponents). Inactive reference
worlds stop rasterization, and the screen stops once all first lives terminate,
with mass and alive fraction divided by the requested screen horizon. Terminal-frame resets contribute no
opponent-respawn or food events.

Neither test proves CPU/GPU full closed-loop bit identity.  Counter RNG and
neural tie behavior are known boundaries.  A failed common-state fixture or a
material distributional gap is an engineering stop: diagnose it, label the
attempt simulator-D, and do not describe proxy results as Mac-gate evidence.

## Model, export, and round-trip expectations

`B8Actor` keeps the deployable ego2s-bv8 actor architecture distinct from the
separate `ClockedPopArtCritic`.  All actor convolution, fusion, and action-logit
parameters receive PPO updates after the preparation boundary.  The B8 reference
is frozen and is used only to initialize/calibrate the actor and calculate the
masked KL term.  A frozen-reference-plus-residual inference path is outside
TRAIN-01.

The exporter writes the learned actor into the existing ego2s checkpoint format.
It strips compilation wrapper prefixes, omits critic/optimizer/reference state,
preserves the six-action order and observation metadata, and maps logits into a
dueling-compatible exported actor.  Dueling's statewise offset may preserve
masked greedy action choices without making the PPO logits calibrated Q values.
Do not use the exported output as a value/confidence estimate in the inspector.

The required round-trip tests cover zero-update B8 greedy agreement on fresh
development observations, legal masked-action identity and logit-difference
behavior through export, and inference through the existing checkpoint loader.
They record close top-two action margins for diagnostic use.  They do not promise
universal Mac/CUDA bit equality, and they do not modify a served checkpoint pin.

## Resume, exports, and monitoring

Each snapshot is an immutable completion unit.  It records actor, critic,
optimizers, PopArt state, collector/task state, RNG state, current iteration,
metrics, frozen control identity, and provenance.  Export sidecars and completion
markers publish after their payloads.  The Mac-side verified ledger accepts only
the matching allocation namespace and complete hash-bound units; it skips exactly
those verified units on a successor.

There is one original dispatch plus at most two **manual**, outcome-blind
completion attempts after a verified mechanical stop and missing work.  A
successor gets a fresh job ID, manifest, output directory, and receipt while
retaining the original evidence.  Calibration-only or failed-preparation state is
not a valid full-PPO resume.  A checkpoint trend, proxy result, or ordinary
reward curve is never a reason to construct a completion attempt.

The proxy is development feedback.  It uses deterministic masked `argmax` for
both candidate and cached B8, the identical local roster identity for each arm,
and new `a100-dev/` roots outside training and preparation.  It must report
per-mix and equal-mix pooled mass deltas, life/death behavior, boost use, and
long-body occupancy.  A proxy confidence interval describes this simulator-D
sample only; it says nothing by itself about transfer to the Mac strict gate.

At trajectory end, selection may nominate at most one frozen actor checkpoint
under the adopted crossing rule.  Crossing requires a new registration, judge
process, fresh salt-derived paired Mac bank, publish-before-read discipline,
Mac strict/serving qualification, and the owner's release decision.  TRAIN-01
does not perform any of those actions.

## Operator sequence

1. Create a clean detached checkout at the packet pin.  Preserve any active
   worktree edits; do not run from a dirty shared development checkout.
2. Build the committed packet around the frozen control, input registry audit,
   empty verified ledger, finite endpoint, 2x timeout, and excluded-study list.
   The packet must use `research/sandbox_a_resident/train_job.sh`; it must not
   modify or bypass the qualified transport.
3. Run the package preflight against that packet.  It validates the control,
   source pin, manifest schema, registry allocation audit, and scoped tests:

   ```bash
   nice -n 15 ./venv/bin/python -m research.sandbox_a_resident.training_preflight \
     /absolute/path/to/train01-packet
   ```

   A successful invocation prints `PASS_TRAIN01_PREFLIGHT`; that text is an
   expected command result, not a result asserted by this document.
4. Run the packet's `tools/ops/dispatch.sh --dry-run` command, preserving its
   output as operational evidence.  Dispatch only the immutable packet after
   both checks have passed.  This guide intentionally does not invent a runtime
   pin, provider command, packet path, or a completed dry-run receipt.
5. On a mechanical stop, verify the terminal receipt and create a successor
   through `training_completion.py`; do not hand-edit the ledger or recompute an
   already verified unit.  At final completion, archive the summary, snapshot
   exports, ledger, preflight evidence, and dispatch receipts together.

## Verification map and common failures

The package's narrow test families are `test_training_rosters.py`,
`test_training_task.py`, `test_training_model.py`, `test_training_learner.py`,
`test_training_proxy.py`, `test_training_resume.py`, `test_training_prepare.py`,
`test_training_acceptance.py`, `test_training_compile.py`,
`test_training_cpu_reference.py`, `test_training_runner.py`, and
`test_training_units.py`, `test_training_completion.py`,
`test_training_cuda_semantics.py`, and `test_training_production.py`.  The preflight runs the scoped package suite serially
with CPU thread limits and `nice -n 15`; it does not claim CUDA qualification.
For explicit CPU-hosted CUDA API-contract coverage, run the same command with
`--cuda-semantics`. It exercises relocated RNG and roster-state doubles, meta-device
critic baseline placement, and CUDA inference routing; it does not execute CUDA
kernels. These checks also run in the ordinary full package preflight.
The full suite and the actual preflight/dispatch evidence belong in the packet's
verification record once executed.

| Symptom | Meaning and operator response |
| --- | --- |
| Roster identity or checkpoint hash mismatch | Stop before collection.  Reconcile the source-level gate descriptor and packet inputs; never replace an identity with a similarly named checkpoint. |
| Preparation quota/support/quality failure | End the attempt before actor PPO.  Keep the preparation evidence; do not resume it as a trained actor. |
| Body-capacity overflow, illegal action, non-finite state, malformed resume identity, or more than 90% of a complete 100k action window in one action | Stop the trajectory and preserve the failing snapshot/telemetry.  These are engineering failures, not a basis for automatic retry. |
| Cached B8 proxy cannot be paired to the same engine/configuration/roster/hash | Invalidate the cache and rebuild the baseline on the matching proxy specification.  Do not compare across simulator or roster identities. |
| CPU common-state mismatch or material B8 distribution gap | Stop and diagnose the sandbox adapter.  Do not call the result CPU parity, a gate, or a release candidate. |

## Estimate and uncertainty

The allocation estimate is **4–8 A100 allocation-hours**, with an **8-hour base
and 16-hour effective 2x runaway timeout**.  This is low-confidence planning
information because the configuration changes the collection task, roster,
long-body memory pressure, optimization work, and periodic H5000 proxy from the
previous timing assays.  It is not a measured current CUDA runtime, price, or
authorization to revise the frozen endpoint or timeout.

## Source evidence

- Astra, `GPU_TRAINING_PLAN_20261027.md`: primary task-aligned PPO, E1024/T256,
  B8 actor/E8 critic, natural H5000 mass objective, preparation, export, and
  proxy requirements.
- Argon, `gpu-training-plan-01/answer.md`: alternate 50/25/25 roster and cheap
  H5000 proxy rationale, retained here as a documented alternative rather than
  silently merged into Astra's recipe.
- `docs/research/governance_a100_sandbox_20261026/ADOPTED.md`: sandbox scope,
  isolation, finite-endpoint, registry, one-candidate crossing, Mac, and owner
  boundaries.
- `src/scripts/tournament_eval.py`: current mix construction and H5000
  evaluation semantics; `training_rosters.py` and `training_acceptance.py`:
  sandbox-local projection and the limited CPU evidence contract.
