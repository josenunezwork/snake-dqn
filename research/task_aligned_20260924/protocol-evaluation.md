# Task-aligned package evaluation contract

This implements the prospective engineering screen chosen during the 24 September
review. It is a package-utility comparison, not a causal data/optimizer contrast,
recipe-population confirmation, architecture ranking, or promotion gate. Source
implementation and compilation do not admit numerical execution. The campaign
controller must bind this file, the evaluation module, the adapter, the final
specification, endpoint identities, and the permitted physical work before launch.

## Fixed scientific scope

Evaluate all three CZ `enemy_visible` update-4608 parents, lineages 2026096101,
2026096102, and 2026096103, and their three prescribed final candidates. The choice
of cohort is outcome-informed and made now. Do not choose an intermediate mark or
the favorable seed. CZ's old all-three learning/access failures and the old
enemy-dose continuation's failed retention remain unchanged. Reuse the published
CW/CX/CZ/old-dose results as historical evidence; this campaign does not repeat
their selected-food, 12-pose H256 or one-opponent screen.

Every new episode is one procedural native initial world, with H3000 and a
terminal hero. Opponents respawn under the adapter's Watch pre-move contract.
The exact CZ input restriction, scalar normalization, model output scale, masks,
movement, reward-independent physical measurements, and lifecycle must be bound
by `contract_sha256`. No stock-loader or cross-stack compatibility is presumed.
The adapter must preserve the same initial descriptor and RNG identity for every
controller on a given world/profile. Later policy-dependent RNG trajectories may
diverge; pairing does not assert identical later food locations.

Profiles are fixed: `solo` has no opponents; `food_pressure` has four GreedyFood
and one RandomSafe opponent; `mixed` has two GreedyFood and three RandomSafe
opponents. Opponent slot order is part of the spec. Each profile has 32 fixed,
distinct held world IDs, shared across policies within that profile. No forced
heading, guaranteed food, teacher-solvability selection, or rejection sampling is
permitted. GreedyFood and RandomSafe hero anchors run once per profile and are
shared across all learned lineages. Neither needs perfect survival for admission.

The frozen spec supplies distinct `held_seed_namespace` and
`training_seed_namespaces`, explicit `training_world_seeds`, and the three lists
in `world_seeds`. The validator checks actual held/training ID disjointness and
held profile disjointness; it imposes no 32-bit seed limit. Namespace-to-ID hashing
belongs to the frozen controller. `novelty_scope` must be
`declared_seed_namespaces_only`: neither novel geometry nor independence from all
historical analyst knowledge is claimed. Keep final held outputs concealed until
the candidate endpoints are sealed. Do not use them for training or selection.

## Physical bounds and measurements

There are 576 learned-policy episodes (three lineages × two roles × three profiles
× 32 worlds) and 192 scripted-anchor episodes: 768 total and at most 2,304,000
native hero frames. A separate qualification allowance of at most 32 episodes
must fit the root campaign's 800-episode cap; it has no scientific lineage. These
counts are ceilings, not throughput predictions. Root's serialized job receipts
must separately cap wall time, memory, checkpoint loads, and model forwards, and
reserve final analysis/audit and handoff time. No timed-out cell is automatically
rerun. Six learned jobs may each load once and process three profiles; scripted
anchors require no checkpoint loads. Count actual operations, not logical calls.

Mass is physical post-move logical mass while alive. Death-frame mass and all
subsequent implicit frames contribute zero. `mass_integral = sum(mass)/3000` and
`survival_fraction = sum(alive)/3000`; do not divide by frames survived. Ambient
food is a physical event count, not reward or total corpse-food consumption.
Boost is the physical boosted-frame flag. Encounter is pre-action minimum L1
distance from hero head to any live enemy segment ≤16; `encounter_frames` counts
only frames 17 onward. That is an exposure descriptor, not a causal mechanism.

Each saved case includes the full initial descriptor, its canonical SHA-256,
aggregate metrics, actual counters, and compact `frame_columns`: `alive`, `mass`,
`ambient_food`, `boost`, and `encounter`. Every column has one entry per observed
step, including a terminal death step. The stream stops at hero death or H3000.
This permits independent arithmetic verification and time-course figures without
replaying a game or querying a model. It does not save state vectors or fabricate
unobserved survivor frames.

## Prospective decision rule

Compute candidate-minus-parent paired deltas over the 32 world families separately
for every lineage/profile. Use two-sided marginal 95% Student-t intervals with
31 degrees of freedom. Do not pool lanes or trained seeds. No familywise or
training-population inference is claimed.

The practical opponent-profile mass margin is `max(0.10 × parent_mean_mass, 1.0)`.
The absolute one-mass-unit floor prevents a near-zero baseline from producing a
misleading large percentage gain. These are prospective engineering tolerances,
not power-derived detectable effects. Each of the six lineage/opponent-profile
comparisons must have point gain at least that margin and a strictly positive
lower confidence bound for `ADVANCE_PACKAGE_SCREEN`.

Additionally, every lineage's solo paired lower bounds must be at least minus 5%
of the fixed parent overall mean for mass and ambient food, and at least −0.02
for survival fraction. Use differences against these fixed mean-scaled margins,
not averages of per-world ratios. Zero parent food makes its allowable absolute
food loss zero. Opponent food/survival/boost changes remain visible in full.

The order is explicit:

1. `NEGATIVE_RELIABILITY_SCREEN` if a solo upper confidence bound is below its
   allowed loss, or an opponent mass upper bound is below zero. This clear
   observed harm takes precedence over uncertain exposure.
2. `INCONCLUSIVE_EXPOSURE` if any parent or the GreedyFood hero anchor encounters
   opponents after frame 16 in fewer than eight worlds in either opponent mix.
   RandomSafe and candidate exposure are descriptive; candidate deaths or choices
   are not used to condition away a bad policy result.
3. `ADVANCE_PACKAGE_SCREEN` if all six practical mass gains and all solo bounds pass.
4. `NEGATIVE_PRACTICAL_GAIN_SCREEN` if all six opponent mass upper bounds are below
   their practical margins, ruling out the specified gain on every sampled lineage/mix.
5. Otherwise `INCONCLUSIVE_SCREEN`; a missed gate alone is not proof of no learning.

These outcomes select the next development decision only. Three lineages remain
a screen. Promotion requires the unchanged separate tournament authority.

## Interfaces, saved reports, and execution boundary

`EvaluationRuntime(run_episode, contract_sha256, initial_counters)` contains an
adapter callback `EpisodeRequest(world_seed, profile, opponents, horizon) ->
EpisodeTrace(frames, initial_descriptor, counters)`. Frames are `FrameMetrics`
instances with the five fields above. Counters have exactly `native_frames`,
`hero_transitions`, `model_forwards`, `checkpoint_loads`, and `optimizer_updates`;
the last must be zero. Counters and flags must be Python integers and booleans.
Factory counters may count checkpoint loads only. The initial descriptor requires
`world_seed`, `hero`, `opponents`, `food`, and `rng_sha256`.

The runtime may additionally expose `run_batch(requests) -> (traces,
shared_counters)` for the 32 independent environments. Each trace retains its
own physical frames and valid hero transitions. Actual batched model calls belong
to `shared_counters.model_forwards`, with no invented per-world allocation; other
shared counters must be zero. Cell totals sum per-world counters, shared model
calls, and factory checkpoint loads. Batch width is not a count of model calls.

`evaluate_cell(spec, policy, profile, runtime, output_path)` saves one 32-world
report. For a three-profile job, attribute a real initial load to its first cell
and supply zero initial counters to the remaining cells. Root's job envelope lists
the three relative `evaluation_reports` paths, their hashes, and summed counters.
It owns heartbeat, deadline, resource limits, and stage identity. The optional
module CLI is single-cell only and does not replace that supervisor.

`validate_report(spec, report)` checks saved frame/aggregate arithmetic, schema,
world completeness, and counters without simulation. `analyze_reports(spec,
reports, output_path=None)` requires exactly 24 cell reports and checks identical
initial states across controllers, consistent policy identity across profiles,
and the full fixed matrix. The controller must additionally bind each reported
checkpoint digest to its frozen parent/candidate manifest. Missing, malformed,
or mismatched evidence raises an integrity error; it is not a scientific negative.
`status: PASS` means a report was structurally complete, not that its scientific
decision advanced. All JSON writes are exclusive and atomic; completed outputs
are never replaced. Audit may independently recompute arithmetic from these saved
columns, but cannot run the simulator or load checkpoints.
