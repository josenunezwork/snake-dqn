# Single-child continuation evaluation

**Source implementation only; not admitted.** This is a new prospective comparison of
one fixed continuation against its source. It neither changes the completed task-aligned
study's `INCONCLUSIVE_SCREEN` nor replaces the separate champion challenge. It has no
promotion authority. The controller must freeze this protocol, the new module, inherited
pure helpers and exact runtime/source bindings before any evaluation.

## Fixed candidate and new worlds

Lineage `2026096102` is fixed before this continuation. The source is its completed global
update 35,613 endpoint. The child receives exactly 31,005 additional rollouts of the same
architecture and task recipe, ending at global update 66,618. Only that final child is
compared. No intermediate checkpoint, favorable world or different training seed may be
selected. Training qualification, training dose, resources and rollout accounting belong
to the separate controller contract; evaluation performs zero optimizer updates.

The controller seals both checkpoint digests before filling the final evaluation spec's
`policy_sha256: {parent: ..., candidate: ...}`. These labels refer to source and child,
not the original CZ mark-4608 parent. Learned policies use `lineage: 2026096102` and roles
`parent` or `candidate`. Shared anchors use `role: anchor`, `lineage: null`, and name
`greedy_food` or `random_safe`.

Each policy plays exactly 32 procedural native initial worlds in each profile, H3000:

| Profile | Opponents, in slot order |
| --- | --- |
| solo | None |
| food_pressure | Four GreedyFood, then one RandomSafe |
| mixed | Two GreedyFood, then three RandomSafe |

CPU float32 E32 serving uses the already qualified native adapter, its inherited input
restriction, Q scale, masks, normalizations, Watch preparation, terminal hero and respawning
opponents. `source_config` carries the exact inherited configuration; `contract_sha256`
is the canonical digest of the adapter's evaluation contract for that configuration.
All four policies share initial descriptors within every world/profile; later random
consumption and trajectories can differ by policy.

`make_world_seeds()` constructs IDs only, without constructing a game. For index 0–31,
it takes the first eight SHA-256 bytes, big endian, of the UTF-8 string
`task-aligned-continuation-20260924/held-v1/{profile}/{index}`. All 96 IDs must be distinct.
An overlap stops admission/evaluation without resampling.

The final spec contains `training_seed_namespaces`, actual `training_world_seeds`, and
`excluded_world_seeds`. The controller builds the exclusions only from the explicitly
named completed original `evaluation-spec.json` (held and training IDs) and the prior
preview intent's world IDs. New continuation training IDs also enter
`training_world_seeds`. Every new held ID is checked against both lists. Source files
must not enumerate historical raw trajectories. `novelty_scope` remains
`declared_seed_namespaces_only`; no exhaustive all-history geometry claim follows.
The separate champion challenge's own protocol and evidence remain unchanged.

## Physical scope and saved evidence

There are twelve cells: source and child each contribute three profiles, and two shared
anchors contribute three each. This is **384 episodes and at most 1,152,000 native lane
frames / valid hero transitions**. Learned controllers contribute at most 576,000 lane
frames, 18,000 batched model-forward calls and 576,000 evaluated model rows. Anchors make
no model calls. Each learned policy loads once across its three cells; total checkpoint
loads are exactly two. The controller groups profiles appropriately and zeros a reused
runtime's initial-load counters after its first cell.

Each case retains the inherited schema, initial descriptor and digest, full compact
`frame_columns` (`alive`, `mass`, `ambient_food`, `boost`, `encounter`), physical counters,
and aggregate metrics. The stream ends at hero death or H3000. Death and all subsequent
implicit frames contribute zero living mass. Mass and alive fraction use the entire
3,000-frame denominator; food counts ambient events rather than shaped reward. Encounter
means a pre-action enemy body within L1 distance 16 and counts only frames 17 onward.

Batched forward invocations appear only in shared cell counters, once per executed batch
step. They are never allocated once to every world. The learned shared count must equal
the longest observed episode; all anchors' shared model counts are zero. Per-world model
calls, loads and updates must be zero. All cases, counters, saved summaries and frame
arithmetic are validated before a complete cell is saved. Outputs use the inherited
exclusive atomic publisher and are not replaced or silently rerun.

## Prospective single-child decision

Compute child-minus-source deltas over the same 32 paired worlds, separately by profile.
Reuse the frozen two-sided marginal 95% Student-t interval, df31. The worlds do not create
additional trained-policy replications. There is one selected trained lineage, no
familywise/powered recipe-population claim and no causal factor attribution.

The ordered decision is:

1. **`NEGATIVE_CONTINUATION_SCREEN`** if the mass interval's upper bound in any profile
   is below minus 5% of that profile's source mean mass; or solo food's upper bound is
   below minus 5% of source mean food; or solo alive fraction's upper bound is below
   −0.02. These are explicit tolerances for this new study. In particular, opponent
   mass harm uses the new 5% margin, not the completed study's zero margin.
2. **`INCONCLUSIVE_EXPOSURE`** if, in either opponent profile, the source or GreedyFood
   hero anchor encounters an opponent after frame 16 in fewer than eight worlds.
   Child and RandomSafe exposure remain descriptive; no candidate cases are filtered.
3. **`ADVANCE_CONTINUATION_SCREEN`** if both opponent profiles have mean mass gain at
   least `max(0.10 * source mean mass, 1.0)` and strictly positive lower confidence
   bounds, and solo lower bounds establish noninferiority within 5% of source mean
   mass, 5% of source mean food and 0.02 alive fraction.
4. Otherwise **`INCONCLUSIVE_SCREEN`**. There is no additional futility branch and no
   requirement for other training lineages to pass this candidate-specific comparison.

Percent margins use the fixed profile's overall source mean, not per-world ratios. A
zero source food mean permits zero absolute food loss. Clear harm takes precedence over
insufficient exposure. Failure to establish noninferiority alone is not proof of harm.
Passing advances this continuation as a development candidate; it does not establish
superiority over the incumbent, change champion status or substitute for its challenge.

## Interfaces and integration

The module exports `EpisodeRequest`, `EpisodeTrace`, `EvaluationRuntime`, `FrameMetrics`,
`COUNTERS`, `METRICS`, `canonical_hash`, and `save_report` as unchanged pure-helper aliases.
It does not call the old CLI, old three-lineage validator/analyzer, or monkeypatch globals.
`make_world_seeds()` is a metadata-only helper. Public APIs are:

- `validate_spec(spec)`: requires the fully sealed single-lineage spec, source/child
  hashes, fixed endpoints, CPU E32, exact rosters, new IDs, exclusions and tolerances.
- `evaluate_cell(spec, policy, profile, runtime, out)`: requires an admitted external
  `run_batch` callback; returns and exclusively saves one complete 32-world cell.
- `validate_report(spec, report)`: rederives every saved case from physical frame
  columns and validates hashes, roster, world order, summaries and counter accounting.
- `analyze_reports(spec, reports, output_path=None)`: requires exactly twelve cells,
  consistent identities across profiles and identical initial states across all roles;
  returns the single-child decision and every paired metric without new model queries.

Spec fields additionally include `schema_version: 1`,
`study_id: task-aligned-continuation-20260924`, `lineages: [2026096102]`, `horizon: 3000`,
`evaluation_device: cpu`, `batch_worlds: 32`, `source_update: 35613`,
`candidate_update: 66618`, `continuation_rollouts: 31005`, and `decision` equal to the
module's `DECISION` mapping. Before the child exists the controller may hold a template,
but it must not present that template as a valid final evaluation spec.

Cell reports retain `kind: policy_evaluation`. Saved analysis uses
`kind: paired_continuation_decision`, `per_lineage: {"2026096102": ...}`, the exact source
report digests, zero analysis-operation counters, and separately aggregated evaluation
counters. Audit must independently reconcile receipt bounds and saved frame arithmetic;
reuse of this reducer must be disclosed rather than called independent statistical
implementation. Plot generation is delegated to the root and is outside this module.
