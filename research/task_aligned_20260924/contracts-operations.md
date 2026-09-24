# Task-aligned study operations contract

Implementation only. No numerical work, resource preflight, model load, or fixture execution was performed while preparing this harness. The root owner must finish independent source review, bind every imported source, and freeze a new prospective intent before admission. No previous study is reopened.

## Admission interface

`runner.py --intent /absolute/intent.json --root /absolute/new-output-root` consumes a JSON object with:

- `schema_version: 1`, `study_id`, `admitted: true`, `frozen_utc`, `deadline` (timezone-aware UTC strings).
- `output_root` exactly matching the CLI root, `repo`, `python` (absolute venv executable), `device` (`cpu` or `mps`), `handoff_seconds: 120`.
- `supervisor_source`: the reviewed existing Mac supervisor source. Only its helper functions are imported; its `main()` and old evidence root are never used.
- `source_sha256`: absolute-path-to-SHA256 mapping for the runner, audit, stage entrypoint, supervisor, monitor, and complete imported runtime/config closure. The root freezer owns closure completeness. Include `src/scripts/pqn_correctness_diagnostic.py`, whose expected monitor hash is already checked by the reused helper.
- `evaluation_source`: absolute source path bound in `source_sha256`. `evaluation` embeds the frozen evaluation settings with `training_world_seeds: []`; its intent hash authenticates those settings.
- Ordered `stages`, first `qualification`, last and only `audit`. Each contains `id` (safe component), `kind`, `argv` array, `cap_seconds`, `counter_caps`, optional `expected_report_fields`. Exactly three `training` stages contain distinct `lineage` identities. Accepted kinds are `qualification`, `training`, `evaluation`, `analysis`, `audit`.
- A stage may freeze `device: cpu` or `device: mps`, overriding the intent default in that child's `SNAKE_DQN_DEVICE` environment. This permits CPU evaluation and MPS qualification/training without changing resource limits. The stage CLI must apply the same selection when configuring its runtime.

Each argument supports only the explicit Python format placeholders `{root}`, `{out}`, `{intent}`, `{heartbeat}`, `{stage_id}`. The runner supplies a new `output` directory inside each stage; children must not require that directory to be absent. The executable must equal `python`, and the entrypoint in argv slot 1 must be frozen in `source_sha256`. No shell is involved. The root stage CLI owns learning and evaluation orchestration; this controller does not load their models.

The frozen total of stage caps may not exceed 86,400 seconds. The proposed caps are qualification 1,800; three training stages 25,200 each; six model evaluations 900 each; anchors 900; analysis 600; audit 900: 85,200 total. A 24-hour prospective window leaves 1,080 seconds beyond these caps and the 120-second handoff. Every launch and post-lock admission reserves the entire remaining plan plus handoff. No partial restart, skipped failed stage, suffix retry, or outcome-driven extension exists.

## Common report and counters

Every child writes `{out}/report.json`:

```json
{
  "schema_version": 1,
  "study_id": "root-frozen-identity",
  "stage_id": "root-frozen-stage",
  "status": "PASS",
  "counters": {
    "native_frames": 0,
    "hero_transitions": 0,
    "model_forwards": 0,
    "checkpoint_loads": 0,
    "optimizer_updates": 0
  },
  "artifacts": [{"path": "relative-new-output.json", "sha256": "..."}]
}
```

`PASS` means the stage executed and satisfied its operational contract; scientific rejection is a separate decision field. All counter values are nonnegative integers, never booleans. `counter_caps` must include the five common counters and may include extra instrumented counters. Counts describe physical operations, not nominal tensor capacity. `optimizer_updates` counts optimizer steps, not rollout updates. Checkpoint hashing is not a load; tensor deserialization is. No aggregate row count is divided by minibatch size to invent a step count.

The learning/evaluation owners must instrument every counted operation and document whether their forward counter counts invocations or evaluated rows. Report counter names alone do not verify instrumentation. Every new artifact is explicitly listed, rooted inside the stage output, hashed, and bound to its completion receipt. Historical payloads are neither enumerated nor deserialized by the controller or audit.

## Qualification and fixed dose

Qualification is an admitted numerical stage, including runtime schema fixtures and throughput measurement. It reports `runtime_contracts_pass: true`, `fixed_science_fits: true`, `estimated_science_seconds`, `conservative_cycle_seconds`, `fixed_overhead_seconds`, `forecast_per_lineage_seconds`, `shared_probe_lineage: 2026096101`, `selected_rollouts`, and `dose_sha256`.

Time 30 complete mixed-profile cycles (150 rollouts) on lineage 2026096101 only. Use the maximum measured cycle time plus measured fixed loading/saving overhead. Choose the largest multiple of five no greater than 39,060 satisfying `2 * (fixed_overhead_seconds + selected_rollouts / 5 * conservative_cycle_seconds) <= min(training caps)`. Stop below 7,815. This expression is `forecast_per_lineage_seconds`; its triple is `estimated_science_seconds`. Costs must be finite and the cycle cost positive. This is a forecast for the other same-architecture lineages, not three measured timings; workload variation remains a risk and an overrun stops the study. Behavioral outcomes cannot influence dose. Measure actual synchronized mixed-runtime work, including real valid minibatch and forward counts.

Canonical dose hash: SHA256 of compact sorted-key JSON (`separators=(",", ":")`, finite values only) containing `selected_rollouts`, `conservative_cycle_seconds`, `fixed_overhead_seconds`, `shared_probe_lineage`, and `profile_rollouts`, with profile keys `solo`, `S2`, `S6` and counts `dose/5`, `2*dose/5`, `2*dose/5`. The runner writes `qualified-dose.json` exactly once. Training consumes this file, never chooses a dose. Each training report repeats `lineage`, `selected_rollouts`, `completed_rollouts`, `profile_rollouts`, and `dose_sha256`; all must match. The endpoint is fixed rollout count, not 10 million actual hero rows. At most 39,060 × 256 = 9,999,360 scheduled slots per lineage, with actual valid hero transitions separately counted. The 20/40/40 mixture describes rollout count, not guaranteed hero-row weighting.

## Receipts, heartbeat, and audit boundary

The runner creates the frozen output root exclusively, then durable `started.json`. Per stage it creates `started.json`, post-lock `admitted.json`, `supervisor.json`, and only after validated natural code-zero exit, unchanged source, valid report/counters, and authenticated artifacts, `completed.json`. Receipt content and directory are fsynced. A partial receipt or missing completion is failure, never permission to resume. Existing CPU slot 1 and 2 locks serialize all child processes. The frozen monitor applies CPU threads 2, interop 1 through child environment, RSS 8 GiB, MPS driver 24 GiB, and available RAM at least 9.6 GiB. Children must enforce PyTorch thread settings and emit atomic JSON heartbeats more often than 20 seconds, with `monotonic` and actual `mps_driver_bytes` for MPS.

Each evaluation job wraps its per-profile cells in a common report with `evaluation_reports`, an ordered list of relative filenames also listed in `artifacts`. Batch counters equal the sum of the cells. Six learned-policy jobs contain three profiles apiece; the anchor job contains six cells. Analysis consumes these 24 cells in this stage/file order and emits the evaluator's `paired_decision` report.

Within an evaluation cell, totals equal episode counters plus `initial_counters` plus `shared_counters`. Only checkpoint loads may be nonzero in initial counters, and only actual batched model-forward invocations may be nonzero in shared counters. Shared frame, transition, load, or optimizer work is rejected. The auditor reconciles this independently. Optional forward-row counts are separate from these invocation counters and must not be substituted for them.

Every training stage seals `training-world-seeds.json` in its artifacts. Its `worlds` object contains `solo`, `s2`, and `s6`, each with 16 per-environment episode-seed lists. Evaluation adds the sorted union of these three authenticated lineage ledgers to the frozen template's `training_world_seeds`, changing no other field. Each evaluation batch binds its generated `evaluation-spec.json` as an artifact and names it through `evaluation_spec`; analysis embeds the same object. The auditor reconstructs this augmented specification independently from the authenticated training ledgers and rejects any mismatch. This is an actual-ID exclusion check within the declared namespace scope, not proof of unique physical geometry or all-history independence. Generated stage outputs belong to the receipt chain, never to the static source hash map.

`audit.py --intent {intent} --root {root} --out {out} --heartbeat {heartbeat} --stage-id {stage_id}` is the final supervised child. It checks ordered complete receipts, deadline admission, exact frozen commands, counters, caps, output hashes, and the sealed dose. It independently derives per-world metrics from saved frame columns, validates the exact case matrix with the frozen stdlib evaluator, and recomputes decision fields using its saved-record reducer. Shared reducer use is disclosed; this is not an independent reimplementation of every decision formula. It never imports the simulator or deserializes checkpoints. No tournament pass is implied.

The auditor cannot attest its own future completion. The runner validates the audit's natural exit, report, receipt, and final unchanged output chain before emitting `AUDITED_PENDING_ROOT_CLOSEOUT`. The root performs final handoff. Original failures persist as `INVALID_STOP` with `retry_authorized: false`. Source preparation checks do not replace later admitted runtime qualification.
