# Sequential strict runner template (`sequential-strict-template/v1`, `/v2`, `/v2-pooled` and `/v3`)

Three template versions share this code. An intent declares which one it uses in
`template_version`. The spec's `band_policy` decides v1 or v2; `prepare --remote-config`
(opt-in) makes the intent v3 over either band policy:

| Version | `band_policy` | Bands |
|---|---|---|
| `sequential-strict-template/v1` | `block_at_stop` (default) | candidate mean vs calibration reference + offsets (`survival_bands`) |
| `sequential-strict-template/v2` | `paired_ni_at_stop` | paired survival noninferiority, judged at the qualifying look (`paired_survival_bands`); see [Adopting paired survival bands](#adopting-paired-survival-bands-template-v2) |
| `sequential-strict-template/v2-pooled` | `pooled_ni_continue` | survival band v2: pooled paired NI + per-mix catastrophic NI, repeated confidence bounds, judged at the qualifying look and every later look (a qualified run whose bands fail continues, `CONTINUE_BANDS`); see [Survival band v2](#survival-band-v2-template-v2-pooled) |
| `sequential-strict-template/v3` | any (kept in `execution.band_template_version`) | as v1, v2 or v2-pooled; the episodes run on RunPod serverless; see [Running the gate on RunPod](#running-the-gate-on-runpod-template-v3) |

v1 intents keep exactly their v1 content: the same intent keys, plan parameters, plan dict and
sha256, spec descriptor, calibration and look-receipt fields. Golden hashes in
`tests/test_sequential_strict_template_paired.py` pin this.

Status: **template, code and unit tests only.** Nothing in this folder has played a game.
It implements the runner, look barrier, look receipts and independent audit that
[`governance_amendment_sequential_gates_2026-10-02.md`](../../docs/research/governance_amendment_sequential_gates_2026-10-02.md)
lists as "not yet done". The decision functions are `src/evaluation/sequential_gate.py`
(method `strict-sequential-obf-bonferroni-v1`). The fixed-N packages
(`research/apex_veto_v{5,7}_strict_*`) are neither modified nor imported.

| File | Role |
|---|---|
| `sequential_runner.py` | `prepare` (intent), `run` (parent), `status`, `worker` (child) |
| `sequential_audit.py` | independent audit, standard library only, imports nothing from the repo |
| `example_spec.py` | instantiation template; its episode runner raises |

## How to instantiate for a candidate

1. Copy `example_spec.py` into the study package (e.g. `research/apex_veto_v8_strict_<date>/spec.py`)
   and fill in the six marked parts: `arm_identities` (computed from the files),
   `episode_runner` (installs the arm and calls `tournament_eval.rollout`), `build_row`
   (rosters / world identity), `excluded_seeds` (every earlier namespace; the template's
   raises until filled), `validate_record`, and the three required pre-registration
   documents: `protocol_path` (copy the section below), `oc_report_path` (the
   operating-characteristics simulation of the frozen plan) and `band_cost_report_path`
   (the look-1 early-stop band cost). Their sha256s are frozen in the intent and re-checked
   before the run and by the audit. Set fresh `namespaces`, the mixes, `primary_metric`,
   `ni_fraction`, `bands` and `closure_roots` (add the study package).
2. Save the study's screen or pilot paired deltas as `{"<mix>": [delta, ...]}` (at least
   20 per mix, Tier-0 data). This is the skew-check input.
3. Size `N_max` from development variance only (fixed-N requirement inflated about 2% for
   four OBF looks, or by simulating the frozen plan with
   `research/sequential_gate_validation_20261002/simulate.py`). Run the operating-
   characteristics simulation and the early-stop band-cost calculation the amendment asks
   for, and record both in `protocol.md` before `prepare`.
4. Commit, then:

```
./venv/bin/python research/sequential_strict_template/sequential_runner.py prepare \
    --spec research.<study>.spec:SPEC --out-root <artifact root>/run-v1 \
    --n-max <N_max> --mde <MDE> --n-calibration 16 --skew-input <deltas.json> \
    --deadline-utc <ISO, UTC> --authorization-quote "<user's words>"
./venv/bin/python research/sequential_strict_template/sequential_runner.py run \
    --intent <artifact root>/run-v1/intent.json
```

`prepare` refuses a dirty source closure and any slot lock root other than the global one
the v7 package uses (`SLOT_LOCK_ROOT`). There is **no resume**, as in v7: a parent that dies
without `closeout.json` leaves the run permanently `ABANDONED`
(`sequential_runner.py status --intent ...`), and `prepare` and `run` refuse that root.

**Ledger.** Every started run appends one fsynced line to the append-only
`sequential-strict-ledger.jsonl` beside the global slot locks (`LEDGER_PATH`, outside every
output root): study id, final namespace, intent sha256, output root. The line is written
under an exclusive lock after the slots are held and before `output/` exists. `prepare` and
`run` refuse any run whose final namespace is already in the ledger, so a namespace starts
at most one production run. `status` lists every ledger entry with its state, and an entry
without a closeout is `ABANDONED`.

**Dry runs.** `build_intent(..., dry_run=True)` must use the in-process executor (it never
spawns workers), its own slot lock root (never the global one) and its own ledger path. It
may use injected skew/audit runners and a dirty closure. It is used by the tests and for
plumbing checks. The executor and runner identities, `allow_dirty` and `dry_run` are recorded in
`started.json` and `closeout.json`. A dry run can only end `DRY_RUN_PASS`,
`DRY_RUN_FAIL`, `DRY_RUN_SKEW_CHECK_FAILED` or a failure, and it never writes
`receipt.json`. A production intent refuses injected executors and runners, and the audit
FAILs one that shows them.

## What the intent freezes (before any final world)

- Method version, `SequentialGatePlan.as_dict()` and its sha256: mixes, scripted mix,
  `N_max`, look sizes and fractions, family / per-mix / NI alpha, every boundary, nominal
  level and alpha spent, required successes, MDE, futility CP threshold,
  `futility_policy`, `band_policy`, `band_margin_z`. The plan parameters are kept too and
  the runner refuses an intent whose plan does not recompute from them.
- `futility_action`: `stop` (required under `futility_policy="followed"`) or `continue`
  (allowed only under `overridable`; every override is in the look receipts).
- Interleaving: world-major round robin `u = world_index x n_mixes + mix_position`, worker
  `u % 2` runs both arms of unit `u` (incumbent first), and `worker_look_counts`.
- Ordered final world bank (`N_max` seeds) and calibration bank, both from the
  `sha256("<domain>|worlds|<i>")` recipe, disjoint from `excluded_seeds`.
- Calibration rule (delta_NI = `ni_fraction` x calibration incumbent scripted mean of the
  primary metric; band bounds = calibration incumbent mean + offsets).
- Skew check: input path and sha256, probe path and sha256, reps, thresholds
  (per-mix efficacy any-look rate <= 1.2 x alpha_per_mix, i.e. 0.020 at 0.05 / 3;
  scripted NI any-look rate <= 0.06), remedy `stop_and_escalate` with the ladder.
- Caps (calibration, skew check, final, audit; positive integers no larger than the
  defaults), 120 s handoff reserve, 2 workers, no retry, no resume, source closure, arm
  identities, protocol / OC report / band-cost report sha256, `dry_run`, deadline.

## Run order and stopping

1. **Calibration** (incumbent only, every mix, both shards) -> `calibration.json`.
2. **Skew check** -> `skew_check.json`, after delta_NI is frozen and before the first final
   world. `skew_probe.py --part resample` runs once per mix with the frozen `N_max`,
   fractions and delta_NI. Efficacy is judged for every mix and NI for the scripted mix.
   If it fails, the run ends `SKEW_CHECK_FAILED`. No final world is played. The study
   does not proceed under this method. Next steps follow the amendment: either a new
   method version with a skew-robust statistic, validated by its own Monte Carlo, or a
   governance decision on the fixed-N gate. A later first look is not a remedy.
3. **Final looks.** For look `k`, each worker runs exactly its units between
   `worker_look_counts[k-1]` and `worker_look_counts[k]`, then exits. That exit is the
   barrier. The parent requires that:
   - both shard reports are complete;
   - the records on disk are exactly the look's unit prefix;
   - every record's bytes match its report.

   It then writes the create-only `looks/look-<k>.json`, which holds the look index,
   n per mix, units and per-worker counts, intent / plan / calibration / skew sha256, the
   previous receipt's sha256, every record's sha256, the deltas digest, band checks per
   look, the decision, `valid`, `passes`, the action, and the full `sequential_decision`
   output.

   The run stops on `STOP_PASS`, `STOP_FAIL_BANDS`, `STOP_FUTILE` (unless
   `futility_action="continue"`) and on the last look's `FINAL_PASS` / `FINAL_FAIL`. A
   worker refuses to start look `k+1` unless receipt `k` exists and says `continue`. It
   binds that sha256 (look 0: calibration and the passing skew check) into its shard's
   `started.json`, before any record.
4. **Audit** (`sequential_audit.py` child), then `closeout.json`, plus `receipt.json` on a
   pass.

Outcomes:

- `STRICT_PASS`: `STOP_PASS` or `FINAL_PASS`, a valid decision, audit PASS and
  agreement.
- `STRICT_FAIL`: `STOP_FAIL_BANDS`, `STOP_FUTILE` or `FINAL_FAIL`, with audit PASS.
- `SKEW_CHECK_FAILED`: as above, with audit PASS.
- `INCOMPLETE`: a cap or the deadline was reached.
- `INVALID_STOP`: any failure, drift, watchdog, battery, signal (SIGTERM, SIGHUP,
  SIGINT / KeyboardInterrupt), audit FAIL or disagreement.
- `DRY_RUN_PASS`, `DRY_RUN_FAIL`, `DRY_RUN_SKEW_CHECK_FAILED`: the dry-run counterparts;
  never a receipt.
- `STOP_INFEASIBLE`: `N_max` above its cap.

A failure is never relabelled, and no run is ever resumed or rerun. A `STOP_FUTILE` means
the gate failed, not that there is no effect. After an early stop, per-mix means are naive
and descriptive only.

## Hardening kept from the fixed-N packages

- Monotonic-clock heartbeats, watched by the supervisor.
- AC power required at admission and before every segment, and polled by the supervisor
  while children run (`on_battery` stops them; the run closes out `INVALID_STOP`).
- Workers check their parent (`getppid`) before and after every episode and exit without
  writing anything more if it died.
- Two CPU slot locks held by the parent and inherited by the workers, which assert them.
- Stage caps, with remaining caps plus the handoff reserve checked before each segment.
  A stage's time is charged on the monotonic clock from its first segment start marker,
  through every barrier and look analysis, so the cap cannot be reset between looks.
- Create-only atomic writes everywhere. `write_once` hard-links a fsynced temporary file.
- Intent sha256 bound into every marker, record and receipt.
- Before every segment: source-closure re-hash, arm-identity recompute and output-tree
  drift check.
- RSS, available-memory and stale-heartbeat watchdogs.
- No retry.

## Independent audit

`python -I sequential_audit.py --root <run root> --out <dir>` (exit 0 PASS, 1 FAIL)
recomputes the following in its own stdlib code:

- **Plan.** The OBF boundaries by its own recursive integration (within 2e-4 of the
  intent, checked against gsDesign in the tests), plus nominal levels, alpha spent, look
  sizes, the plan sha256 and the interleaving counts.
- **Calibration and banks.** Calibration from the calibration records, and the world
  banks.
- **Skew check.** The verdict from the hash-bound probe outputs. The resampling itself
  needs numpy and is not re-run.
- **Every look.** Each look up to the stop is replayed from the raw records: paired t
  with its own incomplete beta, NI bound, conditional power, bands with the `band_check`
  margin, decision, validity and action. Each must agree with its receipt, and a plain
  mean and t cross-check the reducer.
- **Prefix integrity.** No record lies beyond the stopping look's per-worker counts. Each
  record sits in its unit's segment, and each segment's `started.json` binds the receipt
  that allowed it.
- **Closeout.** A root without `closeout.json` is `UNCLOSED` (exit 1), never PASS. The runner
  calls the audit itself with `--pre-closeout`, between `producer-outcome.json` and
  `closeout.json`.
- **Outcome.** The expected outcome is checked against `producer-outcome.json`,
  `decision.json` and `closeout.json`. A dry run may only end `DRY_RUN_*` and has no
  `receipt.json`.
- **Provenance.** Unless the intent is a dry run, the audit FAILs on an in-process
  executor, an injected skew or audit runner, `allow_dirty`, a dirty closure, or a slot
  lock root or ledger other than its own copies of the global constants. It also
  FAILs on any resume marker, or on a pre-registration document whose sha256 changed.

## Deviations from the amendment (explicit)

1. **Skew-check timing and location.** The amendment lists the resampling check among the
   items frozen in `intent.json` before the first final world, with the frozen delta_NI.
   delta_NI is an output of the calibration stage. So the check runs after calibration and
   before the first final world. Its result goes to the create-only `skew_check.json`, not
   to `intent.json`. The intent freezes the check's inputs, thresholds and remedy. The
   final look-0 workers bind the receipt's sha256 before any final record, and the audit
   verifies that binding.
2. **NI judged on the scripted mix only.** The probe reports an NI any-look rate for every
   mix it is given, and its own `passes` field applies both thresholds. Non-inferiority is
   tested only on the scripted mix, so the runner judges efficacy (<= 0.020) on every mix
   and NI (<= 0.06) on the scripted mix only. The probe's `passes` field is recorded but
   not used.

The same two points are recorded in the amendment's "Implementation notes" and in the
intent (`skew_check.deviations_from_amendment`).

## Changes after v1 was merged

- 2026-10-03 (v8 strict study, first production use): run as a script, the runner is
  `__main__`, so a study spec's `from research.sequential_strict_template.sequential_runner
  import StudySpec` loaded a second copy of the module and `validate_spec` refused every spec
  ("spec must be a StudySpec") in `prepare` and in every worker child. The tests import the
  module, so they never saw it. The runner now aliases itself in `sys.modules` when it is
  `__main__`; `tests/test_apex_veto_v8_strict.py` drives `prepare` as a script. No gate,
  plan, audit or outcome logic changed.

## Limits (template)

- No serving stage and no smoke mode. A study that needs them adds them in its own
  package.
- The audit trusts the probe outputs' numbers, which are hash-bound but not recomputed.
- Band metrics are assumed independent of the efficacy metric. The amendment's
  selection-calibration caveat applies: simulate the plan with resampled joint records
  if they are strongly tied.
- `N_max` feasibility here is only `N_max <= 300`. A study adds its own runtime
  projection (the 70% rule) in `prepare`.

## Adopting paired survival bands (template v2)

The rule is from
[`governance_amendment_paired_bands_2026-10-03.md`](../../docs/research/governance_amendment_paired_bands_2026-10-03.md),
ratified 2026-10-03 for new strict pre-registrations: `paired_ni_at_stop`, M = 0.05,
alpha = 0.05 `pointwise`, absolute floor 0.30. It never applies to a gate that
pre-registered `block_at_stop`, including the running v8 strict gate.

**Rule.** At the qualifying look k, for each band (survival_fraction, one per mix), take the
look-k prefix of both arms on the same final worlds. With `d_i = candidate_i - incumbent_i`:

    pass iff mean(d) - t_{n_k-1}(band_nominal_p[k]) * sd(d) / sqrt(n_k) > -M
         and (if a floor is set) mean(candidate) >= floor

Every band must pass. Bands are judged once, at the qualifying look, and never delay a stop:
a failure is `STOP_FAIL_BANDS` at an interim look and `FINAL_FAIL` at the last. Calibration
still sets delta_NI. Its band-metric means are only descriptive, because the paired rule has
no reference bound.

**Steps.**

1. In the study's `spec.py`, set:
   - `band_policy="paired_ni_at_stop"`;
   - `bands=paired_survival_bands(MIXES)`;
   - `paired_band_check_path`, for example `research/<study>/paired_band_check.json`;
   - `paired_band_pool_path`: the study's own saved paired screen or pilot records (Tier 0
     data), in the format of
     `research/paired_band_validation_20261003/paired_survival_20261003.json`. That stock
     v7/v8 file itself is refused.

   v2 requires the mixes `frozen`, `scripted` and `mixed`, with exactly one survival_fraction
   band per mix, because the check simulates only those. Both files are pre-registration
   documents, and their sha256s are frozen in the intent.
2. Write the frozen plan document. It holds the exact `plan_parameters` and `plan` that
   `intent.json` will freeze:

   ```
   ./venv/bin/python research/sequential_strict_template/sequential_runner.py plan \
       --spec research.<study>.spec:SPEC --out <scratch>/plan.json \
       --n-max <N_max> --mde <MDE> --paired-band 0.05,0.05,pointwise,0.30
   ```

3. Run the per-study check on that pool, from the repo root. Give `--data` as the same path
   as `paired_band_pool_path`; a relative path is resolved from the repo. Use at least
   20,000 replicates, mass effects of 0.5, ~0.67, 1, 1.5 and 2 x MDE plus the screen
   estimate, and an explicit `--delta-ni` (the development estimate). Without that flag,
   `simulate.py` silently uses the v7 value, and `prepare` refuses the check.

   ```
   OMP_NUM_THREADS=1 ./venv/bin/python research/paired_band_validation_20261003/simulate.py \
       --part gate --data research/<study>/paired_pool.json --plan-params <scratch>/plan.json \
       --delta-ni <development delta_NI> --reps 20000 \
       --thetas <0.5 MDE>,<0.67 MDE>,<MDE>,<1.5 MDE>,<2 MDE>,<screen estimate> \
       --out research/<study>/paired_band_check.json
   ```

   Also run `--part bands` and report its no-regression pass rates in `protocol.md`.
4. Commit the check output and the pool beside the other pre-registration documents. Then
   run `prepare` with the same `--paired-band` value and `--development-delta-ni <the
   --delta-ni value>`. The settings must be stated explicitly; there is no default.

**What `prepare` checks.** It refuses unless the check output passes, judged again by the runner:

- it was run with `--plan-params` equal to this intent's plan parameters, `--delta-ni` equal
  to `--development-delta-ni`, and `--data` equal to the spec's pool;
- it used at least 20,000 replicates and covered the thetas above;
- it has a joint rate (P(qualify and the band regressed by exactly M passes)) for every
  pool x theta x band mix;
- every joint rate is at most 1.2 x `band_alpha`;
- its own `check.passes` agrees;
- the pool data still has the sha256 recorded in the output, and its pools are exactly the
  pools in the output.

The output and the pool are hash-bound in `intent.preregistration` (`paired_band_check`,
`paired_band_pool`) and summarized, with the development delta_NI, in
`intent.paired_band_check`. `run` checks them again before any child starts. If the check
fails, use `band_bound = "rci_obf"` (and re-run the check) or do not adopt this policy.

**Look receipts (v2).** `bands_by_look` holds, for each look and band: n, df, mean delta, SD,
SE, `nominal_p`, t critical value, lower bound, margin, candidate and incumbent means, floor,
`passes_ni`, `passes_floor`, `passes` and a digest of the pairs. `band_judged_look` is the
qualifying look, or `null` if the run has not qualified. `band_results` is that look's band
list.

**Audit (v2).** The audit uses only its own stdlib code and imports nothing from the repo.

- **Plan.** It checks the plan's band fields. It recomputes `band_nominal_p`: `band_alpha` at
  every look for `pointwise`, or its own OBF boundaries for `rci_obf`.
- **Bands.** It recomputes every band at every look from the raw record pairs, with its own
  mean, SD, SE and t quantile, and checks that both values of each pair come from the same
  world (mix, index, seed, unit). It compares each recomputed band with its receipt, and also
  checks the judged look and `band_results`. It reports a mismatch under
  `looks.paired_bands`.
- **Check output.** It judges the per-study check output again under
  `preregistration.paired_band_check`. It also checks:
  - the pool and delta_NI bindings;
  - that the stock pool was not used;
  - that there is exactly one survival band per mix;
  - the intent's `band_rule` text against the plan.
- **Report schema.** The audit report is `sequential-strict-audit/v2` for v2 runs and stays
  `/v1` for v1 runs.

**Limits.**

- The check validates survival bands over the three standard mixes only.
- `simulate.py` does not simulate `band_floor`, which is a tripwire.
- The audit judges the check output's numbers again but does not re-run the resampling,
  which needs numpy. The output is hash-bound.

## Survival band v2 (template v2-pooled)

The rule is [`governance_amendment_survival_band_v2_2026-10-06.md`](../../docs/research/governance_amendment_survival_band_v2_2026-10-06.md)
(proposed 2026-10-06; a production `prepare` refuses until its ratification section reads
exactly `- Decision: ratified` with an option and no changes, and the plan's band values must
equal the ratified option).
Opt in with a spec `band_policy="pooled_ni_continue"` and `bands=paired_survival_bands(MIXES)`
(exactly frozen, scripted, mixed), plus `paired_band_check_path` / `paired_band_pool_path` as
for v2.

- **Plan.** `--paired-band M,alpha,rci_obf,floor,mix_margin` (five values; option 1 is
  `0.05,0.05,rci_obf,0.30,0.075`, option 2 `0.05,0.05,rci_obf,0.30,0.10`). `band_bound` must be `rci_obf`. The plan adds
  `band_mix_margin`; v1/v2 plan dicts are unchanged (golden hashes in
  `tests/test_sequential_gate.py`).
- **Per-study check.** `research/survival_band_v2_20261006/simulate.py --part study --data
  <the study's pool> --plan-params <plan.json> --delta-ni <dev delta_NI> --reps 20000 --thetas
  <0.5,0.67,1,1.5,2 x MDE[,planning effect]> --out research/<study>/paired_band_check.json`.
  Acceptance: every joint rate P(PASS and a regression exactly at a margin) for the pooled
  regression and each one-mix regression is <= 1.2 x band_alpha. A failing check means the
  study does not adopt the policy (no retuning). The amendment's own calibration pools are
  refused as a study pool; the simulator is bound by sha256 in the intent.
- **Receipts.** `bands_by_look` holds one row per look (pooled bound, every mix's bound, floor,
  verdicts, pairs digest); `band_judged_looks` lists the qualified looks, `band_judged_look` /
  `band_results` are this look's (`None` before qualification). `CONTINUE_BANDS` maps to the
  action `continue`.
- **Audit.** Own-code recompute of the pooled and per-mix bounds at every look, the
  `CONTINUE_BANDS` decision sequence, the per-study check
  (`preregistration.pooled_band_check`) and the amendment binding (`intent.band_amendment`:
  sha256, ratification and option re-derived from the document). Report schema
  `sequential-strict-audit/v2-pooled` (v3 for remote runs).

## Running the gate on RunPod (template v3)

The rule is [`governance_amendment_strict_on_runpod_2026-10-05.md`](../../docs/research/governance_amendment_strict_on_runpod_2026-10-05.md)
(owner decision 2026-10-05; production use waits for its ratification section). Without
`--remote-config` nothing below applies and v1/v2 intents, plans, receipts and audits are
byte-identical (golden hashes in `tests/test_sequential_strict_template_paired.py`; the Tier-1
serverless runtime id is pinned in `tests/test_sequential_strict_template_remote.py`).

| File | Role |
|---|---|
| `remote_backend.py` | plan (speed-up vs 2 Mac slots), execution block, identity check, `RemoteSession` (one endpoint), `RemoteExecutor` (look segments as world-unit jobs) |
| `remote_worker.py` | runs ONE episode on a worker, from the frozen commit's archive; re-checks intent, closure and spec first |
| `strict_sls_handler.py` | the serverless handler, seeded as its own runtime (`serverless.py seed --handler strict`) |
| `remote_policy.json` | strict-specific knobs (4x per-step rule, `min_speedup`; attempts, guards, idle timeout); prices, image, volume and quota come from `research/runpod_fanout/` |

**What stays on the Mac.** The run parent: look boundaries, the barrier, shard markers,
create-only records and reports, receipts, the strict ledger, the audit, the closeout. It holds
an orchestrator lock (`<root>/remote/orchestrator.lock`), not CPU slots, and checks AC power and
the lid before every segment and while remote work runs.

**Units.** A unit is every arm of one (phase, mix, world). A job carries whole units only
(one wave: at most `slots` episodes, `slots = vCPU / 2`), every episode of a unit runs on one
worker, and a unit's records are written only when all of them came back from that job. Each
segment writes a create-only `remote.json`: its planned units, the job, worker, attempt and
platform of each, lost jobs and spend.

**Records** gain `platform` (backend, platform id, CPU model, ISA flags, worker id) and, on
RunPod, `fanout` (job id, attempt, unit, endpoint). Before a record is written the orchestrator
checks the worker's bindings (intent sha256, spec descriptor, arm identity, roster row sha256,
source-closure digest recomputed on the worker), the pinned horizon (`horizon_path`), the
spec's `validate_record`, and one platform stamp per unit.

**Failures.** A lost job (failed, timed out, no output, status unavailable) re-dispatches every
unit it carried, whole, alone from then on, up to `max_attempts_per_unit`; an episode that comes
back twice must have identical deterministic bytes or the run stops. An exception in the study
code on a worker, a binding mismatch, the spend cap, battery, a closed lid or a lost watchdog
stops the run `INVALID_STOP` after the endpoint is torn down; the remote wall cap or the stage
deadline ends it `INCOMPLETE`. A refusal before admission (seeding, quota, capacity, ledger,
probe) starts nothing.

**Steps.**

1. In the study's `spec.py`, set `remote_worker_setup(intent, ckpt_dir)` (its checkpoints are
   `ckpt_dir/<sha256>.pth`) and `remote_checkpoints()` (each sha256 on
   `research/runpod_fanout/checkpoint_allowlist.json`; uploads need owner approval).
2. Add the line `Execution platform: runpod-serverless` to `protocol.md`, and write the remote
   config (a pre-registration document, frozen by sha256):

   ```json
   {"schema": "sequential-strict-remote-config/v1", "platform": "runpod-serverless",
    "budget_usd": 25.0, "identity_budget_usd": 4.0,
    "remote_wall_minutes": 120, "identity_wall_minutes": 45,
    "mac_episode_seconds": 34.4, "mac_episode_seconds_source": "<where it was measured>",
    "engine": "live", "horizon": 5000, "horizon_path": "evaluation_profile.scored_horizon"}
   ```

   Optional: `cloud_episode_seconds` (a measured value beats the pod-calibrated default),
   `workers`, `vcpu_per_worker`, `flavors`, `identity_worlds_per_mix` (default 3),
   `record_pins`, `force_below_min_speedup` (formerly `force_below_5x`, still read as an alias), `sizing_objective` (`fastest`, the default: the fastest
   size within every cap; or `cheapest` at >= `min_speedup`, 4x).
3. `sequential_runner.py remote-plan --spec ... --remote-config ... --n-max ... --mde ...
   --n-calibration ...` prints the projected wall and speed-up vs 2 Mac slots (seeding, the
   identity check, cold starts and barriers counted; every look played), the sizing, the caps
   and what still blocks it. `--account` adds read-only balance and worker-quota reads.
4. Owner steps when `remote-plan` lists them: `serverless.py seed --handler strict --commit <C>
   --confirm` and `serverless.py template-create --handler strict --confirm`.
5. `prepare ... --remote-config <config>` (refuses below 4x unless forced, an unratified
   amendment in production, a protocol without the platform line, a checkpoint off the
   allow-list). The amendment is bound by sha256, so ratify it (its "Ratification" section,
   `- Decision: ratified`) and commit before `prepare`; seed the commit `prepare` freezes.
6. `sequential_runner.py identity-check --intent <root>/intent.json`: once per intent. It opens
   the endpoint first (a refusal starts nothing), then plays the sample on the Mac (2 slots,
   `identity-worker` children) and on RunPod, and writes `identity_check.json` with salted
   digests only. A session that got past `open` is never followed by another: deleting
   `identity_check.json`/`identity_check/` refuses ("already opened a session"), and so does
   deleting the session dir too (the shared ledger still names its run).
7. `sequential_runner.py run --intent <root>/intent.json`: PASSED runs on RunPod; FAILED or
   ABANDONED runs the same intent on the Mac (2 slots); NOT_RUN refuses.

**Audit (v3, report schema `sequential-strict-audit/v3`).** `execution.platform_named`
(ratification re-derived from the hash-verified amendment text, not taken from the intent's
flag), `execution.speedup_rule`, `identity.result_and_backend` (sample recomputed from the
pre-registered ranking, verdict recomputed from the digests, rows bound to `rosters.json`, the
backend the state implies), `identity.run_once` (exactly one identity session opened, the one
`identity_check/started.json` and the result's receipt name; every shared-ledger run of the
identity job still has its run dir; the ledger defaults to the frozen fan-out policy's
`artifacts_root`, override with `--ledger`), `platform.per_world_single`
(every record stamped, one stamp and one job per world unit), `remote.segments_exact_units`
(each segment's `remote.json` lists exactly its planned units, published whole from the job
the records name), `remote.spend_stop_is_invalid` (either SpendStop, cap reached or balance
unreadable, ends `INVALID_STOP`) and `serving.stays_on_mac` (the intent says serving
qualification stays on the Mac and the closeout records no promotion); segment gates bind the
identity state.

**Binding of the identity result (explicit deviation).** The task asks for the identity result
to be "bound into the intent before the final stage". `intent.json` is create-only and frozen
before the check runs (the sample is part of it), so, as for the skew check, the result goes to
the create-only `identity_check.json` and its state and sha256 are bound into `started.json`,
every segment's `started.json` gate (calibration and final), the closeout, and re-checked before
every segment; the audit recomputes the verdict and the backend it implies.

**Limits.** The identity check detects gross platform sensitivity, not rare near-tie flips
(see the amendment). Remote wall estimates are pod-calibrated x `episode_time_factor`. The
balance read is account-wide; other runners' ledger reservations are subtracted.

## Protocol template (copy into the study's `protocol.md`)

```
# <Candidate> vs <incumbent>: Tier-2 group-sequential strict challenge (pre-registration)
Method strict-sequential-obf-bonferroni-v1, runner sequential-strict-template/v1 (or /v2).
Question and decision it informs: ...
Arms and identities: ... (arm_identities output, module sha256s)
Mixes, rosters, world namespaces (calibration, final; excluded earlier namespaces): ...
Calibration: N_cal worlds/mix; delta_NI = <ni_fraction> x scripted incumbent mean; bands: ...
Plan: N_max = ..., looks ... (sizes ...), MDE ..., futility CP 0.10, futility_policy ...,
  futility_action ..., band_margin_z 1.645; boundaries (efficacy / NI): ...
Bands (v2 only): paired_ni_at_stop, M ..., alpha ..., bound ..., floor ...; per-study check
  output <path> (sha256 ...), pool <path> (sha256 ...), max joint rate ... <= 1.2 x alpha.
Sizing basis (development variance only) and operating characteristics (simulate.py): ...
Early-stop band cost at look 1 (pilot SD of each band metric): ...
Skew check input: <path>, sha256 ...; thresholds 0.020 / 0.06; remedy: stop and escalate.
Caps, deadline, no resume, dry_run=false, outcomes: as in the template README.
(template v3 only, else omit) the exact line below, which names the platform:
Execution platform: runpod-serverless
```
