# Sequential strict runner template (`sequential-strict-template/v1`)

Status: **template, code and unit tests only.** Nothing in this folder has played a game.
It implements the runner, look barrier, look receipts and independent audit that
[`governance_amendment_sequential_gates_2026-10-02.md`](../../docs/research/governance_amendment_sequential_gates_2026-10-02.md)
lists as "not yet done". The decision functions are `src/evaluation/sequential_gate.py`
(method `strict-sequential-obf-bonferroni-v1`). The fixed-N packages
(`research/apex_veto_v{5,7}_strict_*`) are neither modified nor imported.

| File | Role |
|---|---|
| `sequential_runner.py` | `prepare` (intent), `run` / `resume` (parent), `worker` (child) |
| `sequential_audit.py` | independent audit, standard library only, imports nothing from the repo |
| `example_spec.py` | instantiation template; its episode runner raises |

## How to instantiate for a candidate

1. Copy `example_spec.py` into the study package (e.g. `research/apex_veto_v8_strict_<date>/spec.py`)
   and fill in the four marked parts: `arm_identities` (computed from the files),
   `episode_runner` (installs the arm and calls `tournament_eval.rollout`), `build_row`
   (rosters / world identity) and `validate_record`. Set fresh `namespaces`, the mixes,
   `primary_metric`, `ni_fraction`, `bands`, `excluded_seeds` (every earlier namespace) and
   `closure_roots` (add the study package). Put the study's `protocol.md` (copy the
   section below) at `protocol_path`.
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

`prepare` refuses a dirty source closure. Only a parent that died without
`closeout.json` may be continued with `resume --intent ...` (pre-registered in the intent,
at most `max_resumes` times).

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
- Caps (calibration, skew check, final, audit), 120 s handoff reserve, 2 workers, no
  retry, resume policy, source closure, arm identities, protocol sha256, deadline.

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
- `INVALID_STOP`: any failure, drift, watchdog, signal, audit FAIL or disagreement.
- `STOP_INFEASIBLE`: `N_max` above its cap.

A failure is never relabelled, and a closed-out run is never resumed. A `STOP_FUTILE` means
the gate failed, not that there is no effect. After an early stop, per-mix means are naive
and descriptive only.

## Hardening kept from the fixed-N packages

- Monotonic-clock heartbeats, watched by the supervisor.
- AC power required at admission and before every segment.
- Two CPU slot locks held by the parent and inherited by the workers, which assert them.
- Stage caps, with remaining caps plus the handoff reserve checked before each segment.
  Wall time is cumulative across final segments.
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
- **Outcome.** The expected outcome is checked against `producer-outcome.json`,
  `decision.json` and `closeout.json`.

## Limits (template)

- No serving stage and no smoke mode. A study that needs them adds them in its own
  package.
- The audit trusts the probe outputs' numbers, which are hash-bound but not recomputed.
- Resume covers parent death only. A worker crash seen by the parent is `INVALID_STOP`.
- Band metrics are assumed independent of the efficacy metric. The amendment's
  selection-calibration caveat applies: simulate the plan with resampled joint records
  if they are strongly tied.
- `N_max` feasibility here is only `N_max <= 300`. A study adds its own runtime
  projection (the 70% rule) in `prepare`.

## Protocol template (copy into the study's `protocol.md`)

```
# <Candidate> vs <incumbent>: Tier-2 group-sequential strict challenge (pre-registration)
Method strict-sequential-obf-bonferroni-v1, runner sequential-strict-template/v1.
Question and decision it informs: ...
Arms and identities: ... (arm_identities output, module sha256s)
Mixes, rosters, world namespaces (calibration, final; excluded earlier namespaces): ...
Calibration: N_cal worlds/mix; delta_NI = <ni_fraction> x scripted incumbent mean; bands: ...
Plan: N_max = ..., looks ... (sizes ...), MDE ..., futility CP 0.10, futility_policy ...,
  futility_action ..., band_margin_z 1.645; boundaries (efficacy / NI): ...
Sizing basis (development variance only) and operating characteristics (simulate.py): ...
Early-stop band cost at look 1 (pilot SD of each band metric): ...
Skew check input: <path>, sha256 ...; thresholds 0.020 / 0.06; remedy: stop and escalate.
Caps, deadline, resume policy, outcomes: as in the template README.
```
