---
name: snake-dqn-colab-jobs
description: |
  Use whenever you write, change, review, launch, or monitor Snake DQN code that will run on a Google Colab
  runtime. New jobs use research/colab_cli and the official CLI; research/colab_queue and worker notebooks are
  legacy only for the already-running Q1 work. Mandatory pre-flight before declaring a Colab job ready.
---

# Snake DQN: Colab CLI job checklist

## Transport boundary

- **New work:** use only `research/colab_cli` through `python -m research.colab_cli.supervisor`. The Mac supervisor
  is the single broker for allocation and stop. It uses explicit OAuth2, `COLAB_CLI_ASSIGN_RETRIES=0`, a unique
  session name, and a separate CLI config per lane.
- **Legacy only:** `research/colab_queue`, its job manifests, and `notebooks/colab_queue_worker*.ipynb` remain in
  place solely for the two already-running notebook workers and Q1. Do not modify, enqueue, migrate, or start them
  for new work. Use `snake-dqn-queue-jobs` only when babysitting that legacy path.
- Never call `colab new`, `exec`, `upload`, `download`, or `stop` directly for a job. Do not use `colab auth` or
  `colab drivemount`: they are human-interactive, and Drive is prohibited on a CLI runtime.

## Admission and manifest

- [ ] Pre-registration/approval exists before a learning job. The supervisor is not spending authorization.
- [ ] Use a unique `job_id`, a full 40-character `commit`, repository-relative Bash `script`, string-only `env`,
      `requirements` (requested `gpu`, optional pre-priced `allowed_gpus`, `min_vcpu`, `est_minutes`), positive `timeout_minutes`, a fresh absolute local
      `output_dir`, hash-bound `inputs`, `budget` (`unit_cap`, `rate_units_per_hour`, `reserve_units`), and bounded
      `max_export_bytes`. `timeout_minutes` is at least 3 and 2x the estimate; `lease_seconds` defaults to 180 and
      is accepted only from 180 through 600.
- [ ] Run `validate` before `run`; validation does not allocate. A used or nonabsolute output directory is a refusal,
      never a reason to delete or overwrite evidence.
- [ ] Count all account assignments, including notebook workers. The account-wide cap is three concurrent
      assignments. No automatic allocation retries, job retries, or unit purchases.
- [ ] Price setup, export, idle time, and reserve. First wall-clock or unit cap wins. CPU debit is measured, never
      assumed zero. `rate_units_per_hour` is a pre-priced ceiling; observed account debit conservatively includes
      other notebook/lane spend and provider lag. Default to CPU/L4; retain an H100 only for a finite priced,
      approved chain of useful work.
- [ ] Treat actual job units as unknown until acceptance qualifies the conservative account-debit guard under
      concurrent notebook spend and provider-accounting lag; it may stop a lane early.

## Environment and inputs

- [ ] **No Mac-only runtime paths or files.** Ship a Mac-computed receipt or input by supervisor upload and bind its
      SHA-256. A runtime has no `/Users/...`, local artifact history, or Drive mount.
- [ ] Runtime bootstrap uses a Python 3.12 `uv` environment. Select cu126 for L4/A100/H100; use cu128 for Blackwell
      (compute capability >= 10). Before ready/bootstrap, the supervisor verifies model/capability against
      `allowed_gpus` and writes `hardware.json`; no unpriced substitution is allowed.
- [ ] Run the real entrypoint on a tiny local input first with the manifest environment. Time at least three real
      units when practical; set the estimate from those timings and a timeout of at least 2x the measured estimate.
- [ ] CPU-bound worlds use deterministic process pools and fixed-order merging. Prove serial equals parallel bytes.

## Runtime execution, telemetry, and exports

- [ ] The kernel launcher starts one supervised subprocess and returns. Later polling executions have explicit
      timeouts, renew the generation-fenced lease, and read state/progress files; never blindly resubmit a launcher
      to a busy kernel. A CLI exit status of zero does not clear a kernel error.
- [ ] Write progress (`done/total`, elapsed, ETA) about every two minutes. The job launcher, not the supervisor,
      makes units resumable and creates its complete optimizer, replay, actor, target, RNG, and counter snapshot
      before marking an export complete. Weights alone are not exact resume.
- [ ] Exports are immutable payload + SHA-256 sidecar + completion marker. The supervisor downloads only bounded
      completed exports to a temporary Mac path, checks the digest, then atomically renames. Failed transfers are
      rejected and temporary files are removed. Optional Drive-for-desktop backup happens only after local
      verification.
- [ ] The runtime watchdog stops spending after lease expiry: request a checkpoint, give the child its bounded grace,
      then unassign. A Mac disconnect can lose work after the latest acknowledged export.
- [ ] Final success needs a terminal create-only receipt, verified local artifacts, `stop`, and no matching
      assignment in account-wide `colab sessions`. `run` cleanup or a client timeout is not release proof.

## Exact interface

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
LAUNCH=/Users/josenunez/Projects/ml/snake-dqn-colab-cli/research/colab_cli/launch.sh
SUPERVISOR_STATE=/absolute/path/outside/repo/colab-cli/<lane>-<attempt>
COLAB_BIN=/Users/josenunez/.local/bin/colab
"$LAUNCH" --check --state-dir "$SUPERVISOR_STATE" --colab "$COLAB_BIN" validate /absolute/path/job.json
"$LAUNCH" \
  --state-dir "$SUPERVISOR_STATE" --colab "$COLAB_BIN" validate /absolute/path/job.json
tmux new-session -d -s colab-cli-<lane>-<attempt> \
  "'$LAUNCH' --state-dir '$SUPERVISOR_STATE' --colab '$COLAB_BIN' run /absolute/path/job.json"
"$PY" -m research.colab_cli.supervisor \
  --state-dir "$SUPERVISOR_STATE" --colab "$COLAB_BIN" status
"$LAUNCH" --state-dir "$SUPERVISOR_STATE" --colab "$COLAB_BIN" chain \
  /absolute/path/first-h100.json /absolute/path/second-h100.json \
  --retention-minutes 45 --retention-units 10
"$LAUNCH" --state-dir "$SUPERVISOR_STATE" --colab "$COLAB_BIN" recover \
  /Users/josenunez/.config/colab-cli/snake/pending-<session>.json
```

`launch.sh` is foreground, applies AC/open-lid guards and `caffeinate -is`, and belongs inside the caller-owned
prepared tmux service. `status` is read-only and does not prove exact provider hardware capability. `recover` only
accepts a broker-owned account `pending-*.json` or `active-*.json`; never infer an identity from the sessions list.

Use `"$COLAB_BIN" --auth oauth2 usage` and `sessions` for read-only account evidence. See `docs/colab-cli.md` and
`docs/research/advice/2026-10-18_colab_cli_operating_model.md` for operator guidance and acceptance boundaries.

## Before you say done

1. Run the supervisor tests and the job's narrow tests; lint changed Python/shell files.
2. Validate the manifest and report its commit, input digests, output path, measured per-unit time, worker count,
   estimate, wall/unit caps, rate, and reserve.
3. State whether the job was only prepared or actually run. For a completed run, report terminal receipt and
   independently verified session disappearance. Do not treat dashboard/screen state as evidence.
