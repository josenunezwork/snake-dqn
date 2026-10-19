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
  is the single broker for allocation and stop. It uses explicit OAuth2, a unique session name, a separate CLI
  config per lane, durable assignment recovery, and explicit subprocess timeouts. A durable pending UUID enables
  authenticated backend recovery after an interrupted allocation; it cannot prove an offline VM release during the
  allocation-to-watchdog window. Do not rely on undocumented CLI environment variables to create retry or
  request-timeout behavior.
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
      assignments. At admission, record every pre-existing endpoint and its reviewed positive units-per-hour rate in
      `budget.other_assignments`; an unknown endpoint is a refusal. After allocation, calibrate a stable positive
      observed provider rate (up to 120 seconds; three reads five seconds apart; within 0.01 units/hour) after
      subtracting reconciled other assignments and allowing provider lag. Require observed <= priced upper-bound
      rate + 0.05 units/hour; an unknown, unstable, unexplained, or over-ceiling calibration stops as INCOMPLETE.
      Thereafter detect drift from the observed rate. No automatic allocation retries, job retries, or unit purchases.
- [ ] Price setup, export, idle time, and reserve. First wall-clock or unit cap wins. `rate_units_per_hour` is the
      conservative priced upper bound that charges the session from elapsed time; it remains the admission,
      reservation, and cap rate after a lower observed rate is calibrated. CPU debit is measured, never assumed zero.
      Acceptance defaults to priced ceilings of CPU 0.5 and L4 2.0 units/hour (overridable with `--cpu-rate` and
      `--l4-rate`); measured CPU/L4 rates are outputs, not required inputs.
      Cross-check the account debit only
      after subtracting the measured baseline rate of the accounted assignments. The five-minute provider-lag and
      rounding allowance produces a receipt warning, not a raw-account spending cap; never let Q1 or another lane's
      debit trip this session's cap. Default to CPU/L4; retain an H100 only for a finite priced, approved chain of
      useful work.

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
      timeouts, renew the generation-fenced lease, and read state/progress files. An uncertain RPC means the kernel
      may still be busy: use the file transport and do not resubmit work until independent backend evidence proves
      the kernel idle. A CLI exit status of zero does not clear a kernel error.
- [ ] Write progress (`done/total`, elapsed, ETA) about every two minutes. The job launcher, not the supervisor,
      makes units resumable and creates its complete optimizer, replay, actor, target, RNG, and counter snapshot
      before marking an export complete. Weights alone are not exact resume.
- [ ] Exports are immutable payload + SHA-256 sidecar + completion marker. The supervisor transfers complete files
      in bounded 8 MiB chunks with size-scaled timeouts. Renewal is throttled to at most every 30 seconds, so a
      chunk transfer does not queue two polling RPCs; each 8 MiB call has a 60-second bound. The supervisor verifies
      into a temporary Mac path, then atomically renames. Only idempotent reads receive their bounded three-attempt
      retry; uncertain mutating kernel operations are never retried. The limit is `max_export_bytes` (at most 2 GiB). Failed transfers are
      rejected and temporary files are removed. Optional Drive-for-desktop backup happens only after local
      verification.
- [ ] The runtime watchdog stops spending after lease expiry: request a checkpoint, give the child its bounded grace,
      then use the authenticated Colab tunnel unassign GET/XSRF-POST sequence. It cannot call
      `google.colab.runtime.unassign()` because a detached subprocess lacks a notebook frontend. The Mac uploads a
      private, short-lived access token with exactly `colaboratory` and `userinfo.email` scopes; it never uploads the
      owner refresh token, client secret, or token URI, and a scope/expiry mismatch is a refusal. The token is
      account-scoped rather than endpoint-bound and is visible to trusted same-user runtime code; redirects are
      denied, naive expiry is normalized as UTC, and the runtime requires lease + grace + 10 seconds + a 130-second
      release margin before starting the child.
- [ ] Final success needs a terminal create-only receipt, verified local artifacts, up to three 75-second stop
      attempts while the endpoint remains listed, and no matching assignment in account-wide `colab sessions`.
      `STOP_UNVERIFIED`, `run` cleanup, or a client timeout is not release proof.

## Exact interface

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
LAUNCH=/Users/josenunez/Projects/ml/snake-dqn/research/colab_cli/launch.sh
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
The <=4-unit acceptance must pass before a production allocation. Its L4 Mac-disconnect fault requires the exact
assignment to disappear from sessions within 300 seconds after lease expiry, before broker fallback cleanup; the
total bound includes lease, grace, terminal allowance, and the 130-second release window. Anything else is a failure.

Use `"$COLAB_BIN" --auth oauth2 usage` and `sessions` for read-only account evidence. See `docs/colab-cli.md` and
`docs/research/advice/2026-10-18_colab_cli_operating_model.md` for operator guidance and acceptance boundaries.

## Before you say done

1. Run the supervisor tests and the job's narrow tests; lint changed Python/shell files.
2. Validate the manifest and report its commit, input digests, output path, measured per-unit time, worker count,
   estimate, wall/unit caps, rate, and reserve.
3. State whether the job was only prepared or actually run. For a completed run, report terminal receipt and
   independently verified session disappearance. Do not treat dashboard/screen state as evidence.
