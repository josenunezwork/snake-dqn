---
name: snake-dqn-runpod
description: |
  Use whenever a task in /Users/josenunez/Projects/ml/snake-dqn mentions RunPod, pods, GPU rental, the pod
  runners (research/perf_podbench_20261013/pod_bench.py, research/redesign_m3c_20261008/gpu_run_v2.py,
  research/runpod_fanout/runner.py), rp.py, pod plans/ledgers/watchdogs, spend caps, stock polling, or
  cleaning up pods. Covers the money-safety contract (dry-run first, caps, lifetime, watchdog, Keychain
  keys never printed), the per-step 4x rule, and the pitfalls already paid for. Implementers prepare and
  dry-run; only the owner/operator spends.
---

# Snake DQN: RunPod pod tooling and safety

Facts verified 2026-10-18 against the repo and `~/.claude/skills/runpod/{NOTES,OPTIMIZATION,HARDWARE}.md`
(read NOTES.md for prices/API details before planning anything). Current standing rules
(`docs/STATUS.md`): **no new pod work until the pod runner is stable**; approved spend and uploads only;
always clean up; strict gates and serving qualification run on the Mac, never on pods.

## Non-negotiable safety rules

1. **Dry run first, always.** Every tool has a no-spend mode; show its output in the handoff:
   - `gpu_run_v2.py plan --plan P.json [--run-dir D]` (inputs, pod body, rp.py dry-run request)
   - `pod_bench.py bench --dry-run [--commands C.json]`
   - `runner.py plan JOB.json [--budget USD]`
   - `rp.py` (the shared client, `~/.claude/skills/runpod/scripts/rp.py`): every mutating call is a dry
     run unless `--confirm`; reads (`rp.py account`, `rp.py rest GET /pods`, `price-cpu`, `price-gpu`,
     `datacenters`) run directly.
2. **`--confirm` = money.** Only after the owner's explicit approval in chat for THAT launch. You
   (Sol) never run `--confirm`; print the exact command for the operator.
3. **Caps, every launch:** `--spend-cap` (gpu_run_v2 default $10; pod_bench default $1.50, max $8),
   `--lifetime-hours` (gpu_run_v2 default 8; pod_bench 1.0, max 1.5). The runner refuses if
   `lifetime x plan.max_hourly > spend_cap`. Per-step caps stay ~$10 (owner standing permission 2026-10-12).
   The fan-out runner reads `research/runpod_fanout/fanout_policy.json` (project cap, $5 floor,
   `pod_max_lifetime_seconds` 5400, `max_hourly_per_pod_usd` 1.0, Tier-1 only; names containing
   strict/serving/gate/qualif are refused).
4. **Hard stop comes from the Mac.** No pod TTL exists. Each session arms: a detached Mac watchdog that
   deletes only the pod of its exact name at lifetime + 5 min, the shared ledger reservation
   (`snake-dqn-artifacts/runpod-fanout/ledger-v2.json`), a balance-drop guard, pod-side job kill at the
   deadline + dead-man switch, and delete-and-verify on every exit. Keep the Mac on AC while a pod lives.
5. **Keys.** Account key: macOS Keychain service `runpod-api` (read only by rp.py; `RUNPOD_API_KEY`
   env overrides). Optional restricted self-delete key: Keychain service `runpod-selfdelete`, account
   `runpod` (pods read/write only), read by `gpu_run_v2.read_self_delete_key()` and delivered to the pod
   agent in memory over the authenticated HTTPS proxy. NEVER: print a key, echo it, put it in argv, env,
   pod body, template, logs, state files, tests, or your handoff. Tests must mock Keychain
   (`PYTEST_CURRENT_TEST` disables the real read). The owner adds Keychain items via Keychain Access,
   not via a command you write.
6. **Uploads are allow-listed** (`research/runpod_fanout/checkpoint_allowlist.json`; gpu_run_v2 plan
   `extra_inputs` are sha256-pinned and refused on mismatch). Never upload a new checkpoint or dataset
   without the owner's approval written in the plan.
7. **The per-step >= 4x rule.** A step runs on RunPod only if its projected wall-clock (incl. ~1.5 min
   cold start, setup ~15 min, uploads, pulls) is >= 4x faster than the same step on the Mac, from
   MEASURED rates (Mac vector61 eval ~0.0042 s/frame-world per slot; pod fast host ~0.005, slow host
   ~0.05; one M5 Pro core ~3.6x a cpu3c vCPU). gpu_run_v2 enforces it live as gate G2 (`plan.g2_factor`
   x `plan.mac_rate` at ~30 min, else delete + Mac fallback). Waivers are owner-granted per step and
   recorded in the run sidecar; none carries over. Cheap pods before serverless (serverless CPU billed
   ~$0.59/worker-hr, 6-10x catalog for small workers).

## The three tools

| Tool | Purpose | Spend commands |
| --- | --- | --- |
| `research/redesign_m3c_20261008/gpu_run_v2.py` | N training runs on ONE GPU pod (RTX 4090 secure default) from a sha-pinned plan JSON (`m3c-gpu-plan/v1`); adaptive concurrency, telemetry, incremental checkpoint pulls, G2 gate | `plan` / `launch --confirm` / `cleanup --confirm` |
| `research/perf_podbench_20261013/pod_bench.py` | one short CUDA benchmark session (4090 or H100 SXM), commands checked against the COMMITTED bench scripts' argparse; pods named `perfbench-*` | `bench --dry-run` / `bench --confirm` / `cleanup --out-dir D --confirm` |
| `research/runpod_fanout/runner.py` | Tier-1 episode fan-out (screens/sweeps/censuses) on CPU pods or serverless, per-world platform rule; pods `rpf-<job>--*` | `plan` / `run --confirm` / `status` / `cleanup --job-id ID --confirm` / `compare` / `merge` |

Shared knobs: `--pod-prefix name-` (a runner only sees/guards/deletes pods with its prefix; distinct
prefixes let sessions coexist), `--stock-wait-hours H` (<= 3) with `--stock-poll-s` (default 60; a
refused create costs $0; 429 backs off), `--gpu-fallbacks` (opt-in; for 4090-sized plans PIN the GPU and
let the 1-min poll wait -- owner rule 2026-10-14), `--accept-no-self-delete "<who decided, when>"`
(>= 10 chars; required when the in-pod probe is not 200), `--run-dir` / `--out-dir` (a NEW dir per attempt).
Run them with `./venv/bin/python` from a clean committed tree (pod_bench requires it; gpu_run_v2 checks
every run's argv against the committed trainer's argparse).

## Relaunch and cleanup hygiene (memory: pod-runner-relaunch-hygiene)

- Before relaunching: `ps -Ao pid,command | grep gpu_run_v2` (read, do not kill) and the old log ends
  with `EXIT=`. Never `tmux kill-session` a live runner, never `mv` its run dir (runners ignore SIGHUP
  during cleanup and then fail on moved paths). New tmux session name + new `--run-dir` per attempt.
- Stale ledger run with no pod (`rp.py rest GET /pods` -> `[]`): close it with
  `SharedLedger.finish_run(run_id, success, episodes_complete)`; do not edit the JSON by hand.
- Session end: `rp.py rest GET /pods` must be empty (and `GET /endpoints` for serverless). Leftover pods
  bill until deleted: the runner's `cleanup --confirm` deletes by prefix and verifies; `sweep_strays`
  handles late-listed pods from refused stock rounds. A watchdog "left armed" alert means a pod MAY exist:
  tell the operator to run `cleanup --confirm`.
- Never kill PIDs from `pgrep -f`; the 2026-10-03 tmux-server kill left a pod billing (see
  `snake-dqn-mac-ops`).

## Pitfalls already paid for

- **Stock refusals:** `POST /pods` returns HTTP 500 `"create pod: There are no instances currently
  available"`; the runners classify it `no_capacity` (`NO_CAPACITY` regex) and only THAT (plus 429) is
  retried by the stock wait; any other error ends the wait. 32-vCPU CPU stock is rare (EUR-IS-1), 4090s
  vanish in minutes: poll every ~60 s, pin the GPU. Community cloud (`pod_bench --cloud COMMUNITY`) is
  cheaper but low stock, unknown host CPU, no network volumes; probe determinism + speed before use.
- **Data-center ids are an enum** on REST pod create (`dataCenterIds`, allowed list in
  `~/.claude/skills/runpod-pods/reference.md`, e.g. `EU-RO-1`, `EUR-IS-1`, `US-CA-2`); ids from GraphQL
  stock queries or guesses that are not in the list are rejected. Read stock per DC from
  `GET /v2/catalog/cpus|gpus?include=AVAILABILITY&product=POD...`; ignore the top-level `availability`.
  Incident 2026-10-17 (T18 H100, `snake-dqn-artifacts/t18-h100pod-20261017/state.json` try 7): the create loop
  iterated stock data centers from GraphQL and sent `EUR-NO-2`, which REST rejected with HTTP 400 (not in the
  enum). Filter candidate DCs against the REST enum before trying them.
- **Injected pod key is useless:** `RUNPOD_API_KEY` inside a pod gets 403 on `GET /v1/pods/$RUNPOD_POD_ID`
  (2026-10-08, recorded in `perf-cuda-bench-20261013-try2/state.json`: "pod self-delete not proven (probe
  403)"). In-pod self-delete only works with the owner's restricted `runpod-selfdelete` key (unverified on a
  live pod as of 2026-10-17); otherwise launches need the explicit waiver and Mac-side stops.
- **Host speed varies ~10x on the same CPU model** (EPYC 9655P 0.004-0.05 s/frame-world): setup speed
  probe refuses hosts > 3x slower; size lifetimes from the slow-host floor.
- **Loading torch from a network volume** took ~40 s per process and broke heartbeats: copy the runtime to
  container disk. **Volume statfs** shows the backing store, not our quota.
- **Mac vs x86 float ties** flip ~1/50 long episodes (Accelerate vs MKL): keep every world on one CPU
  model; no strict evidence from pods.
- **REST v1 dies 2026-11-15** (410); GraphQL early 2027. rp.py, `pod_agent.self_delete()` and pod bodies
  must move to `https://api.runpod.io/v2` (`cpu:{id,vcpuCount}`, `mounts.network`, `DELETE /v2/pods/{id}`).
- **torch.compile on the pod** needs the devel image (`IMAGE_DEVEL`); the runtime image silently falls
  back to eager.
- `pod_bench` has no G2 gate by design (nothing to compare with on the Mac); its spend bound is
  lifetime x max_hourly + ledger + balance-drop cap.

## Before you say done (implementer)

1. Run the tool's dry run / `plan` and paste the cost bound, pod body summary, and any refused commands.
2. Tests (pick those for what you touched): `./venv/bin/python -m pytest -q tests/test_m3c_gpu_run_v2.py
   tests/test_gpu_run_v2_opt_in.py tests/test_m3_gpu_run.py tests/test_runpod_fanout.py
   tests/test_runpod_serverless.py tests/test_runpod_sls_batch.py` (pod_bench is covered through the
   gpu_run_v2 tests and its own `--dry-run`), with Keychain and network mocked; grep your diff for
   anything that could print a credential.
3. State the measured Mac rate, the projected pod rate, the resulting speed-up vs 4x, utilization plan
   (>70% busy on the bottleneck resource; OPTIMIZATION.md checklist), the cap, lifetime and prefix.
4. Handoff (see `snake-dqn-git-workflow`): the exact `--confirm` command for the OPERATOR, the cleanup
   command, and a note that you did not create, start or delete any pod.
