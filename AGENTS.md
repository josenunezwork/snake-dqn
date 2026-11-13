# AGENTS.md

Guidance for Codex (`Codex.ai/code`), Claude (`CLAUDE.md`), and Argon (`agy`) when working in this repository.

## Current state and operating rules — read first

[docs/STATUS.md](docs/STATUS.md) (with [docs/INDEX.md](docs/INDEX.md)) is the current-truth source and wins on conflicts. Superseded pre-2026-11-09 campaign, 2026-10-27 implementer, legacy `research/colab_queue/` (`notebooks/colab_queue_worker*.ipynb`, Q1-only) rules, and uncondensed pre-2026-11-09 text live verbatim in [docs/history/AGENTS_superseded_2026-11-09.md](docs/history/AGENTS_superseded_2026-11-09.md).
Implementers and reviewers must read [docs/REGISTERED_ITEMS.md](docs/REGISTERED_ITEMS.md) and [docs/IMPLEMENTER_CHECKLIST.md](docs/IMPLEMENTER_CHECKLIST.md) before editing.

## Owner ruling 2026-11-10: goal and loosened rules

Ratified `2026-11-10 ~19:20 PT` ([docs/STATUS.md](docs/STATUS.md), [CAMPAIGN_LOG.md](docs/research/final_run_20261019/CAMPAIGN_LOG.md) `2026-11-10 PT 19:25`); overrides conflicting older text:
1. **Goal & release bars:** Train the strongest possible slither.io-style agent. Mac serving bars (`24 fps`, `GameState.update p95 <= 11.688 ms`, Mac strict gate, serving qualification) gate **release** only, not what is trained; a large teacher distilled to a Mac-size student is allowed.
2. **`gpu_sitl` Step B1b = exploration lane:** Step B1b smoke and pilot (`research/gpu_sitl`) may use any GPU (`L4` included); wall-time and unit caps are runaway guards the operator may raise and log with no judge amendment. Bars, held-out namespaces, and seeds are unchanged.
3. **Kill rules close only the tested recipe:** Clauses such as `"permanently closes all learned-student work on ego2s-bv8 inputs"`, `"KILL_BRANCH_B ... 0 further Colab units and no sweeps"`, and whole-line `STOP_AND_MAINTAIN` triggers stop only that recipe; a different recipe may be tried after an advisor design note. Served agents stay as they are until a release.
4. **Stock polling:** Scarce GPU stock is polled about once a minute (`~1/min`), not every `20` minutes.
5. **New main track:** Large-scale from-scratch training on the exact GPU simulator, using all Colab slots, is opened (*design pending, advisor round 2026-11-10*).
6. **Always kept:** Held-out/test data never used for training or tuning; honest reporting; verified stops and cleanup; `L4` shake-out before new GPU code runs on registered hardware; checkpoint/resume for runs `> ~1 h`; releases need confirmation-lane evidence and the owner's OK.

## Fast-process rule (owner, 2026-10-30, standing) — overrides older ceremony rules

Stronger models now; cut ceremony that does not pay for itself.

1. **Two lanes.** *Exploration* (default: screens, smokes, pilots, speed work): no pre-registration docs/amendments; at most `1` quick review, only if `> ~10` units or touching shared code; not release evidence, may be acted on immediately. *Confirmation* (runs deciding a release or strict gate only): full rigor — pass bars fixed pre-data, held-out/fresh worlds, paired stats, verified stop.
2. **One reviewer per change.** No repeated delta reviews of small mechanical fixes — if tests pass, ship. A Claude Opus agent implements (owner 2026-11-13; before that Carbon); a separate fresh Opus agent that did not write the change reviews (Carbon or Astra also allowed); never the author run (`STATUS.md` #2; in exploration, skip review if unavailable). Never wait.
3. **Fewer canaries.** Unchanged transport tree + job package (only parameters/seeds differ) → dispatch main run directly. New GPU code runs an `L4` shake-out before registered hardware; for `H100`/`A100` confirmation runs, `1` canary on `H100` *or* `A100` suffices (never block on `H100` `503`).
4. **Automatic mechanical retries.** Verified stop + `0` outputs + known mechanical class (transport timeout, upload cap, allocation refused) → auto-retry with a fresh identity up to package limit, no new review/packet cycle; report after.
5. **Transport registry holds several qualified trees.** Activating a fixed transport must never block or race jobs queued/running on the previous qualified tree.
6. **Dispatch.** Keep Colab runtimes busy (measured per-type caps `L4/T4/G4/A100=3`, `CPU=5`, unprobed `H100=1`, total ceiling `12` in `tools/ops/ready_queue.py`; poll scarce GPU stock `~1/min`, not `20 min`) from a ready queue; don't ask the owner about operational steps; run independent work in parallel.
7. **Always kept, any lane:** held-out/test data never used for training/tuning; honest reporting of measurements; verified stops and runtime cleanup; safety-prohibited actions excluded; releases require confirmation lane + normal sign-off.

## Implementers (owner 2026-11-09, standing)

Project code is implemented and reviewed only by **Argon** (`agy` CLI, own git worktree, local commits) and **Astra** (Codex); never use the Sol persona. **Owner's 2026-11-10 rules (`CAMPAIGN_LOG.md` `2026-11-10 11:35`):** the `agy` model is **Gemini 4 Carbon** (`gemini-4-carbon-*`, `agy` 1.5.5; `argon_*.sh` defaults to Carbon, `ARGON_FAMILY=argon` opts back, and "Argon" in older rules/logs means this `agy` lane). An author run never reviews its own build — Claude Opus 5.5 (as a separate fresh `agy` reviewer) reviews Carbon's builds, with a separate fresh `agy` run or Astra also allowed (the coordinator session does not review). **Owner's 2026-11-13 rule ("use opus agents for coding"), which replaces the implementer choice above:** code is written by Claude Opus agents (the coordinator's Agent tool with `model: "opus"`, each in its own git worktree, local commits, trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`); a different fresh Opus agent reviews; the coordinator session still neither writes nor reviews product code.

## Campaign authority (owner-ratified 2026-10-19; updated 2026-11-10) — overrides older text

T34 ExIt-B8 closed `2026-11-04 PT`; active tracks follow the `2026-11-10` owner ruling above ([docs/STATUS.md](docs/STATUS.md), [docs/INDEX.md](docs/INDEX.md)). Release/confirmation contracts on `main`, in order: [SUCCESS_CONTRACT.md](docs/research/final_run_20261019/SUCCESS_CONTRACT.md) · [DESIGN_V2_SHADOW.md](docs/research/final_run_20261019/DESIGN_V2_SHADOW.md) (+ Amendment 1) · [P7_STATISTICS.md](docs/research/final_run_20261019/P7_STATISTICS.md) · unsuperseded [FINAL_PLAN.md](docs/research/final_run_20261019/FINAL_PLAN.md).

- **Claims, comparator & release bars:** **Claim A** = bare network; **Claim B** = visibly labelled assisted hero (bare wins unresolved ties). Fixed-controller inference unit = paired world; comparator = `B8 s3` (`b8-s3`) on release bars; Watch product metrics bind per `SUCCESS_CONTRACT.md` §3. Kept for **release/confirmation** (not limiting training): Mac-only strict gate + serving qual (`docs/research/governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md` — Mac↔x86 float32 tie divergence), fresh paired banks, survival band v2, confirmation pre-data freezing, sequential Phase R default, batching code changes before ratifying. Never edit a sha256-bound governance/research doc (intents, receipts, `prepare`); add a sibling erratum or status file.
- **Served agents & rollbacks:**
  - **Watch:** `132`-session confirmed `exact-stop-l1-os` bank `watch-replay-bank-live-132-l1os-installed` (`SNAKE_WATCH_REPLAY_CONTROLLER=exact-stop-l1-os`, live bare `b8-s3` fallback) since 2026-11-06, upgrading `watch-replay-bank-live-132-l1` (`exact-stop-l1`).
  - **Play:** live bare `b8-s3` default (`Classic B8`) since 2026-10-17 + opt-in `Field: Classic B8 / Mixed` since 2026-11-04.
  - **Rollbacks & frozen lines:** `frp3-s12 + v8` (sha `eec144bf`); `SNAKE_SERVE_CHECKPOINT=champion` restores `champion+v8`; with checkpoint unset, `SNAKE_SERVE_VETO_VARIANT=v7`/`v5`/`v2` gives `champion` + variant. `vector61` is **FROZEN** (`67`-D sight default if reopened). RunPod: none active (if reopened: `>=4×` Mac wall-clock, cheap pods before `6–10×` serverless, approved spend/uploads only, always clean up).
- **Right tool first (owner, 2026-11-02):** before sizing/dispatching, decide CPU- vs GPU-bound and price units/result and wall time across Mac and Colab (`CPU-only`/`T4`/`L4`/`A100`/`H100`). Watch6/exact-stop/replay is CPU-only (`SNAKE_DQN_DEVICE=cpu`): A100 and L4 both give `12` vCPU at equal speed, A100 costs `~3.5x` L4, a Mac P-core is `~6x` a Colab vCPU; use A100/H100 for CPU work only as extra concurrent slots with unit cost stated in plan. Argon/Carbon output (records, summaries, commands) requires an independent claim-by-claim check (`~/.claude/skills/compute-choice/SKILL.md`).
- **Always saveable and resumable (owner, 2026-11-02):** every Colab run `> ~1 h` must checkpoint inside each work unit (not only between units) so any interruption loses `<= ~45 min`, resuming the newest verified checkpoint with a determinism test proving `resumed == uninterrupted` (bit-identical where science requires); omitting requires a written reason. Watch6 paired worlds use `research/world_checkpoint` (`job.py`/`job.sh`: checkpoints every `1,800` frames and on `SIGTERM`/`SIGUSR1`, refuses mismatched model/code/host).
- **Colab credits & approvals (owner, 2026-10-20 / 2026-10-24 / 2026-11-10, standing):** use Colab Pro+ credits whenever they speed up work or add parallelism via `research/colab_cli` (`QUALIFICATION.md`), `L4` default (all slots open for GPU training). Mac-only release gates stay on Mac (Path B timing/paced, strict gate, serving qual). **Units never block & no Colab budgets:** no unit ceilings; per-job timeouts/caps (incl. `gpu_sitl` B1b smoke & pilot) are runaway guards the operator raises and logs without judge rulings; only confirmation-lane pre-registered science caps need a pre-data judge amendment. Only science/pre-registration, integrity, safety, and release rules block. All Colab use is pre-approved; confirmation registrations/amendments go to `1` independent Fable/Opus judge (logged in `CAMPAIGN_LOG.md`, reported to owner); trying a new recipe after a stop needs an advisor design note. **Release/serving changes require the owner's OK** (`SUCCESS_CONTRACT.md` §4; `RULING_label_ordering.md` §4).

## Colab jobs (2026-10-18) — read before touching anything that runs on Colab

All **new** Colab work uses the Mac supervisor (`research/colab_cli/`) and official CLI (`--auth oauth2`, `COLAB_CLI_ASSIGN_RETRIES=0`, unique session names + per-lane state files), counting all assignments (incl. notebook workers) against `ACCOUNT_TYPE_CAPS` (`L4 = 3`, `T4 = 3`, `G4 = 3`, `A100 = 3` = `12` mixed GPUs measured 2026-11-10; `CPU = 5` measured 2026-11-02; `H100 = 1` unprobed cap until multi-H100 probe) and combined ceiling `ACCOUNT_ASSIGNMENT_CAP = 12` (CPU counts toward `12` until `12 GPU + 5 CPU` is probed; clamped to `6` while any active job uses a pre-2026-11-10 `cap <= 6` tree) in `research/colab_cli/`, `tools/ops/dispatch.py`, and `tools/ops/ready_queue.py`. Never call `colab new/stop/exec/upload/download` directly. The supervisor validates pinned commit/script/input hashes and a fresh absolute output dir, uploads (never mounts Drive), runs a lease-fenced subprocess, polls file telemetry with explicit timeouts, verifies bounded immutable exports on Mac, writes a create-only receipt, stops, and confirms disappearance from `colab sessions`.

Use `snake-dqn-colab-jobs` when preparing/reviewing transports. Runtime rules: no Mac-only inputs; Python `3.12` `uv` env; `cu126` (L4/A100/H100) or `cu128` (Blackwell); measured estimates + first-cap-wins wall/unit limits; deterministic vCPU use; `~2` min progress; fresh output dirs; complete checkpoints; hash-bound inputs/receipts. Check telemetry and terminal receipt (CLI exit code != job success). Never do ad-hoc manual retries inside a CLI session (only `dispatch.sh`'s `1` pre-allocation accounting-race retry and `ready_queue.py`/`auto_retry.py`'s bounded outcome-blind retries after verified stop + `0` outputs under Fast-process #4 are allowed), buy units, run `colab auth`/`drivemount`, or use Drive in CLI runtimes.

## Operations — Resume-by-default

[Ruling (2)](docs/research/advice/2026-10-21_big_lever/JUDGMENT_RESUME_POLICY.md#2-standing-requirement--idempotent-completed-units--verified-ledger-resume-skip) binds every Colab package (dev and registered). Before Mac preflight, offline-test deterministic idempotent exported units (namespace + frozen allocation index), atomic immutable payloads with `.sha256` sidecars and `.complete` markers published last, and a Mac-verified ledger that skips verified units and refuses foreign namespaces/allocations (empty ledger on attempt 1; bind/upload as input; never recompute verified units into results).

Every manifest dispatched through `tools/ops/dispatch.sh` must declare:

```json
{"resume": {"unit_kind": "episode", "ledger_path": "inputs/verified-ledger.json", "skip_verified": true, "max_outcome_blind_retries": 2}}
```

Strings must be nonempty, `skip_verified` boolean `true`, `max_outcome_blind_retries` integer `2`. Only an explicit `resume_exemption` (`kind`: `short-canary` `<= 40` min or `acceptance` `<= 60` min, nonempty `reason`, finite positive timeout) may replace it — never declare both ([tools/ops/README.md](tools/ops/README.md#resume-by-default)).

Define mechanical, outcome-blind retry triggers and per-attempt envelopes (pre-registered for confirmation runs): `1` original + `<= 2` completion attempts after verified mechanical failure, verified stop, and missing units (fresh manifest, output dir, receipt per attempt; prior evidence immutable; ledgers built from path/size/hash only). Only the Mac aggregator takes the look over verified units, refusing duplicate IDs/foreign namespaces (remote summaries stay sealed diagnostics; preserve `N` and pause points; trajectories resume only the latest verified eligible full-state snapshot). Canary proof must match current code and manifest pin before main dispatch. Claude/operator dispatches; implementer (Carbon/Astra) prepares and tests.

**Owner's 2× rule:** initial Colab unit caps, reserves, and timeouts default to `2×` ruling/forecast values (confirmation-lane science caps frozen pre-data; exploration wall/unit caps, incl. `gpu_sitl` B1b smoke & pilot, are runaway guards the operator may raise in a fresh manifest and log with no judge amendment). Record base and effective envelopes; never double measured estimates, rates, `N`, snapshot cadence, or the `2`-retry max, never raise a cap to bypass a science bar, and never edit immutable manifests.

## Code Style

- Line length `100`; `black` + `isort` (`profile: black`); Google docstrings; `.pre-commit-config.yaml` hooks; function signature type hints required (enforced in review; no `mypy` in `requirements-dev.txt` / pre-commit).

## Where things are

- **Status, goal, what is running:** [docs/STATUS.md](docs/STATUS.md) (one page). **What was learned:** [docs/FINDINGS.md](docs/FINDINGS.md).
- **Live research code:** top section of [research/INDEX.md](research/INDEX.md) — `research/arena_ppo` (PPO trainer, old grid game), `research/game_v3` (new continuous-heading game), `research/v3_ppo` (its trainer, on a branch until merged). The other ~165 folders under `research/` are finished experiments.
- **Product code (served Watch/Play, Apex/raster stack, web app):** project overview, architecture and state contracts, essential commands and key files are in [docs/PRODUCT_CODE_GUIDE.md](docs/PRODUCT_CODE_GUIDE.md). **Read it before editing `src/`, `web/` or `configs/`.**
- **Colab operations:** `tools/ops/dispatch.sh` (the only dispatch path), [tools/ops/README.md](tools/ops/README.md), `research/colab_cli/`.
- **History:** [docs/STATUS_ARCHIVE.md](docs/STATUS_ARCHIVE.md), [docs/research/history/INDEX.md](docs/research/history/INDEX.md), [docs/history/AGENTS_superseded_2026-11-09.md](docs/history/AGENTS_superseded_2026-11-09.md).
- **Use** `./venv/bin/python`; run heavy local work under `nice -n 15` with thread caps set to 1.
