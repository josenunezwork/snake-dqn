# Current state — updated 2026-11-11 PT

Start: [project index](INDEX.md) · [registered items](REGISTERED_ITEMS.md) · [implementer checklist](IMPLEMENTER_CHECKLIST.md) · [campaign log](research/final_run_20261019/CAMPAIGN_LOG.md) · [research roadmap (advice)](research/ROADMAP.md) · [documentation map](README.md) · [study packages](../research/INDEX.md) · [research record](research/INDEX.md) · [lineage history](research/history/INDEX.md) · [glossary](research/GLOSSARY.md).
What we learned (plain-language findings map): [FINDINGS.md](FINDINGS.md).

This page is the one-page status: goal, what is being built, what is running, what is served, and the binding rules. Everything older (release evidence detail, T34, October history) is in [STATUS_ARCHIVE.md](STATUS_ARCHIVE.md), moved there unchanged on 2026-11-11.

## Goal

Train the strongest possible slither.io-style agent. Mac serving bars gate **release** only; they do not limit what is trained (a large teacher distilled to a Mac-size student is allowed).

## What is being built (main track, opened 2026-11-10)

Large-scale from-scratch PPO self-play on GPU simulators, using the Colab slots. Results so far are in [FINDINGS.md](FINDINGS.md); nothing from this track is released or evaluated against the served agents yet.

| Code | What it is | State on 2026-11-11 |
|---|---|---|
| [`research/arena_ppo`](../research/arena_ppo/README.md) | PPO self-play trainer for the **old grid game** (v2), on the exact GPU simulator. Network designs `ego_only`, `shared_trunk`, `per_seat`; sizes S and M. | Merged. Used as the test bed. First 6 h runs done; second set running. `shared_trunk` is not funded further (equal per step, half the speed). |
| [`research/game_v3`](../research/game_v3/README.md) | **New game**: continuous heading, circular arena (radius 600), 24 snakes, 10 actions, GPU simulator and observation builder. | Merged. Refactored 2026-11-11 (`47e69258`: state-field registry with strict loading, `step` split into named phases mirrored by the CPU reference). CUDA matches CPU bitwise on an A100 before and after. Not yet rendered in Watch/Play. |
| [`research/v3_ppo`](../research/v3_ppo/README.md) | PPO self-play trainer for game v3. | Merged 2026-11-11; refactored the same day (`f23ca2b5`: `ppo.py` split into job, trainer, checkpoint, resume and export modules; one table for network sizes). Compiled speeds: A100 256 worlds S 22k / M 12.5k learner-steps/s, L4 128 worlds S 7.6k. First 4 h runs done (see FINDINGS). Known open bugs: final and periodic full checkpoints do not fit the Colab transfer budget (runs end INCOMPLETE; weights are saved in light checkpoints). |
| `research/arena_ppo/evaluate` (branch `carbon-arena-eval-1111`) | Evaluation of old-game policies against scripted opponents, an untrained network, other checkpoints, and B8 if its features can be reproduced exactly. | In progress. Until it lands, every training number is self-play length only. |

**Running on Colab:** game v3 training runs (small and medium networks, A100 and L4); the old-grid-game runs ended on 2026-11-11 and that game is now the test bed only. Job folders: `../snake-dqn-artifacts/big-lever/exp/argon-20261026/` (index of the whole artifacts tree: `../snake-dqn-artifacts/INDEX.md`).

**On hold:** `gpu_sitl` Step B1b pilot (search-in-the-loop on the B8 lineage), pinned at `b772f89f`.

**How work is done:** Gemini 4 Carbon implements in its own worktree; a separate Opus reviewer reads each round (only BLOCKER/MAJOR block a merge); the coordinator steers, dispatches Colab jobs through `tools/ops/dispatch.sh`, and writes docs. Scarce GPUs are polled with `../snake-dqn-artifacts/ops/scripts/gpu_hunt.sh`.

## Owner ruling 2026-11-10 (overrides conflicting older text)

Source: [`CAMPAIGN_LOG.md`](research/final_run_20261019/CAMPAIGN_LOG.md) `2026-11-10 PT 19:25`.

1. **Goal & release bars:** Train the strongest possible slither.io-style agent; Mac serving/strict/qualification bars gate **release** only (large teacher → Mac-size student distillation allowed).
2. **`gpu_sitl` Step B1b = exploration lane:** Smoke and pilot (`research/gpu_sitl`) may use any GPU (`L4` included); wall-time and unit caps are runaway guards the operator may raise and log with no judge amendment; bars, held-out namespaces, and seeds are unchanged ([status note](research/advice/2026-11-09_live_search_reopen_registration_argon.status_2026-11-10.md)).
3. **Kill rules close only the tested recipe:** Clauses such as `"permanently closes all learned-student work on ego2s-bv8 inputs"`, `"KILL_BRANCH_B ... 0 further Colab units and no sweeps"`, and whole-line `STOP_AND_MAINTAIN` triggers mean that recipe stops and a different recipe may be tried after an advisor design note; served agents stay as they are until a release.
4. **Scarce GPU stock polling:** Scarce GPU stock is polled about once a minute (`~1/min`), not every `20` minutes.
5. **New main track:** Large-scale from-scratch training on the exact GPU simulator, using all Colab slots (up to `12` concurrent runtimes under measured per-type caps `L4/T4/G4/A100 = 3`; `CPU = 5` measured; `H100 = 1` until probed; `ACCOUNT_ASSIGNMENT_CAP = 12` total ceiling until `12 GPU + 5 CPU` is probed), is opened (*design pending, advisor round 2026-11-10*).
6. **Always kept:** Held-out/test data never used for training or tuning; honest reporting; verified stops and cleanup; `L4` shake-out before new GPU code runs on registered hardware; checkpoint/resume for runs `> ~1 h`; releases need confirmation-lane evidence and the owner's OK.

## What is served (unchanged until a release)

- **Watch** = `132`-session confirmed `exact-stop-l1-os` replay bank (`watch-replay-bank-live-132-l1os-installed`, `SNAKE_WATCH_REPLAY_CONTROLLER=exact-stop-l1-os`), with live bare `b8-s3` as fallback. Released 2026-11-06.
- **Play** = live bare `b8-s3` (`Classic B8` default; `Field: Classic B8 / Mixed` opt-in).
- **How it runs (since 2026-11-07 PT):** persistent tmux session `snake-web` running `tools/ops/watch_server.sh`, which reads the `snake-web` entry of `.claude/launch.json` on every start and restarts the server if it exits (the Claude app had been stopping its preview server when sessions ended). Logs: `../snake-dqn-artifacts/ops/watch-server/server.log` (rotated, 3 × 100 MB). To apply a rollback: edit `launch.json`, then `tmux kill-session -t snake-web` and `tmux new-session -d -s snake-web -c <repo> "bash tools/ops/watch_server.sh"`. Do not also `preview_start snake-web` (port 8000 conflict).
- Release evidence, monitoring thresholds, rollback targets and the Play roster details: [STATUS_ARCHIVE.md → Current served setup](STATUS_ARCHIVE.md#current-served-setup-watch--play). They remain active.

## T34 CLOSED (2026-11-04 PT) — final state

T34 (search-teacher campaign on the B8 lineage) is closed by its pre-registered closeout audit. It shipped the Watch replay bank above; live search and search→bare distillation were closed. Full record: [STATUS_ARCHIVE.md](STATUS_ARCHIVE.md#t34-closed-2026-11-04-pt--final-state) and [lineage history](research/history/INDEX.md).

## Fast-process rule (owner, 2026-10-30, standing) — overrides older ceremony rules

Stronger models now; cut ceremony that does not pay for itself.

1. **Two lanes.** *Exploration* (default: screens, smokes, pilots, speed work): no pre-registration documents or amendments; at most one quick review, and only if the run costs > ~10 units or touches shared code; results are not release evidence but may be acted on immediately. *Confirmation* (only runs that decide a release or the strict gate): full rigor — pass bars fixed before data, held-out/fresh test worlds, paired statistics, verified stop.
2. **One reviewer per change.** No repeated delta reviews of small mechanical fixes — if tests pass, ship. Carbon (`agy`) implements; Claude Opus 5.5 (run as a separate fresh `agy` reviewer), a separate fresh `agy` run, or Astra reviews (`CAMPAIGN_LOG.md` `2026-11-10 11:35`; in exploration, skip review if unavailable). Never wait.
3. **Fewer canaries.** Unchanged transport tree + unchanged job package (only parameters/seeds differ) → dispatch the main run directly. New GPU code runs an L4 shake-out before registered hardware; for H100/A100 confirmation runs, one canary on H100 *or* A100 suffices (never block on an H100 proof when H100 returns 503).
4. **Automatic mechanical retries.** Verified stop + no outputs + known mechanical class (transport timeout, upload cap, allocation refused) → retry automatically with a fresh identity up to the package's limit, no new review/packet cycle; report afterwards.
5. **Transport registry holds several qualified trees.** Activating a fixed transport must never block or race jobs queued/running on the previous qualified tree.
6. **Dispatch.** Keep all Colab runtimes busy (`ACCOUNT_TYPE_CAPS`: measured `L4 = 3`, `T4 = 3`, `G4 = 3`, `A100 = 3` = `12` mixed GPUs on `2026-11-10`, measured `CPU = 5` on `2026-11-02`, unprobed `H100 = 1` until multi-H100 stock probe; total ceiling `ACCOUNT_ASSIGNMENT_CAP = 12` including CPU until `12 GPU + 5 CPU` is probed, clamped to `6` while any active job runs on a pre-`2026-11-10` `cap <= 6` transport tree in `research/colab_cli/accounting.py`, `tools/ops/ready_queue.py`, and `tools/ops/dispatch.py`; poll scarce GPU stock `~1/min`, not every `20 min`) from a ready queue; don't ask the owner about operational steps; do multiple requested things in parallel.
7. **Always kept, any lane:** held-out/test data never used for training or tuning; honest reporting of what was actually measured; verified stops and runtime cleanup; safety-prohibited actions excluded; releases need the confirmation lane plus normal sign-off.

---

## Campaign authority (owner-ratified 2026-10-19) — overrides older text

T34 ExIt-B8 closed `2026-11-04 PT`; active work follows the `2026-11-10` owner ruling at the top of this file (`gpu_sitl` Step B1b exploration + large-scale GPU from-scratch main track, *design pending, advisor round 2026-11-10*). Release/confirmation contracts on `main`, in order:
[SUCCESS_CONTRACT.md](research/final_run_20261019/SUCCESS_CONTRACT.md) ·
[DESIGN_V2_SHADOW.md](research/final_run_20261019/DESIGN_V2_SHADOW.md) (+ Amendment 1) ·
[P7_STATISTICS.md](research/final_run_20261019/P7_STATISTICS.md) · unsuperseded parts of
[FINAL_PLAN.md](research/final_run_20261019/FINAL_PLAN.md).

- **Superseded by them:** the "single bare network only" goal (now **Claim A**; **Claim B** = a visibly labelled assisted hero may also succeed; bare wins unresolved ties); "beat `frp3-s12 + v8` without a veto" (met by B8 s3, closed); "M3-C next"; five-training-seed inference for fixed controllers (unit = paired world); historical FRP3/champion "Tier 1 + Tier 2" guards as automatic T34 obligations (comparator is B8 s3 for release bars); RunPod for T34 (none); the Q0/Q1 budget envelope. Watch product metrics bind at release per contract §3.
- **Kept (release/confirmation):** Mac-only strict gate and serving qualification (gating release only, never training; large teacher distilled to a Mac-size student is allowed), fresh paired banks, survival band v2, pre-data freezing (for confirmation runs).
- **Path B v2 retention (re-ratified 2026-10-20 ~20:05 PT):** the registration ratified at `97232d51` (study `t34-retention-v2-operator-01`) was SUPERSEDED pre-data by the [bare-unpaced amendment](research/specs/2026-10-20_t34_retention_amendment_bare_unpaced.md) and is VOID for execution. Any execution of that study uses only the frozen worktree `/Users/josenunez/Projects/ml/snake-dqn-retention-v2-frozen-02` (detached at `81c54731a06f5a630bc1b95edca876832906a812`), study `t34-retention-v2-operator-02`, with [`research/t34_pathb/REGISTRATION_RETENTION_V2_RATIFIED.json`](../research/t34_pathb/REGISTRATION_RETENTION_V2_RATIFIED.json) (authoring checkout / `history_ledger.repo`: `/Users/josenunez/Projects/ml/snake-dqn-retention-v2-author`; ledger `research/t34_pathb/ledger/t34-retention-v2-operator-02.jsonl`). The Colab unpaced arm is dispatchable only at `81c54731`. Path B uses the relative boost floor (`0.5 × paired bare share <= paced <= 0.10`); Round 0 had its own pre-look boost-floor amendment (merge `30b2739a`). Timing gate only after Phase 0 finishes and the Round 0 Mac replay.
- **Colab credits & no-budget rule (owner, 2026-10-20, 2026-10-24 & 2026-11-10, STANDING):** use the owner's Colab Pro+ credits whenever they speed up work or add a parallel research lane — do not wait for the Mac or ask first. Go through the qualified CLI supervisor (`research/colab_cli/`, [colab-cli.md](colab-cli.md)), L4 by default (all Colab slots open for the GPU training track; poll scarce GPU stock `~1/min`, not `20 min`), price per useful result. Mac-only release gates stay on the Mac (Path B timing/paced tests, strict gate, serving qualification). Initial portfolio target (2026-10-20): spend ~1,160 Colab units by ~2026-11-20 (~39 u/day) on decision-producing lanes ([CAMPAIGN_LOG.md](research/final_run_20261019/CAMPAIGN_LOG.md)). **Units never block (2026-10-24 & 2026-11-10, STANDING):** Colab units are never a reason to decline, shrink, or delay work; only science/pre-registration, integrity, safety, and owner release rules may block; per-job caps/timeouts (including `gpu_sitl` Step B1b smoke and pilot) are runaway guards the operator raises and logs without judge rulings; only confirmation-lane science caps written into a pre-registration before data need a pre-data judge amendment. **No Colab budgets (2026-10-24 later, STANDING):** *"remove budget — fine using all Colab credits in 1 week if even possible"* — no unit budgets or ceilings; actively seek high-value uses.
- **Approvals (owner, 2026-10-20 & 2026-11-10):** all Colab use is pre-approved. Confirmation registrations/amendments go to one independent judge (Astra/Fable; **2026-10-20 ~19:05 PT, owner:** independent Opus stand-ins serve when quota-limited), recorded in [CAMPAIGN_LOG.md](research/final_run_20261019/CAMPAIGN_LOG.md), and reported to the owner; trying a different recipe after a recipe kill requires an advisor design note (`2026-11-10` ruling #3). **Release/serving changes still require the owner's explicit OK** ([SUCCESS_CONTRACT.md](research/final_run_20261019/SUCCESS_CONTRACT.md) §4; [RULING_label_ordering.md](research/advice/2026-10-20_a100/RULING_label_ordering.md) §4 — a judge may only recommend a release).

---

## Operating rules (current)

These rules replace older conflicting text (e.g. the 5× RunPod rule, "always serverless", or
`champion + v8` / `frp3-s12 + v8` as the served agent).

- **Served defaults (unchanged until a release):** Watch = 132-session confirmed `exact-stop-l1-os` replay bank (`watch-replay-bank-live-132-l1os-installed`, `SNAKE_WATCH_REPLAY_CONTROLLER=exact-stop-l1-os`, released 2026-11-06 after Bank-v2 `CONFIRM`; rollback `watch-replay-bank-live-132-l1` / `exact-stop-l1`) with live `b8-s3` fallback; Play = live bare `b8-s3` (`3e649783`) without a veto (`Classic B8` default, with opt-in `Field: Classic B8 / Mixed` against `frp3-s12 + v8` opponents; leaderboard partitioned by field). Rollbacks and monitoring thresholds are above in this file.
- **Implementers (owner, 2026-10-27 & 2026-11-10, standing):** project code is implemented and reviewed only by **Carbon** (Gemini 4 Carbon via `agy` CLI, own git worktree, local commits; called "Argon" in older rules/logs) and **Astra** (Codex). Claude Opus 5.5 (as a separate fresh `agy` reviewer), a separate fresh `agy` run, or Astra reviews Carbon's builds; authors never review their own builds. Do not use the Sol implementer persona (`AGENTS.md`, commit `1ad5dc95`).
- **Standing scope decisions (2026-10-17; updated 2026-11-10):**
  - *October redesign goal met and closed:* `b8-s3` beat `frp3-s12 + v8` without a veto.
  - *`vector61` FROZEN:* no new governed FRP studies on the 61-D vector line (if reopened, FRP-v5-S2's 67-D sight inputs are the default).
  - *Recipe-scoped kill rules & new GPU track (owner 2026-11-10):* kill rules close only the tested recipe — a different recipe may be tried after an advisor design note ([status note](research/advice/2026-11-09_live_search_reopen_registration_argon.status_2026-11-10.md)); large-scale from-scratch training on the exact GPU simulator using all Colab slots is open (*design pending, advisor round 2026-11-10*).
  - *"M3-C next" superseded:* the pre-release 2026-10-08 plan for Q drift, late decline, and under-boosting is historical provenance, not active work.
  - *Confirmation & runaway-guard process:* batch code changes before ratifying confirmation studies; [sequential Phase R](research/governance_amendment_sequential_phase_r_2026-10-07.md) is the default for new governed screens; owner's 2× rule doubles base Colab unit/timeout forecasts initially without doubling measured rates or `N` (exploration caps incl. `gpu_sitl` B1b smoke/pilot are runaway guards the operator may raise and log with no judge amendment).
- **Strict gates and serving qualification run on the Mac only (gating release only, never training):** Mac uses AC + open lid, thermal guard, `3×2`-thread Tier-1 slots (strict gates 2 slots; historical [compute policy](research/compute_policy_2026-10-02.md)). The [strict-on-RunPod amendment](research/governance_amendment_strict_on_runpod_2026-10-05.md) is **NOT RATIFIED and was superseded on 2026-10-08** ([status note](research/governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md)); it was never used. Mac↔x86 bitwise identity fails at random on exact float32 Q ties (including S2's 1/54 identity failure, which does not waive identity). Any future remote strict gate requires a new amendment with a tie-explaining identity check and a per-world single CPU model.
- **RunPod (not used in T34; standing guard if reopened):** use RunPod for a step only if projected wall-clock time (from measured timings incl. setup/transfers) is **≥4× faster than the Mac** (lowered from 5× on 2026-10-05). Waivers below 4× come from the owner per step and never carry over (recorded waivers: FRP-v5-H Phase R on pods at ~3.3–3.8× on 2026-10-07; FRP-v5-S2 Phase R on pods on 2026-10-07; FRP-v3 Phase R at ~4.7× was an exception under the old 5× bar and qualifies directly under 4×). Poll **cheap CPU pods first**; serverless is only a fallback (~6–10× catalog cost). No new pod work until the pod runner is stable.

---

