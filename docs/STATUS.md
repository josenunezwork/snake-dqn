# Current state — snapshot updated 2026-11-01; Watch bank updated 2026-11-01; live checkpoint released 2026-10-17

Start: [project index](INDEX.md) · [research roadmap (advice)](research/ROADMAP.md) · [documentation map](README.md) · [study packages](../research/INDEX.md) · [research record](research/INDEX.md) · [glossary](research/GLOSSARY.md).

This is a dated documentary snapshot, not a live job or deployment check. The owner-ratified campaign
documents and standing rules below take precedence over older launch notes and external memory; this
file is updated when they change.

---

## Current served setup (Watch & Play)

### Watch: 168-session exact-search replay bank (owner "Switch now", 2026-10-30 ~16:10 PT; expanded 2026-10-31 & 2026-11-01)

Watch plays **Colab-precomputed replays** (`exact-stop-l2-prime` and `exact-stop-l1` search heroes over
product Watch6 v2, `5× frp3-s12 + v8` opponents, 6,000-frame sessions; pipeline in
[`research/watch_replay/`](../research/watch_replay/)) visibly labelled **"Recorded assisted hero"**:

- **Active bank:** `snake-dqn-artifacts/watch-replay-bank-live-168` (**168 verified sessions** = `132`
  `exact-stop-l2-prime` sessions from banks `01–08` and `10–12` + `36` all-length `exact-stop-l1`
  sessions from banks `14l1–16l1`, verified `36/36` on distinct seeds and switched 2026-11-01 in
  `395eacbb`; the previous 132-session bank is kept at `watch-replay-bank-live-132` for rollback;
  older 48/84/108-session copies were deleted 2026-11-01). Bank-02 descriptive stats: mean 412 mass/session, 0 deaths in the 5
  sessions reaching `L > 1,000`.
- **Launch & fallback:** `.claude/launch.json` (`snake-web`) sets `SNAKE_WATCH_SOURCE=replay` and
  `SNAKE_WATCH_REPLAY_DIR`. **Live bare `b8-s3` remains the automatic fallback** (empty, invalid, or
  exhausted bank), and **Play always stays live** (`b8-s3` without a veto).
- **Watch replay rollback & serving caveats:** unset `SNAKE_WATCH_SOURCE` (or set
  `SNAKE_WATCH_SOURCE=live`) and restart. Note: [`GameSession`](../web/backend/session.py)
  (`__init__`, `reset_game`, and `set_mode`) currently selects `ReplayBank` whenever
  `SNAKE_WATCH_SOURCE=replay` and `checkpoint is None` (without checking
  `served_checkpoint == "b8-s3"`, `watch_roster == "mixed"`, `Watch6`, or prior `set_epsilon`/`set_food`
  live opt-outs across `reset_game`), so any env-var checkpoint/roster rollback
  (`SNAKE_SERVE_CHECKPOINT=frp3-s12|champion` or `SNAKE_SERVE_WATCH_ROSTER=all_b8`) must also unset
  `SNAKE_WATCH_SOURCE` (or set `=live`) until `_replay_opted_out` and the multi-entrypoint
  mixed-Watch6 `b8-s3` guard tracked in [research/ROADMAP.md](research/ROADMAP.md) §2.1 land.
- **`exact-stop-l1` validator, mixed-controller bank & first-cycle ordering (`73974411` & `395eacbb`, 2026-11-01):**
  `web/backend/watch_replay.py` and `research/watch_replay/` accept `"exact-stop-l1"` (`L >= 1`,
  `interval = 5`, `H = 300`) in `ASSISTED_CONTROLLERS` alongside `"exact-stop-l2-prime"`, and
  `web/serve.py` was restarted at `b5b63cd1` to serve `watch-replay-bank-live-168` (`132`
  `exact-stop-l2-prime` + `36` `exact-stop-l1` `.watch.gz` recordings). Two current `ReplayBank`
  details tracked in [research/ROADMAP.md](research/ROADMAP.md) §2.1: (1) `ReplayBank._prefetch`
  walks `sorted(root.glob("*.watch.gz"))[:1024]` on cycle 1 before `SystemRandom().shuffle` runs on
  cycle 2 (`132 × 250 s = 9.17 h` of `bank01–12` `l2-prime` play before `bank14l1–16l1` after a
  restart, fixable via injectable `shuffle_discovery=True`), while `verify_payload` sleeps
  `stop.wait(0.001)` per frame (`~8–10 s` cold load, fixable via a `64`-frame cooperative yield
  without `24 fps` GIL stalls); and (2) restricting playback to a single confirmed controller
  (`exact-stop-l2-prime`-only via `watch-replay-bank-live-132`, or `exact-stop-l1`-only after
  confirmation) requires a single-controller `SNAKE_WATCH_REPLAY_DIR` or landing
  `SNAKE_WATCH_REPLAY_CONTROLLER`.
- **Product efficacy measurement status:** synchronous `exact-stop-l2-prime` efficacy vs bare B8 in the
  product H14400 setting is **not yet confirmed and is currently running in Watch6 V3**
  ([`research/watch6_search_efficacy_v3/`](../research/watch6_search_efficacy_v3/)), and `exact-stop-l1`
  H14400 efficacy (`Watch6-L1-V1` in [research/ROADMAP.md](research/ROADMAP.md)) has not yet entered
  confirmation. The earlier Watch6 V2
  96-pair confirmation ([`research/watch6_search_efficacy/`](../research/watch6_search_efficacy/)) ended
  **`INVALID_OR_INCOMPLETE`** on 2026-10-31 (shard 6 / index 5 exhausted original + 2 completions on
  pre-start transport failures; Astra rulings `t34-sol/watch6-shard6-ruling/RULING_SHARD6.md` and
  `t34-sol/watch6-v3-registration-q/ANSWER.md` denied extra attempts and an 88+8 V3; running shards
  stopped 10:10 PT with verified stops; zero outcomes were read).

### Play & live Watch fallback: released bare `b8-s3` checkpoint (2026-10-17)

- **Released checkpoint:** **`b8-s3` hero without a veto** (2026-10-17), checkpoint SHA prefix `3e649783`.
  Evidence: T7 fresh-bank Phase R `STRICT_PASS` and Tier 2 PASS (margin `0.034` at SE `0.023`), followed
  by all eight T12 serving checks (check 4 cleared by T12B) and T15 mixed-roster qualification.
  Release claim: **more mass at equal survival against the previously served agent, with no veto**
  (not a safety claim). See [release, startup verification and rollback record](research/b8_s3_release_2026-10-17.md).
- **Default live Watch roster:** **mixed** (Watch-only; in Play mode, `1` human faces `N` — default `5`,
  adjustable `1–11` — live bare `b8-s3` AI opponents) — in live Watch, bare B8 hero at slot 0 faces
  **`frp3-s12 + v8` opponents** (SHA prefix `eec144bf`) at slots `1..N−1`. T13b's fixed six-snake rule selected mixed (death gap
  `0.096/1k <= 0.10`, p95-length gap `127 > 100` cells); T15 passed mixed serving qualification
  (`SERVING_PASS`), and the owner approved the default switch in T15B. See
  [T13b result](research/t13b_six_snake_roster_2026-10-17_results.md),
  [T15 qualification](research/t15_mixed_roster_qualification_2026-10-17.md), and
  [T15B mixed Watch release erratum](research/b8_s3_release_erratum_mixed_watch_2026-10-17.md).
- **Pin-gating & checkpoint rollbacks:** the default is pin-gated; B8 refusal falls back to verified
  `frp3-s12 + v8`, then `champion` (SHA prefix `43d4e2c5`), with a logged reason.
  - Roster rollback: `SNAKE_SERVE_WATCH_ROSTER=all_b8` (unsetting selects mixed).
  - Checkpoint rollback: `SNAKE_SERVE_CHECKPOINT=frp3-s12` or `SNAKE_SERVE_CHECKPOINT=champion` with
    remaining serving env at defaults. With no checkpoint named, `SNAKE_SERVE_VETO_VARIANT=v7` (also
    `v5`/`v2`) retains `champion` + that variant; explicit `frp3-s12` with `v7` remains unwrapped.
- **Serving boundaries & monitoring:** B8 was trained with `v8` labels and is served without a veto.
  Self-play rosters are ungated; pre/post-release leaderboard rows are not comparable. `ego2s` ceilings
  are `24 fps` through 6 driven AI (`Watch6`/`Play6`) and `12 fps` through 12 (`Watch12`/`Play12`).
  Train mode is refused and unqualified. Monitor for `>= 7` days and `100,000` live Watch hero
  decisions; roll back on fallback/`ego2s` exceptions, roster-specific hero deaths/1k `> 0.544` at
  mixed Watch6 (all-B8 thresholds: `> 0.848` at Watch6, `> 0.992` at Watch8, `> 1.536` at Watch12),
  or matched default-speed p95 `> 1.5×` qualification ([monitoring erratum](research/b8_s3_monitoring_erratum_2026-10-17.md)).
  Watch8's death baseline does not qualify its all-B8 latency; mixed Watch6/8/12 have T15 latency
  baselines, while mixed Watch8/12 have no registered death baseline. Observation windows are separate
  by roster mode and size (initial owner check-in window: 2026-10-24).

---

## Active campaign state & Oct 31 / Nov 1 results (T34 ExIt-B8 — 2026-11-01)

For synthesized current-round and later-round research plans (**advice, not decisions**), see
[research/ROADMAP.md](research/ROADMAP.md).

- **Confirmation running — Watch6 V3 exact-stop efficacy ceiling**
  ([`research/watch6_search_efficacy_v3/`](../research/watch6_search_efficacy_v3/)): `24 × 4` L4 `W = 4`
  shards (`96` fresh paired `H = 14,400` Watch6-v2 sessions: `b8-s3` bare vs synchronous
  `exact-stop-l2-prime` at `L >= 600`, `H = 300`, `5× frp3-s12 + v8`, frame-exact live/`GridBatchSim`
  mirror; governs `exact-stop-l2-prime` only, not the `36` `exact-stop-l1` recordings in
  `watch-replay-bank-live-168`). Accounting adapter **V6.1** ratified (`75471246`; at `05:01 PT` on
  2026-11-01: `14/24` `PASS` shards `01..08, 10, 11, 13..16`, `6/24` active `17, 18, 20..23`, and
  `4/24` V6.1-accepted mechanical failures `00, 09, 12, 19`). **Linear-retry & bytecode guards:**
  `final_look.py:_verify_shard_lineage` rejects forked attempt trees sharing a parent receipt, so
  shards `09, 19` must use **only** their already-staged `ops/auto-retry` packets (**never** run
  `recover.py` on `09, 19`), shards `00, 12` dispatch `recovery-shard-{00,12}-01` via
  `python -B -m tools.watch6_v3_accounting.dispatch`, and the sole final look **must be invoked under
  `python -B`** (`./venv/bin/python -B -m tools.watch6_v3_accounting.final_look ...`) from clean
  `snake-dqn-argon-v3-accounting` because `_forbid_pycache_directory` fails closed on any
  `__pycache__`/`.pyc` inside the V3 closure.
- **Completed exploration — Watch6 all-length screen (`ADOPT` for new replay banks)**
  ([`research/watch6_alllen_screen/`](../research/watch6_alllen_screen/), readout
  `t34-sol/argon-alllen-a100/analysis.json`, `N = 48` paired 6,000-frame Watch6-v2 sessions):
  - `bare` (`b8-s3`): mass **`351.1`** (`CI95 [320.2, 381.9]`), deaths **`0.288/1k`** (`83` deaths: `53` at `L < 600`, `30` at `L >= 600`, `21` at `L >= 800`).
  - `l2_prime_600` (served `L >= 600`): mass **`461.2`** (`+110.1` vs `bare`, `CI95 [+72.1, +148.1]`), deaths **`0.160/1k`** (`-44.6%` vs `bare`; `46` deaths: `34` at `L < 600`, `12` at `L >= 600`, `11` at `L >= 800`), `449.6` mean branches/session.
  - **`exact_stop_alllen` (`L >= 1`, `interval = 5`, `H = 300`):** mass **`559.4`** (**`+208.4` vs `bare`**, `CI95 [+170.1, +246.7]`; **`+98.3` vs `L >= 600`**, `CI95 [+50.6, +146.0]`), deaths **`0.052/1k`** (**`-81.9%` vs `bare`**; **`-67.4%` / `-0.108/1k` vs `L >= 600`**, `CI95 [-0.161, -0.054]`; `15` total deaths: **`1` at `L < 600`**, **`14` at `L >= 600`**, **`13` at `L >= 800`**), **`2.82×` search branches** (`1,268.5` mean branches/session, `0.142%` `L < 600` overrides, `mean_max_length = 1,096.0`).
  - Pre-stated adoption rule (`>= +15.0` incremental mass and `<= 0.0` `deaths/1k` vs `L >= 600`) met: commit `73974411` added `"exact-stop-l1"` to `research/watch_replay/` and `web/backend/watch_replay.py`, and A100 `W = 6` banks `14l1–16l1` (`3 × 12` = `36/36` verified sessions on disjoint seeds in **`3.03–3.07 h/bank`, `~61.5 A100 u` observed total** at `6.73 u/h`; `t34-sol/argon-banks-l1/SUMMARY.md`) were verified and merged into `watch-replay-bank-live-168` (`395eacbb`, `web/serve.py` restarted on `b5b63cd1`). Exploration screen only — confirming `exact-stop-l1` as a Claim-B release still requires prospective H14400 confirmation (`Watch6-L1-V1`).
- **Completed parity — GPU `DeviceSim` v2 parity `PASS` on A100**
  ([`research/devicesim_v2_parity/`](../research/devicesim_v2_parity/),
  `t34-sol/argon-devicesim-v2/packet-a100-v2-parity-01/main-run-01/receipt.json`): **`0` mismatches**
  across **`1,536,000` snake-ticks** (`E = 256 × 1,000` Watch6-v2 ticks, commit `b500d111`, subject to
  `<= 1e-4` CUDA FP32 near-tie Q re-decision checks), following `553 passed` local tests and Mac
  `E = 4 × 8` preflight `PASS`.
- **Completed Oct 30 Claim-A exploration receipts awaiting Mac readout (and `spatial-student-pilot` fit canceled):**
  - `b8-head-closedloop` (`origin/astra-doom-auc-smoke`, `N = 192` v1 H5000 worlds, `m in {0, 10, 25, 50}`, run without `boost_decisions` logging): A100 receipt **`PASS`** (`t34-sol/b8-head-closedloop/main-01/output/receipt.json`), awaiting `< 3 min` Mac `closedloop_readout.py` (plus a `0 u`, `10 min` v2 boost `[0.5%, 10%]` check if any arm survives).
  - `spatial-qbytes-01` (`origin/astra-spatial-student-pilot`): A100 Q-byte ruler receipt **`PASS`** (`t34-sol/spatial-qbytes-fix/t34-spatial-qbytes-01/ruler-output-a100-fallback/receipt.json`), awaiting Mac parity/timing audit; the `40–48 A100 u` `spatial-student-pilot` GPU fit is canceled (`0 u`, retaining compiled `.so` caches) per `t34-sol/student-direction/{astra,fable}.md` and T4 Gate-1 `KILL_T4`.
- **Colab concurrency caps:** **`6` concurrent runtimes total** (`ACCOUNT_ASSIGNMENT_CAP = 6`, tree `36285bc6` active in `ops/colab-cli-qualified.json` and verified live at `6/6` concurrent runtimes — `3 A100 @ 6.73 u/h + 3 L4 @ 1.94 u/h = 26.01 u/h`, balance `~2,154.66 u`, all `3` A100 slots now free after `14l1–16l1` finished), at most **`3` concurrent L4** (`ACCOUNT_L4_CAP = 3` in `tools/ops/ready_queue.py`).
- **Closed Oct 29–31 learned-policy, critic, and macro-fallback verdicts (`KILL` / `FAIL`):**
  - **TRAIN-01 GPU-resident PPO** ([`research/sandbox_a_resident/`](../research/sandbox_a_resident/),
    `t34-sol/train01-closeout/SUMMARY.md`) — **Closed** after `384` updates (`100.7M` rows): H5000 equal-mix
    paired mass **`+8.09`** (`CI95 [-18.66, +34.85]`; scripted `+26.32`, frozen `+2.85`, mixed `-4.90`;
    update-320 peak `+14.23` `[-11.86, +40.32]`) vs bar `>= +40`; no strict-gate nomination.
  - **C/T Stage-2 distillation** (`t34-sol/ct-stage2-verdict/SUMMARY.md`) — **`FAIL_DISTILLATION_RETENTION_OR_SAFETY`**
    (`22/32` `68.75%` deaths rescuable, `22/32 <= 65` ticks, indep-suffix H300 `+66.77`; student retention
    `46.9–50.0%` with `0 pp` over `frozen_phi`, topology AUC `0.5752 < 0.83`, ordinary flips `44.4–64.5%`
    vs `<= 1%`, `3–5` lethal ordinary flips).
  - **Doom U-Net** (`t34-sol/doom-result-01/result.json`) — **`GRAY` / lost to B8 features** (`83 × 145` spatial U-Net late per-action doom ROC-AUC `0.753` vs 33.7k B8-feature head `0.810`, `Δ = -0.056`, `CI95 [-0.091, -0.023]`; all-strata death-lookback `0.7626` vs `0.8226`).
  - **B8 advantage head offline OOF** (`t34-sol/b8-head-test/RESULT.md`) — **Selectivity `FAIL`** (late H300
    mass `+30.09`, `LB90 +23.96`, but `26.42%` late rescue vs `>= 35%` and `97.60%` non-death flips vs `<= 15%`).
  - **T4 Gate-1 privileged factored critic** ([`research/t4_gate1/`](../research/t4_gate1/),
    `t34-sol/argon-t4-gate1-fix2/queue-03/READOUT.json`) — **`KILL_T4`** (`0/8` bars on `384` disjoint
    Watch6-v2 worlds: `L >= 600` residual EV `+0.0266`, death-root H300 rescue `+19.99`, mature doom AUC
    `0.5617`, ordinary H300 mass `-22.21`, lethal flips `3.13%` -> no clean-slate GPU RL or `83 × 145` critic distillation).
  - **R6 early-game `L < 600` safe-harvest census** ([`research/r6_l600_census/`](../research/r6_l600_census/),
    `t34-sol/argon-mac-checks/r6-census/RESULT.json`) — **`KILL`** (`64` worlds / `384` roots: `34.90%`
    `134/384` oracle-harvestable roots `>= +5` H500 mass, `7.69%` non-bare H500 lethal flips, 5-fold OOF
    ROC-AUC `0.474–0.521`, `0` passing thresholds).
  - **R7 macro-option fallback** ([`research/r7_macro_fallback/`](../research/r7_macro_fallback/),
    `t34-sol/argon-r7-repack/R7_READOUT.json`) — **`KILL`** (`3/5` bars on C/T train `1-step-all-die` roots:
    `0.316 ms` Mac 1-thread p95, `33.0%` vs `9.0%` `L >= 800` `all-die` rescue, but `+3.85 pp < +10.0 pp`
    total `L >= 600` lift and `2 > 0` newly caused ordinary lethal flips from aggressive `O3`/`O5`; `0` assessment roots opened).

---

## Fast-process rule (owner, 2026-10-30, standing) — overrides older ceremony rules

Stronger models now; cut ceremony that does not pay for itself.

1. **Two lanes.** *Exploration* (default: screens, smokes, pilots, speed work): no pre-registration documents or amendments; at most one quick review, and only if the run costs > ~10 units or touches shared code; results are not release evidence but may be acted on immediately. *Confirmation* (only runs that decide a release or the strict gate): full rigor — pass bars fixed before data, held-out/fresh test worlds, paired statistics, verified stop.
2. **One reviewer per change.** No repeated delta reviews of small mechanical fixes — if tests pass, ship. If a reviewer is unavailable (e.g. Argon quota), use an Opus/Fable stand-in or, in exploration, skip review. Never wait.
3. **Fewer canaries.** Unchanged transport tree + unchanged job package (only parameters/seeds differ) → dispatch the main run directly. One canary on H100 *or* A100 suffices; never block on an H100 proof when H100 returns 503.
4. **Automatic mechanical retries.** Verified stop + no outputs + known mechanical class (transport timeout, upload cap, allocation refused) → retry automatically with a fresh identity up to the package's limit, no new review/packet cycle; report afterwards.
5. **Transport registry holds several qualified trees.** Activating a fixed transport must never block or race jobs queued/running on the previous qualified tree.
6. **Dispatch.** Keep all Colab runtimes busy (measured account cap 6 total, L4 sub-cap 3 in `tools/ops/ready_queue.py` since 2026-10-31) from a ready queue; don't ask the owner about operational steps; do multiple requested things in parallel.
7. **Always kept, any lane:** held-out/test data never used for training or tuning; honest reporting of what was actually measured; verified stops and runtime cleanup; safety-prohibited actions excluded; releases need the confirmation lane plus normal sign-off.

---

## Campaign authority (owner-ratified 2026-10-19) — overrides older text

Active campaign: **T34 ExIt-B8**. Binding documents on `main`, in order:
[SUCCESS_CONTRACT.md](research/final_run_20261019/SUCCESS_CONTRACT.md) ·
[DESIGN_V2_SHADOW.md](research/final_run_20261019/DESIGN_V2_SHADOW.md) (+ Amendment 1) ·
[P7_STATISTICS.md](research/final_run_20261019/P7_STATISTICS.md) · unsuperseded parts of
[FINAL_PLAN.md](research/final_run_20261019/FINAL_PLAN.md).

- **Superseded by them:** the "single bare network only" goal (now **Claim A**; **Claim B** = a visibly labelled assisted hero may also succeed; bare wins unresolved ties); "beat `frp3-s12 + v8` without a veto" (met by B8 s3, closed); "M3-C next"; five-training-seed inference for fixed controllers (unit = paired world); historical FRP3/champion "Tier 1 + Tier 2" guards as automatic T34 obligations (comparator is B8 s3 for every bar); RunPod for T34 (none); the Q0/Q1 budget envelope. Watch product metrics are binding per contract §3.
- **Kept:** Mac-only strict gate and serving qualification, fresh paired banks, survival band v2, pre-data freezing (for confirmation runs).
- **Path B v2 retention (re-ratified 2026-10-20 ~20:05 PT):** the registration ratified at `97232d51` (study `t34-retention-v2-operator-01`) was SUPERSEDED pre-data by the [bare-unpaced amendment](research/specs/2026-10-20_t34_retention_amendment_bare_unpaced.md) and is VOID for execution. Any execution of that study uses only the frozen worktree `/Users/josenunez/Projects/ml/snake-dqn-retention-v2-frozen-02` (detached at `81c54731a06f5a630bc1b95edca876832906a812`), study `t34-retention-v2-operator-02`, with [`research/t34_pathb/REGISTRATION_RETENTION_V2_RATIFIED.json`](../research/t34_pathb/REGISTRATION_RETENTION_V2_RATIFIED.json) (authoring checkout / `history_ledger.repo`: `/Users/josenunez/Projects/ml/snake-dqn-retention-v2-author`; ledger `research/t34_pathb/ledger/t34-retention-v2-operator-02.jsonl`). The Colab unpaced arm is dispatchable only at `81c54731`. Path B uses the relative boost floor (`0.5 × paired bare share <= paced <= 0.10`); Round 0 had its own pre-look boost-floor amendment (merge `30b2739a`). Timing gate only after Phase 0 finishes and the Round 0 Mac replay.
- **Colab credits & no-budget rule (owner, 2026-10-20 & 2026-10-24, STANDING):** use the owner's Colab Pro+ credits whenever they speed up work or add a parallel research lane — do not wait for the Mac or ask first. Go through the qualified CLI supervisor (`research/colab_cli/`, [colab-cli.md](colab-cli.md)), L4 by default, price per useful result. Mac-only work stays on the Mac (Path B timing/paced tests, strict gate, serving qualification). Initial portfolio target (2026-10-20): spend ~1,160 Colab units by ~2026-11-20 (~39 u/day) on decision-producing lanes ([CAMPAIGN_LOG.md](research/final_run_20261019/CAMPAIGN_LOG.md)). **Units never block (2026-10-24, STANDING):** Colab units are never a reason to decline, shrink, or delay work; only science/pre-registration, integrity, safety, and owner release rules may block; per-job caps/timeouts are runaway guards (the operator raises the guard and logs it, no judge rulings over unit shortfalls; caps written into a pre-registration before data still need a pre-data judge amendment). **No Colab budgets (2026-10-24 later, STANDING):** *"remove budget — fine using all Colab credits in 1 week if even possible"* — no unit budgets or ceilings; actively seek high-value uses.
- **Approvals (owner, 2026-10-20):** all Colab use is pre-approved. Confirmation registrations/amendments go to one independent judge (Astra/Fable; **2026-10-20 ~19:05 PT, owner:** independent Opus stand-ins serve when quota-limited), recorded in [CAMPAIGN_LOG.md](research/final_run_20261019/CAMPAIGN_LOG.md), and reported to the owner. **Release/serving changes still require the owner's explicit OK** ([SUCCESS_CONTRACT.md](research/final_run_20261019/SUCCESS_CONTRACT.md) §4; [RULING_label_ordering.md](research/advice/2026-10-20_a100/RULING_label_ordering.md) §4 — a judge may only recommend a release).

---

## Operating rules (current)

These rules replace older conflicting text (e.g. the 5× RunPod rule, "always serverless", or
`champion + v8` / `frp3-s12 + v8` as the served agent).

- **Served defaults:** Watch = 168-session exact-stop replay bank (`watch-replay-bank-live-168`: `132` `exact-stop-l2-prime` banks `01–08, 10–12` + `36` `exact-stop-l1` banks `14l1–16l1`) with live `b8-s3` fallback; Play = live bare `b8-s3` (`3e649783`) without a veto. Rollbacks and monitoring thresholds are at the top of this file.
- **Implementers (owner, 2026-10-27, standing):** project code is implemented and reviewed only by **Argon** (Gemini 4 Argon via Antigravity CLI, own git worktree, local commits) and **Astra** (Codex). The reviewer is always the other one; authors never review their own builds. Do not use the Sol implementer persona (`AGENTS.md`, commit `1ad5dc95`).
- **Standing scope decisions (2026-10-17):**
  - *October redesign goal met and closed:* `b8-s3` beat `frp3-s12 + v8` without a veto.
  - *`vector61` FROZEN:* no new governed FRP studies on the 61-D vector line (if reopened, FRP-v5-S2's 67-D sight inputs are the default).
  - *Late-death bare-network training arms closed:* reopen only with new evidence of a different mechanism or observation/architecture ([campaign retrospective](research/redesign_campaign_retrospective_2026-10-17.md)).
  - *"M3-C next" superseded:* the pre-release 2026-10-08 plan for Q drift, late decline, and under-boosting is historical provenance, not active work.
  - *Confirmation process:* batch code changes before ratifying; [sequential Phase R](research/governance_amendment_sequential_phase_r_2026-10-07.md) is the default for new governed screens; existing fixed-N studies keep their registered rules; owner's 2× rule doubles base Colab unit/timeout runaway guards before data without doubling measured rates or `N`.
- **Strict gates and serving qualification run on the Mac only:** Mac uses AC + open lid, thermal guard, `3×2`-thread Tier-1 slots (strict gates 2 slots; historical [compute policy](research/compute_policy_2026-10-02.md)). The [strict-on-RunPod amendment](research/governance_amendment_strict_on_runpod_2026-10-05.md) is **NOT RATIFIED and was superseded on 2026-10-08** ([status note](research/governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md)); it was never used. Mac↔x86 bitwise identity fails at random on exact float32 Q ties (including S2's 1/54 identity failure, which does not waive identity). Any future remote strict gate requires a new amendment with a tie-explaining identity check and a per-world single CPU model.
- **RunPod (not used in T34; standing guard if reopened):** use RunPod for a step only if projected wall-clock time (from measured timings incl. setup/transfers) is **≥4× faster than the Mac** (lowered from 5× on 2026-10-05). Waivers below 4× come from the owner per step and never carry over (recorded waivers: FRP-v5-H Phase R on pods at ~3.3–3.8× on 2026-10-07; FRP-v5-S2 Phase R on pods on 2026-10-07; FRP-v3 Phase R at ~4.7× was an exception under the old 5× bar and qualifies directly under 4×). Poll **cheap CPU pods first**; serverless is only a fallback (~6–10× catalog cost). No new pod work until the pod runner is stable.

---

## Historical snapshots and milestones (superseded as live status)

> **Historical record only.** The sections below preserve exact commit SHAs, receipt paths, judge
> rulings, and pre-release qualification steps for provenance. They do not describe current live
> jobs or authorize resuming superseded plans.

### Historical snapshot — 2026-10-27 PT development state (`main` `aa5d6dfc`)

Artifact paths below are relative to `/Users/josenunez/Projects/ml/snake-dqn-artifacts/`. No strict
pass, serving qualification, or release was established by this snapshot.

- **Direction shift (2026-10-26):** `F5` closed before any fit by owner decision (`81863023`;
  [CAMPAIGN_LOG.md](research/final_run_20261019/CAMPAIGN_LOG.md) closure row dated 2026-10-26). Work
  redirected to GPU-resident training and Mac served search. The `A-3` throughput result (`88ba3f56`)
  measured collection + PPO throughput only, not skill or live-game identity.
- **GPU training (`TRAIN-01`, simulator-D development; closed 2026-10-30 at `384` updates without nomination — see active campaign section above):** At 2026-10-27 23:56 PT, packet 15
  (runtime/seal/verification `5c0a101a` / `a51380c1` / `6e54ed4c`, log
  `ops/logs/sandbox-a-train01-20261027-15.log`) had completed finite learner updates through update
  16 with diagnostic-update-16 proxy in progress (`sandbox-a-resident/train01-attempt-15/output/receipt.json`
  not yet written), and packet 16 was operator-reported in build; prospective ruling
  `t34-sol/astra-rule-argon-1027/RULING.md` (update 16 did not expand the nomination set; no 384-update
  endpoint or B8 efficacy claim). Earlier `TRAIN-01` receipts in
  `sandbox-a-resident/train01-attempt-{05,07,08,09,10,11,13}/output/receipt.json` all recorded
  verified stop and `cleanup_error=null`: `05/07/10` `INCOMPLETE` (supervisor interruption), `08/09`
  `INCOMPLETE` (`RPC_EXEC_POLL_TIMEOUT`), `11` `INCOMPLETE` (`CHILD_NONZERO_EXIT` after non-finite
  PPO actor loss; preserved iteration-0000 full state), `13` `INCOMPLETE` (`UPLOAD_CAP` before resume
  staging); `06/12/14` had no executed receipts. Compact resume in `14/15` passed CPU smoke; Astra's
  `t34-sol/astra-review-train01-14/REVIEW.md` rejected a MAJOR finite-input KL overflow, repaired in
  `astra-review-train01-15/REVIEW.md` (`ACCEPT-WITH-FIXES`, minor saturated KL/entropy gradient note).
- **Mac served search (2026-10-26–27):** Earlier host timing attempts remained `INVALID` under
  `served-tick-20261026/`, and `desktop-13` refused observation binding pre-panel. Prospective
  rulings: `t34-sol/astra-guard-supervisor/RULING.md` and `astra-desktop-apps/RULING.md`. Desktop
  fixes `a3799098`, merge `2c748fcb`, and protected-service precedence `6fa8e658` received `ACCEPT`
  reviews (`t34-sol/astra-review-argon-desktop4/REVIEW.md`, `astra-review-desktop-merge/REVIEW.md`,
  `astra-review-desktop-live2/REVIEW.md`). Opt-in authoritative game/DTO split (`8ffc34d0`,
  `59217589`) and `W1/K1` contention arm (`892bcd1e`, `709ccdf3`) received accepted source repairs
  (`big-lever/exp/argon-20261026/review-astra-served-split-fix/answer.md`,
  `t34-sol/astra-review-argon-w1b/REVIEW.md`, `astra-review-w1-merge/REVIEW.md`) without a scored
  timing-panel verdict.

### Historical sequence — 2026-10-17 B8 qualification & release (`T12` → `T12B` → `T12C`)

- **T12 HOLD (`SERVING_FAIL`, 2026-10-17):** checks 1–3 and 5–8 passed; check 4 failed its paced
  Watch12 high-speed retry (`CHECKPOINT_RELEASED_DEFAULT` remained `frp3-s12` during the hold; B8
  selected only via `SNAKE_SERVE_CHECKPOINT=b8-s3`). Evidence:
  [T12 B8 serving qualification](research/t12_b8_serving_qualification_2026-10-17.md).
- **T12B check-4 repair (`SERVING_PASS`, 2026-10-17):** authorized check-4-only rerun passed its
  pinned sibling amendment (`Watch6`/`Play6` server-clamped to `24 fps`, `Watch12`/`Play12` to
  `12 fps`, zero paced overruns; policy math/forward unchanged; checks 1–3 and 5–8 carried forward
  with original T12 failure preserved). Evidence:
  [T12B check-4 result](research/t12_b8_serving_qualification_2026-10-17.md#t12b-check-4-result-and-evidence).
- **T12C release (2026-10-17):** `b8-s3` promoted to `CHECKPOINT_RELEASED_DEFAULT`, superseding the
  T12/T12B hold; default and both rollback configs (`frp3-s12`, `champion`) verified live on `main`
  and `t12-20261017` and stopped cleanly ([release record](research/b8_s3_release_2026-10-17.md)).
  T15/T15B subsequently switched the default live Watch roster from `all_b8` to `mixed` on the same day.

### Historical results — October 5–8 FRP & Redesign M1–M3

Deltas below are H5000 mass integral unless stated; comparisons differ by row. Screening,
learning-slope, and distillation outcomes do not confer promotion.

| Date | Study | Recorded outcome | Result / comparison |
| --- | --- | --- | --- |
| 10-05 | FRP-v2 | `GO_L` | +41.3 (LB90 +26.3) vs champion+v8, 5/5 seeds; scripted survival −0.057 withheld `GO_H5000`. |
| 10-05–06 | FRP-v3 | `GO_R` → LH-1 `CLEAR` → `STRICT_PASS` → `SERVING_PASS` → released | Phase R +65.1 (HK LB90 +43.0) vs champion+v8, 5/5 seeds (served 2026-10-06 until B8 s3 on 2026-10-17). |
| 10-07 | FRP-v4 | `PARTIAL` | +6.8 [LB90 −15.6, UB90 +29.1] vs frp3-s12+v8, 3/5 seeds; no replication. B4 platform check failed; not evidence. |
| 10-07 | FRP-v5-S | `FIDELITY_NOT_PASSED` | Hazard 0.1533/1k > 0.15; no training or Phase R; replaced by fresh-world S2. |
| 10-08 | FRP-v5-H | `KILL_H` | H5−control −41.5 (UB −24.8), 1/8 seeds positive; H5−incumbent −26.6; control +14.3. |
| 10-08 | FRP-v5-S2 | `GO_S_RECIPE` | +21.5 (LB90 +8.4) vs incumbent; S−control +8.3 (LB −9.6). Gate infeasible: N_max 5551 > 600; no promotion (artifact suffix `20261009` is a namespace label; merge/result date is **2026-10-08**). |
| 10-08 | Redesign M1 | Done; G1 met | Grid/BatchSim 72/72 H5000; live/sim raster 45k frames (excluding unused scripted-snake hunger field); 21–28k hero steps/s/process with scripted opponents. |
| 10-08 | Redesign M2 | `FAIL` | No-veto student vs no-veto frp3-s12: −29.5 (LB −43.7; NI margin −15). |
| 10-08 | Redesign M2b | `PASS` | No-veto student vs no-veto frp3-s12: +52.4 (LB +34.8); both with v8: +11.1 (LB −16.4), descriptive. |
| 10-08 | Redesign M3-B | Learning-slope gate `PASS` | 5×20M RL; slope LB90 +0.46 > 0; ~5.9k transitions/s combined, 4.25× Mac; $3.92. |
| 10-08 | Redesign M3 Phase R | `NO_GO_EARLY` | Look 1, n=22/seed×mix; candidate+v8 vs frp3-s12+v8: +11.4 (SE 13.7), UB +40.5 < G=50; MI10 −18 (SE 37). Verify/audit PASS. |

**Historical provenance sources (Oct 5–8):** pre-release questions on M3-C drift/boost and RunPod incremental pulls/tie-explaining identity checks are historical only ([external policy sources / discrepancies](README.md)). External Claude memory notes (`standing-decisions-2026-10-08.md`, `redesign-scope.md`, `frp3-s12-release.md`, `frp-v2-phase2-go-l.md`, `frp-v3-go-r.md`, `frp-v4-partial.md`, `frp-v5h-kill.md`, `frp-v5s2-recipe.md`, `mac-pod-float-ties.md`, `runpod-5x-rule.md`, `runpod-cheap-pods-first.md`) and `campaign.json` keys (`standing_decisions_2026-10-08`, plus `final_outcome` fields added 2026-10-08 to `training_bet_frp_v4`, `training_bet_frp_v5h`, `training_bet_frp_v5s`, `training_bet_frp_v5s2`, `governance_strict_on_runpod`) are indexed in [README.md](README.md) and [research/INDEX.md](research/INDEX.md).
