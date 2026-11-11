# Current state — updated 2026-11-10 PT

Start: [project index](INDEX.md) · [registered items](REGISTERED_ITEMS.md) · [implementer checklist](IMPLEMENTER_CHECKLIST.md) · [campaign log](research/final_run_20261019/CAMPAIGN_LOG.md) · [research roadmap (advice)](research/ROADMAP.md) · [documentation map](README.md) · [study packages](../research/INDEX.md) · [research record](research/INDEX.md) · [lineage history](research/history/INDEX.md) · [glossary](research/GLOSSARY.md).
What we learned (plain-language findings map): [FINDINGS.md](FINDINGS.md).

- **Project goal:** Train the strongest possible slither.io-style agent. Mac serving bars (`24 fps`, `GameState.update p95 <= 11.688 ms`, Mac strict gate, serving qualification) gate **release** only and do not limit what is trained; a large teacher distilled to a Mac-size student is allowed.
- **What is served (unchanged until a release):** **Watch** = `132`-session confirmed `exact-stop-l1-os` replay bank (`watch-replay-bank-live-132-l1os-installed`, `SNAKE_WATCH_REPLAY_CONTROLLER=exact-stop-l1-os`, live bare `b8-s3` fallback); **Play** = live bare `b8-s3` (`Classic B8` default, `Field: Classic B8 / Mixed` opt-in). Rollbacks below remain active.
- **What is running (`2026-11-10 ~19:25 PT`):** `gpu_sitl` Step B1b smoke (attempt `26`) on `L4` pinned at `b772f89f`, then pilot; advisor round in progress for the new from-scratch GPU training track (*design pending, advisor round 2026-11-10*).
- **Owner ruling (`2026-11-10 ~19:20 PT`, [`CAMPAIGN_LOG.md`](research/final_run_20261019/CAMPAIGN_LOG.md) `2026-11-10 PT 19:25` — overrides conflicting older text below):**
  1. **Goal & release bars:** Train the strongest possible slither.io-style agent; Mac serving/strict/qualification bars gate **release** only (large teacher → Mac-size student distillation allowed).
  2. **`gpu_sitl` Step B1b = exploration lane:** Smoke and pilot (`research/gpu_sitl`) may use any GPU (`L4` included); wall-time and unit caps are runaway guards the operator may raise and log with no judge amendment; bars, held-out namespaces, and seeds are unchanged ([status note](research/advice/2026-11-09_live_search_reopen_registration_argon.status_2026-11-10.md)).
  3. **Kill rules close only the tested recipe:** Clauses such as `"permanently closes all learned-student work on ego2s-bv8 inputs"`, `"KILL_BRANCH_B ... 0 further Colab units and no sweeps"`, and whole-line `STOP_AND_MAINTAIN` triggers mean that recipe stops and a different recipe may be tried after an advisor design note; served agents stay as they are until a release.
  4. **Scarce GPU stock polling:** Scarce GPU stock is polled about once a minute (`~1/min`), not every `20` minutes.
  5. **New main track:** Large-scale from-scratch training on the exact GPU simulator, using all Colab slots, is opened (*design pending, advisor round 2026-11-10*).
  6. **Always kept:** Held-out/test data never used for training or tuning; honest reporting; verified stops and cleanup; `L4` shake-out before new GPU code runs on registered hardware; checkpoint/resume for runs `> ~1 h`; releases need confirmation-lane evidence and the owner's OK.

---

## Historical snapshot header — 2026-11-04 (T34 CLOSED); Watch bank updated 2026-11-03/11-06; live checkpoint released 2026-10-17

> **Historical snapshot header (superseded as live status by the `2026-11-10 PT` current-state block above; serving setup, monitoring thresholds, and rollback instructions below remain active).**

This is a dated documentary snapshot, not a live job or deployment check. The owner-ratified campaign
documents and standing rules take precedence over older launch notes and external memory; this
file is updated when they change.

---

## Current served setup (Watch & Play)

### Watch: 132-session `exact-stop-l1-os` replay bank (released 2026-11-06 PT)

- Bank `watch-replay-bank-live-132-l1os-installed` (controller `exact-stop-l1-os`: exact-stop-l1 + open-stop on unchecked ticks + length-scaled exact horizon). Evidence: Bank-v2 screen ADOPT, prospective confirmation CONFIRM (96 fresh worlds, H=14,400: +258.7 hero mass/world, 95% CI [+190, +327]; deaths −56%), Mac 24 fps qualification runs 3+4 QUALIFIED. Rollback: `watch-replay-bank-live-132-l1` + `exact-stop-l1` (previous release, below).
- **How it runs (since 2026-11-07 PT):** persistent tmux session `snake-web` running `tools/ops/watch_server.sh`, which reads the `snake-web` entry of `.claude/launch.json` on every start and restarts the server if it exits (the Claude app had been stopping its preview server when sessions ended). Logs: `../snake-dqn-artifacts/ops/watch-server/server.log` (rotated, 3 × 100 MB). To apply a rollback: edit `launch.json`, then `tmux kill-session -t snake-web` and `tmux new-session -d -s snake-web -c <repo> "bash tools/ops/watch_server.sh"`. Do not also `preview_start snake-web` (port 8000 conflict).

### Watch: 132-session confirmed `exact-stop-l1` replay bank (expanded 2026-11-03 ~11:30 PT; first released with 36)

- **Active (11:30 PT):** `watch-replay-bank-live-132-l1` (36 A100 + 96 G4 sessions, banks 14l1–24l1), pure bank direct, controller `exact-stop-l1`; Mac qual runs 2+3 QUALIFIED consecutively (run 1 one 53.9 ms overrun). Rollback: `.../watch-replay-bank-live-36-l1` (same controller) or the l2-prime lines below. Records: campaign log 2026-11-03 11:30 row.

- **Active (2026-11-03):** `snake-dqn-artifacts/watch-replay-bank-live-36-l1` — **36 confirmed `exact-stop-l1`
  sessions** (banks `14l1–16l1`, `source_commit 73974411`, `H = 6,000`, hero `b8-s3` `3e649783…`), served
  **pure bank direct** with `SNAKE_WATCH_SOURCE=replay`, `SNAKE_WATCH_REPLAY_DIR=.../watch-replay-bank-live-36-l1`,
  `SNAKE_WATCH_REPLAY_CONTROLLER=exact-stop-l1`. Basis: Watch6-L1-V1 `GO_L1_SUPERIOR` (sealed, ledger `2eff534c…`).
  Install receipt `t34-sol/argon-l1-serve/install-bank-live-36-l1.json` (`96e54c93…`); Mac 24 fps qualification
  (`qualify.py` at `10058e29`, cherry-picked to main): run 1 `ab94e748…` NOT_QUALIFIED (1/120 overrun on the
  filtered-over-168 path only; pure path all PASS), run 2 `b9cafcf4…` QUALIFIED (all bars, both paths). Ruling
  `t34-sol/argon-l1-serve/RELEASE_RULING_L1_WATCH.md` (Fable `APPROVE-WITH-CONDITIONS`, `2ee9075f…`). Variety:
  36 × 6,000 frames ≈ 2.5 h before the shuffle repeats; expansion to ≥ 96–132 sessions via G4 banks is in progress
  and needs a new install receipt + fresh qualification before repointing. **Rollback:** restore
  `SNAKE_WATCH_REPLAY_DIR=.../watch-replay-bank-live-132` + `SNAKE_WATCH_REPLAY_CONTROLLER=exact-stop-l2-prime`
  and restart; emergency `SNAKE_WATCH_SOURCE=live`. Keep live-132 on disk ≥ 7 days; check serving log daily for 3 days.
  The filtered-over-168 path is NOT qualified for serving.

### Previous: 132-session confirmed `exact-stop-l2-prime` replay bank (released 2026-11-01 ~16:00 PT; rollback target)

Watch plays **confirmed Colab-precomputed replays** (`exact-stop-l2-prime` search hero at `L >= 600` over
product Watch6 v2, `5× frp3-s12 + v8` opponents, 6,000-frame sessions; pipeline in
[`research/watch_replay/`](../research/watch_replay/)) visibly labelled **"Recorded assisted hero"**:

> [!NOTE]
> **Historical snapshot (2026-11-01 PT) — superseded by [`watch-replay-bank-live-132-l1`](#watch-132-session-confirmed-exact-stop-l1-replay-bank-expanded-2026-11-03-1130-pt-first-released-with-36) and [T34 CLOSED (2026-11-04 PT) — final state](#t34-closed-2026-11-04-pt--final-state).** Preserved for rollback and qualification provenance; `Watch6-L1-V1` subsequently completed (`GO_L1_SUPERIOR`) and Watch now serves the 132-session `exact-stop-l1` replay bank `watch-replay-bank-live-132-l1` (`SNAKE_WATCH_REPLAY_CONTROLLER=exact-stop-l1`).

- **Active bank & controller filter (`f5912f44`):** `snake-dqn-artifacts/watch-replay-bank-live-132`
  (**132 confirmed `exact-stop-l2-prime` sessions** from banks `01–08` and `10–12`), served with
  `SNAKE_WATCH_SOURCE=replay` and `SNAKE_WATCH_REPLAY_CONTROLLER=exact-stop-l2-prime` in
  `.claude/launch.json` (`snake-web`, verified live on `:8000`). The `168`-session mixed bank
  `watch-replay-bank-live-168` (`132` `exact-stop-l2-prime` + `36` confirmed `exact-stop-l1` sessions
  from banks `14l1–16l1`; confirmed by Watch6-L1-V1 `GO_L1_SUPERIOR`, 2026-11-03) remains on disk
  (ratified `Watch6-L1-V1`, [`research/watch6_l1_confirm/REGISTRATION_L1_V1.md`](../research/watch6_l1_confirm/REGISTRATION_L1_V1.md) + `t34-sol/argon-l1v1-pkg/RULING_L1_V1.md`; see [research/ROADMAP.md](research/ROADMAP.md) §2.3).
- **Serving guards, cooperative prefetch & Mac `24 fps` qualification (`dc834ded`–`fec81656`):**
  - `GameSession` ([`web/backend/session.py`](../web/backend/session.py)) gates Watch replay on
    `_is_clean_mixed_watch6_b8()` across `__init__`, `_build`, `reset_game`, and `set_mode` (so env-var
    `SNAKE_SERVE_CHECKPOINT` or `SNAKE_SERVE_WATCH_ROSTER` rollbacks, `Watch8/12`, or explicit checkpoints
    automatically run live); refuses `MODE_TRAIN` on forward-only `ego2s-*` / `raster31v2` / `raster31v3`
    and un-overridden reward-contract mismatches *before* calling `_leave_watch_replay()` (recovering
    replay on fallback to Watch); and scopes `_replay_opted_out = True` to `set_epsilon`, live `set_hero`
    change, and `set_food` (cleared by `_build` and `new_game()`, wired to `"new_game"` in `app.py:311`,
    leaving `Play -> Watch` intact).
  - `ReplayBank` ([`web/backend/watch_replay.py`](../web/backend/watch_replay.py)) filters recordings on
    line 1 via `peek_header` + `SNAKE_WATCH_REPLAY_CONTROLLER` (`all | exact-stop-l2-prime | exact-stop-l1`,
    failing closed to live B8 on invalid values), shuffles cycle-1 discovery (`shuffle_discovery=True`),
    uses shallow `Recording.snapshot`, and yields the GIL every `0.25 ms` / `64` frames for
    `max(1.0 ms, 4.0 * slice_elapsed)` (`<= 20%` GIL duty cycle) during `verify_payload`.
  - [`research/watch_replay/qualify.py`](../research/watch_replay/qualify.py)
    (`t34-sol/argon-replay-qual/{SUMMARY.md,qualification.json}`, Fable review `APPROVE_WITH_CONDITIONS`
    with `C1–C3` resolved at `fec81656`) passed `132/132` inventory parity, cold readiness `0.349–0.435 s`
    (`< 0.45 s`), concurrent cold-verification live-B8 fallback `update p95 = 8.78 ms <= 11.688 ms`,
    `0/120` paced-`24 fps` overruns (`step p99 = 11.34 ms <= 41.67 ms`), and seamless recorded playback
    cutover after `~18.5–23.1 s` (`recorded update p95 = 0.002 ms`). Recorded `.watch.gz` v2 banks stay
    at `H = 6,000` (`< 512 MiB` `MAX_DECODED`), while `record=None` confirmation runs at `H = 14,400`.
- **Fallback & rollback:** **Live bare `b8-s3` remains the automatic fallback** (during cold startup
  prefetch, or on an empty, invalid, or single-session exhausted bank), and **Play always stays live**
  (`b8-s3` without a veto). Explicit rollback to live mixed-Watch6 `b8-s3`: unset `SNAKE_WATCH_SOURCE`
  (or set `SNAKE_WATCH_SOURCE=live`) and restart; rollback to the previous mixed bank: point
  `SNAKE_WATCH_REPLAY_DIR` at `watch-replay-bank-live-168` without `SNAKE_WATCH_REPLAY_CONTROLLER`.
- **Product efficacy confirmation status:**
  - **`exact-stop-l2-prime` (`L >= 600`) — CONFIRMED (`GO_EFFICACY_ONLY`, 2026-11-01 14:40 PT, `52135312`):**
    Watch6 V3 ([`research/watch6_search_efficacy_v3/`](../research/watch6_search_efficacy_v3/)) passed
    both co-primary bars over all `96/96` primary `H = 14,400` Watch6-v2 pairs (`0` reserves,
    `2,764,800/2,764,800` mirror frames): **`Δdeaths/1k = -0.125`** (`CI95 [-0.153, -0.097] < 0`) and
    **Fieller pooled mass `+63.7%`** (`CI95 [+49.8%, +78.5%] >= -5%`) $\Rightarrow$
    `ELIGIBLE_FOR_REPLAY_QUALIFICATION` (followed by Mac qualification `fec81656` and release `f5912f44`).
    The earlier Watch6 V2 run ([`research/watch6_search_efficacy/`](../research/watch6_search_efficacy/))
    closed `INVALID_OR_INCOMPLETE` on 2026-10-31 with `0` outcomes read after shard 6 exhausted pre-start
    retries (`t34-sol/watch6-shard6-ruling/RULING_SHARD6.md`).
  - **`exact-stop-l1` (`L >= 1`) — exploration `ADOPT` complete; `Watch6-L1-V1` ratified on R2 (`93c77f43`), `l1v1-packet-02` canaries `PASS` (`0.70 u`), `18` `H = 14,400` shards RUNNING (`l1v1-main-01`, `a3135483`, Wave 1 at `t ≈ 1.85–2.0 h` in 3rd/final arm on `6/6` Colab slots at `26.01 u/h`):**
    passed the `N = 48, H = 6,000` exploration screen (`+98.3` mass, `-67.4%` `deaths/1k` vs `L >= 600`),
    with `36` sessions (`14l1–16l1`) staged on disk. Prospective `H = 14,400` 3-arm confirmation
    ([`research/watch6_l1_confirm/REGISTRATION_L1_V1.md`](../research/watch6_l1_confirm/REGISTRATION_L1_V1.md),
    initial `cf07791b` / `ac250f05` ruling `t34-sol/argon-l1v1-pkg/RULING_L1_V1.md` and `43,200/43,200` `1.272×` Mac mirror `t34-sol/argon-l1v1-pkg-fix1/mac-mirror.json` with `1.892 ms/step` / `612.5 ms/root` `l1` search and shared `L = 1,694` death) had its `l1v1-packet-01` canaries fail on `job.sh:21` `COLAB_EXPORT_DIR: unbound variable` (`17:29 PT`, `0` science exports). Commit `1873b902` (merged on `main` at `93c77f43` / `ca5b93d1`; `t34-sol/argon-l1v1-pkg-fix2/SUMMARY.md`) replaced `COLAB_EXPORT_DIR` with `COLAB_OUTPUT_DIR` (`+ --fixture-horizon 6`) in `job.sh`, re-ran the `14,400×3` Mac mirror (`t34-sol/argon-l1v1-pkg-fix2/mac-mirror.json`, SHA-256 `e89cc8c2…`, `43,200/43,200` frames, `0` mismatches, `1.573×` 3-arm wall ratio — `1.272×–1.573×` across the two mirror runs per Fable R2 Condition 5 — under new science identity **`57ec544fe45f7a8a876dbc9363048468b63358e224b2ee8e11dff41401dcfbd3`**), received **Fable R2 `APPROVE-WITH-CONDITIONS`** (`t34-sol/argon-l1v1-pkg/RULING_L1_V1_R2.md`, amending `REGISTRATION_L1_V1.md` §5.2 canary fixture horizon `H=60 -> H=6` and retiring `cf07791b` to `superseded-cf07791b/`), passed rebuilt `t34-sol/l1v1-packet-02` canaries (`A100 365.4 s / 0.574 u` + `L4 421.0 s / 0.126 u` = **`0.70 u` observed**, `stop_verified = true`, `7/7` exports, `18.4 KiB/world` = `~112 KiB < 2.8%` of `4 MiB` for `6` worlds), and dispatched all `18` primary shards (`t34-sol/l1v1-main-01`, `12×6 A100 W = 6` + `6×4 L4 W = 4`, **`~636.8 u` central / `~637.5 u` incl. `0.70 u` canaries**, `598.1–739.5 u` range, `~27.7 h`; launched `19:10 PT` at `a3135483` with `retry_limit = 1` on `6/6` Colab slots: `3 A100` shards `00, 02, 04` at `6,720–6,840 s` (`poll-000178.json`) + `3 L4` shards `12, 13, 14` at `6,360–6,480 s` (`poll-000163.json`) at `t ≈ 1.85–2.0 h`, having crossed their 2nd intra-world `H = 14,400` arm boundary into their 3rd/final arm at `26.01 u/h`, `6.0` assignments, balance `2,106.84 u`), awaiting the `0 u` Mac `M0b` / `M0_process_v27` live-ops patch before L4 Wave 1 exits at **`~01:55 AM PT`** (`~12.6 min` before A100 Wave 1: enforcing `ACCOUNT_GPU_SUBCAPS = {"L4": 3, "A100": 3, "H100": 3}` in `ready_queue.py:682` so FIFO scanning does not starve L4 Wave 2 `shards 15..17` behind A100 `412` attempts; replacing `_prealloc_tree_clean()` with a pre-compute forbid-list admitting `staging/*`, `accounting-baseline.json`, and `public-input-preflight.json`; ignoring `{"workload-started.json", "performance.json"}` exported at `t = 0 s` in `auto_retry.py:265-270` and `admission.py:503-519` so `0`-`world-*.json` drops stay Tier-0 (`completion_attempt += 0`, `prestart_attempt <= 6`, bypassing `if attempt >= maximum` on `retry_limit = 1`); replacing `admission.py:511-706` with a negative-fault transport classifier admitting `RPC_EXEC_POLL_TIMEOUT` with `traceback=None`, Dead Zones A/B, and all 5 polling `AccountingError` strings; pinning `1873b902:web/backend/session.py` (`da7b4971…`) inside operational validators without touching `protocol.py`; filtering canaries before `final_look.py:603`; and unifying `RETRY.json`/`unalloc_ref`/`prior_receipts` into one linear DAG under `pre_outcome_open_guard()`), completion of all `96` worlds, accounting (`final_look.py`), Mac qualification, and owner OK ([research/ROADMAP.md](research/ROADMAP.md) §1.3 & §2.1–§2.3).

### Play & live Watch fallback: released bare `b8-s3` checkpoint (2026-10-17)

- **Play Field option (2026-11-04 PT 17:00, main `b3789463`):** opt-in "Field: Classic B8 / Mixed" in
  the Play panel. **Classic B8 stays the default.** Mixed = frp3-s12+v8 opponents (probe CANDIDATE,
  `t34-sol/argon-play-mixed-probe/REPORT.md`). Leaderboard is partitioned by field. Force Classic with
  `SNAKE_SERVE_PLAY_ROSTER=all_b8` (or `=mixed` to make Mixed the default for all sessions). Judge ruling
  2026-11-05 (`KEEP_OPT_IN`), revised prospectively 2026-11-09
  ([`CAMPAIGN_LOG.md`](research/final_run_20261019/CAMPAIGN_LOG.md) `2026-11-09 PT 11:21` row, item 4):
  keep `KEEP_OPT_IN`; revisit at `>= 30` completed runs per field, with uptake reported rather than gating.

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
  by roster mode and size (initial owner check-in window: 2026-10-24). In Watch replay mode
  ([`CAMPAIGN_LOG.md`](research/final_run_20261019/CAMPAIGN_LOG.md) `2026-11-09 PT 11:21` row, item 2),
  the observation window is `7` days and `>= 50,000` replay frames served; roll back on any fallback,
  replay step failure, `ego2s` exception, replay update `p95 > 41.67 ms`, or bank exhaustion, with live
  death/latency bars evaluated only on live-fallback frames.

---

## T34 CLOSED (2026-11-04 PT) — final state

T34 is **closed** by its pre-registered closeout audit (campaign log 2026-11-04 17:40 row; readout
`t34-sol/argon-closeout-audit/REPORT.md`). Final shipped product: **Watch** serves 132 confirmed
`exact-stop-l1` replays (`watch-replay-bank-live-132-l1`; Watch6-L1-V1 `GO_L1_SUPERIOR`: deaths −72%, mass
+95% vs bare at H=14,400); **Play** and the live Watch fallback serve bare `b8-s3`. Serving fixes: gen-2 GC
tick stall (`5b4353c6`), Watch↔Play RSS growth (`7ecec2a0`).
Closing evidence (all Mac-only, 0 Colab units):
- **open-stop-h32-nb** (alive + un-enclosed H=32 certificate) PASSED a pre-registered 96-fresh-world gate
  (deaths −70%, mass +51%, harmful overrides 0.88%) — valid offline controller, ~60 ms/root, not 24 fps-live.
- **Search→bare distillation CLOSED** (Stage 1B: frozen-feature heads keep only 8–9% of open-stop rescues at
  ≤1% ordinary flips; same selectivity wall as Sandbox B).
- **Live search CLOSED** (closeout Part B: neither a one-tick H=12 budget nor a 3-tick-delayed async
  open-stop keeps ≥50% of rescues within ≤1% false overrides).
- **Claim A b8-head closed-loop**: all 4 arms lose mass (−305 … −26, LB90 < 0) → KILL (contract schema
  partly missing in the receipt; reported INCONCLUSIVE-on-schema, FAIL on every measured bar).
- Earlier stops this round: hero M4-B2 Stage 1A FAIL; v6 fast kernel parked (full step ≈1.65 ms; the v6 M0
  `0.22 ms` bar is retired prospectively on false premise and the `24 fps` / `11.688 ms` budget governs per
  [`CAMPAIGN_LOG.md`](research/final_run_20261019/CAMPAIGN_LOG.md) `2026-11-09 PT 11:21` row, item 6).
Prospective stop and reopening rules ([`CAMPAIGN_LOG.md`](research/final_run_20261019/CAMPAIGN_LOG.md)
`2026-11-09 PT 11:21` row; past verdicts, one-look results, and held-out data stay as they are):
- **(1) Offline Watch controller ideas:** the Bank-v2 census rule (`STOP_AND_MAINTAIN` if `deaths/1k <= 0.03`)
  was a one-shot rule for the `l1` bank, not a standing ceiling on `l1-os`. An offline Watch controller idea
  gets one Mac-only 48-paired-world `H = 6,000` dev screen; Colab confirmation runs only if paired
  `Δmass >= +15` with `LB95 > 0` and `deaths/1k <= incumbent`; `deaths/1k` alone never closes the line.
- **(3) Long Watch sessions:** replacing the unsatisfiable `N = 200` viewer gate for opt-in, long sessions
  become an exploration-lane opt-in Watch setting after a `0 u` Mac engineering gate and a 12-session
  `H = 14,400` pilot; changing the default still needs a viewer test.
- **(4) Play Mixed trigger:** keep `KEEP_OPT_IN`; revisit at `>= 30` completed runs per field, with uptake
  reported rather than gating.
- **(5) Hero-policy reopening (amended 2026-11-10):** under the `2026-11-10` owner ruling above, Mac `24 fps` /
  `11.688 ms` bars gate **release** only (not what is trained; large teacher → Mac-size student distillation is
  allowed), kill rules close only the tested recipe, and a different recipe may be tried after an advisor design note.
- **(6) v6 M0 latency bar retired:** the v6 M0 `0.22 ms` step-time bar is retired (false premise); at release, the
  `24 fps` / `11.688 ms` budget governs.

## Active campaign state & Oct 31 / Nov 1 results (T34 ExIt-B8 — 2026-11-01 historical snapshot)

> [!NOTE]
> **Historical snapshot (2026-11-01 PT) — superseded by [T34 CLOSED (2026-11-04 PT) — final state](#t34-closed-2026-11-04-pt--final-state) above.** T34 is closed; no Colab runtimes are running and Watch serves the 132-session confirmed `exact-stop-l1` bank (`watch-replay-bank-live-132-l1`). The Oct 31 / Nov 1 items below are preserved for campaign provenance.

For synthesized current-round and later-round research plans (**advice, not decisions**), see
[research/ROADMAP.md](research/ROADMAP.md).

- **RELEASED 2026-11-01 ~16:00 PT — Watch serves only the 132 confirmed `exact-stop-l2-prime` replays**
  (merge `fec81656`: controller filter + serving guards + Mac qualification `research/watch_replay/qualify.py`,
  all bars PASS — `t34-sol/argon-replay-qual/{SUMMARY.md,qualification.json}`; Fable review
  `APPROVE_WITH_CONDITIONS`, conditions C2/C3 fixed). `.claude/launch.json` `snake-web` now sets
  `SNAKE_WATCH_REPLAY_DIR=.../watch-replay-bank-live-132` **and** `SNAKE_WATCH_REPLAY_CONTROLLER=exact-stop-l2-prime`;
  verified on port 8000 (replay `controller == exact-stop-l2-prime`, error null). The `36` unconfirmed `exact-stop-l1`
  sessions are no longer served. Rollback: previous config (`watch-replay-bank-live-168`, no controller var), or
  `SNAKE_WATCH_SOURCE=live` for bare `b8-s3`. Post-release: watch `update_ms`/overrun telemetry (C4).
- **Confirmation DONE — Watch6 V3 exact-stop efficacy: `GO_EFFICACY_ONLY` (2026-11-01 14:40 PT)**
  ([`research/watch6_search_efficacy_v3/`](../research/watch6_search_efficacy_v3/)): sole registered final
  look via the V6.1 accounting adapter (`75471246`) over all `96` fresh paired `H = 14,400` Watch6-v2 sessions
  (`b8-s3` bare vs synchronous `exact-stop-l2-prime` at `L >= 600`, `H = 300`, `5× frp3-s12 + v8`,
  frame-exact `GridBatchSim` mirror; `0` reserve substitutions). **Both co-primary bars pass:**
  deaths/1k frames `−0.125` (paired-t CI95 `−0.153 .. −0.097`, bar `CI95_upper < 0`); pooled mass
  `+63.7%` vs bare (Fieller CI95 `+49.8% .. +78.5%`, bar `lower >= −5%`). Decision:
  `ELIGIBLE_FOR_REPLAY_QUALIFICATION` (efficacy only — not a release; governs `exact-stop-l2-prime`,
  not the `36` `exact-stop-l1` recordings in `watch-replay-bank-live-168`). Outputs:
  `snake-dqn-artifacts/t34-sol/argon-v3-accounting-v6-fix1/final-look-v3/` (`efficacy-decision.json`,
  `final-verified-ledger.json`, each with `.sha256`). Path: 24 shards, 4 V6.1-accepted mechanical
  failures (`00, 09, 12, 19`) each passed on its single retry; shards `00/12` needed the `09:16 PT`
  Fable-ratified quarantine of `23` historical non-V3 H100-503 manifests
  (`t34-sol/v61-fence-ruling-20261101.md` → `ops/manifests-archive-20261101/`, names retired).
- **Completed exploration — Watch6 all-length screen (`ADOPT` for new replay banks)**
  ([`research/watch6_alllen_screen/`](../research/watch6_alllen_screen/), readout
  `t34-sol/argon-alllen-a100/analysis.json`, `N = 48` paired 6,000-frame Watch6-v2 sessions):
  - `bare` (`b8-s3`): mass **`351.1`** (`CI95 [320.2, 381.9]`), deaths **`0.288/1k`** (`83` deaths: `53` at `L < 600`, `30` at `L >= 600`, `21` at `L >= 800`).
  - `l2_prime_600` (served `L >= 600`): mass **`461.2`** (`+110.1` vs `bare`, `CI95 [+72.1, +148.1]`), deaths **`0.160/1k`** (`-44.6%` vs `bare`; `46` deaths: `34` at `L < 600`, `12` at `L >= 600`, `11` at `L >= 800`), `449.6` mean branches/session.
  - **`exact_stop_alllen` (`L >= 1`, `interval = 5`, `H = 300`):** mass **`559.4`** (**`+208.4` vs `bare`**, `CI95 [+170.1, +246.7]`; **`+98.3` vs `L >= 600`**, `CI95 [+50.6, +146.0]`, with `27` mass wins, `7` ties, and `14/48` `29.2%` mass losses vs `L >= 600`), deaths **`0.052/1k`** (**`-81.9%` vs `bare`**; **`-67.4%` / `-0.108/1k` vs `L >= 600`**, `CI95 [-0.161, -0.054]`; `15` total deaths: **`1` at `L < 600`**, **`14` at `L >= 600`**, **`13` at `L >= 800`**, and `3/48` `6.25%` sessions with more deaths than `L >= 600` at `index 36` `seed 3075941873`, `index 39` `seed 3394546952`, and `index 47` `seed 121131908` due to `DenseTrigger.fires()`'s modulo-5 phase shift in `trigger.py:35-42`, unshielded `frame % 5 != 0` inter-check ticks where `ego_raster.py:511` discards `V8Result`, and rollout horizon `H = 300 < L`), **`2.82×` search branches** (`1,268.5` mean branches/session, `0.142%` `L < 600` overrides, `mean_max_length = 1,096.0`).
  - Pre-stated adoption rule (`>= +15.0` incremental mass and `<= 0.0` `deaths/1k` vs `L >= 600`) met: commit `73974411` added `"exact-stop-l1"` to `research/watch_replay/` and `web/backend/watch_replay.py`, and A100 `W = 6` banks `14l1–16l1` (`3 × 12` = `36/36` verified `H = 6,000` sessions on disjoint seeds in **`3.03–3.07 h/bank`, `20.36–20.67 A100 u/bank` = `~61.5 A100 u` observed total** at `6.73 u/h` vs the `2.821×`-calibrated `75.6 u` base / `151.2 u` 2× forecast in `t34-sol/argon-banks-l1/SUMMARY.md:38-40`; at `H = 14,400`, `DenseTrigger(min_length=1, interval=5)` is physically capped at `<= 2,880` roots / `~3,045–3,250` branches = **`1.19–1.28×`** V3's `2,549.4` at `H = 14,400` and `~2.4–2.6×` `l1`'s `1,268.5` at `H = 6,000`) were verified and merged into `watch-replay-bank-live-168` (`395eacbb`, `web/serve.py` restarted on `b5b63cd1`). Exploration screen only — confirming `exact-stop-l1` as a Claim-B release requires prospective H14400 confirmation via [`research/watch6_l1_confirm/`](../research/watch6_l1_confirm/) ([`REGISTRATION_L1_V1.md`](../research/watch6_l1_confirm/REGISTRATION_L1_V1.md), initial `cf07791b` / `ac250f05` Fable ruling `t34-sol/argon-l1v1-pkg/RULING_L1_V1.md` and `43,200/43,200` Mac mirrors `t34-sol/argon-l1v1-pkg-fix1/mac-mirror.json` (`1.272×`) + R2 `t34-sol/argon-l1v1-pkg-fix2/mac-mirror.json` (`1.573×`, `e89cc8c2…`) in place of `M0b`, Fable R2 `APPROVE-WITH-CONDITIONS` `t34-sol/argon-l1v1-pkg/RULING_L1_V1_R2.md` on `1873b902` / `93c77f43` (`COLAB_OUTPUT_DIR` + `--fixture-horizon 6`, science identity `57ec544fe45f7a8a876dbc9363048468b63358e224b2ee8e11dff41401dcfbd3`), `t34-sol/l1v1-packet-02` canaries `PASS` (`0.574 A100 + 0.126 L4 = 0.70 u` observed), and `18` single-wave shards = `12×6 A100 W = 6` + `6×4 L4 W = 4` running in `t34-sol/l1v1-main-01` (`a3135483`, **`~636.8 u` central / `~637.5 u` incl. canaries**, `598.1–739.5 u` range, `~27.7 h`, Wave-1 calibrated at `26.01 u/h` / `2,106.84 u` balance), alongside the `0 u` Mac `M0b` live-ops patch before `~01:55 AM PT` / `~08:38 AM PT` in [research/ROADMAP.md](research/ROADMAP.md) §1.3 & §2.1–§2.3).
- **Completed parity — GPU `DeviceSim` v2 parity `PASS` on A100**
  ([`research/devicesim_v2_parity/`](../research/devicesim_v2_parity/),
  `t34-sol/argon-devicesim-v2/packet-a100-v2-parity-01/main-run-01/receipt.json`): **`0` mismatches**
  across **`1,536,000` snake-ticks** (`E = 256 × 1,000` Watch6-v2 ticks in `3,505.6 s` = `1,434.6`
  steps/s under 5-opponent CUDA flood launches, SplitMix64 RNG, commit `b500d111`, subject to `<= 1e-4`
  CUDA FP32 near-tie Q re-decision checks), following `553 passed` local tests and Mac
  `E = 4 × 8` preflight `PASS`.
- **Completed Oct 30 Claim-A & ruler receipts awaiting Mac readout (and `spatial-student-pilot` fit canceled):**
  - `b8-head-closedloop` (`origin/astra-doom-auc-smoke`, `N = 192` v1 H5000 worlds, `m in {0, 10, 25, 50}`, run without `boost_decisions` logging): **sole remaining Claim-A gameplay screen**; A100 receipt **`PASS`** (`t34-sol/b8-head-closedloop/main-01/output/receipt.json` in `2,666.1 s`, `4.51 A100 u` attributable / `4.98 u` elapsed, `batch-0000..0002.json` verified; `SUMMARY.md:50-52`), awaiting `< 90 s` Mac `closedloop_readout.py` (`M0a`) against binding v1 `H = 5,000` `>= +40` pooled-mass / survival / `<= 15%` override bars (plus a `0 u`, `10 min` v2 boost `[0.5%, 10%]` check if any arm survives).
  - `spatial-qbytes-01` (`origin/astra-spatial-student-pilot`, `t34-sol/spatial-qbytes-fix/SUMMARY.md:85-88`): **GPU Q-byte / `.so` cache throughput & parity ruler only** (not a Claim-A gameplay screen); A100 receipt **`PASS`** (`t34-sol/spatial-qbytes-fix/t34-spatial-qbytes-01/ruler-output-a100-fallback/receipt.json`), awaiting Mac parity/timing audit while the `40–48 A100 u` `spatial-student-pilot` GPU fit is canceled (`0 u`, retaining compiled `.so` caches) per `t34-sol/student-direction/{astra,fable}.md` and T4 Gate-1 `KILL_T4`.
- **Colab concurrency caps:** **`6` concurrent runtimes total** (`ACCOUNT_ASSIGNMENT_CAP = 6`, tree `36285bc6` active in `ops/colab-cli-qualified.json` and currently running at **`6/6` concurrent runtimes on `Watch6-L1-V1` `l1v1-main-01`** — `3 A100` shards `00, 02, 04` @ `6.73 u/h` (`6,720–6,840 s` elapsed at `poll-000178.json`, in 3rd/final arm of World 0) + `3 L4` shards `12, 13, 14` @ `1.94 u/h` (`6,360–6,480 s` elapsed at `poll-000163.json`, in 3rd/final arm of World 0) = `26.01 u/h` calibrated across `6.0` assignments, balance `~2,153 u` pre-run / `2,106.84 u` after Wave-1 calibration / `~1,515 u` post-run), at most **`3` concurrent L4** (`ACCOUNT_L4_CAP = 3` in `tools/ops/ready_queue.py`; enforcing **`ACCOUNT_GPU_SUBCAPS = {"L4": 3, "A100": 3, "H100": 3}`** at `ready_queue.py:682` before L4 Wave 1 exits at `~01:55 AM PT` so FIFO scanning does not starve L4 Wave 2 `shards 15..17`, replacing `admission.py:474-706` with a pre-compute forbid-list + negative-fault transport classifier while ignoring `{"workload-started.json", "performance.json"}` in `auto_retry.py:265-270` and `admission.py:503-519` so `0`-`world-*.json` drops stay Tier-0 (`completion_attempt += 0`, `prestart_attempt <= 6`, bypassing `if attempt >= maximum` on `retry_limit = 1`), pinning `1873b902:web/backend/session.py` (`da7b4971…`) inside operational validators, unifying `RETRY.json`/`unalloc_ref`/`prior_receipts` into one linear DAG in `final_look.py`, and auto-reconciling proven HTTP 412/503 non-allocations in `tools/ops/dispatch.py:2935` is tracked in [research/ROADMAP.md](research/ROADMAP.md) §1.3 & §2.2).
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
    `0.316 ms` Mac 1-thread p95, `33.0%` vs `9.0%` `L >= 800` `all_die_sel0` rescue on `25/129` `19.4%`
    `L >= 800` death roots — `9/25` `36.0%` `macro_triggered` at `+66.7 pp` rescue when triggered vs `16/25`
    `64.0%` re-trapped after tick-`30` greedy-B8 handback, while `104/129` `80.6%` survived `H = 300` inside
    uncertified coils and `options.py` hardcoded `min_boost_length = 10` vs live `5`, restricted `omega1` to
    static `tau <= 1` with `2 * min(black, white)` acting only as an upper bound on non-Hamiltonian 2-cores,
    and vacuously set `t_cert = 1` on `ext_gain == 0` internal tail cells —
    failing `+3.85 pp < +10.0 pp` total `L >= 600` lift and `2 > 0` newly caused ordinary lethal flips from
    single-suffix `sel0` and aggressive `O3`/`O5`; `0` assessment roots opened; informs **`CTS-ZWRHS-L1-v6` (`RH-DAG-1`)**
    (`CoalescedKernelOpponents(precomputed_counts=counts_sel)` eliminating triple opponent BFS `45 -> 15` floods/step + exact post-move `_slot_pad` $O(1)$ TTL (`v8_nb.py:753-843`) + **Subsumed-BFS + Component-Ceiling Theorem** (`v8_nb.py:741-850,1051`, cutting hero BFS work `2.5×`) + read-only stack-local `tail_flat[6]` in `_region_kernel` (`ego_raster_nb.py:574,628`, avoiding `sim._owner_pad` parallel mutation races) + pre-allocated `V8ResultBuffer` + `_i64` zero-copy fast path + `step_prepared(hero_policies=[])` cutting `mac-mirror.json`'s `1.892 ms/step` **`12.6×` to `~0.15 ms/step`**,
    **Receding-Horizon Suffix-Grafting MPC (`RH-DAG-1`, $h = 96 \to 64 \to 1$) + `RolloutBacktrack-16`** branching `32` steps from cached $\hat{w}_{t+96}$ (`~5.2 ms`) or `64` steps from $\hat{w}_{t+64}$ (`~10.4 ms`) with 16-step interior backtrack (`interval = 1` closed-loop $k$-deviation MPC),
    **Future-Rooted 2-Tick Stage B from $\hat{w}_{t+1}$** (`64 + 64` steps, `0` counterfactual-orphan branches) keeping single-process Mac live serving **`p99 <= 12.0 ms < 33.0 ms`** (`0` worker processes),
    and Every-Tick $\pi_{\text{shield}}$ (`v8_tail_redecide` + `res_opp.head_risky[j, res_opp.action[j]] == True` at `v8_nb.py:1330-1350,1626` catching both `OUTCOME_NO_SPACIOUS` and `ne == 0` opponent lunges) + `OppPinchProbe` + $h = 128$ `TailTrackWitness` leaf cert (`min_boost_length = 5`, `ext_gain >= L_128`),
    gated by `0 u` Mac `M0_rhdag1_v6` (`~5 min` on `mac-mirror.json` + `32` cached roots) before 3-arm `Watch6-RHDAG1-V6-Screen96` (`N = 96, H = 14,400`, `{exact-stop-l1, v5, v6}`, **`~7.4 u` / `~14.8 u` 2× cap**, `~17 min`) in [research/ROADMAP.md](research/ROADMAP.md) §3).

---

## Fast-process rule (owner, 2026-10-30, standing) — overrides older ceremony rules

Stronger models now; cut ceremony that does not pay for itself.

1. **Two lanes.** *Exploration* (default: screens, smokes, pilots, speed work): no pre-registration documents or amendments; at most one quick review, and only if the run costs > ~10 units or touches shared code; results are not release evidence but may be acted on immediately. *Confirmation* (only runs that decide a release or the strict gate): full rigor — pass bars fixed before data, held-out/fresh test worlds, paired statistics, verified stop.
2. **One reviewer per change.** No repeated delta reviews of small mechanical fixes — if tests pass, ship. Carbon (`agy`) implements; Claude Opus 5.5 (run as a separate fresh `agy` reviewer), a separate fresh `agy` run, or Astra reviews (`CAMPAIGN_LOG.md` `2026-11-10 11:35`; in exploration, skip review if unavailable). Never wait.
3. **Fewer canaries.** Unchanged transport tree + unchanged job package (only parameters/seeds differ) → dispatch the main run directly. New GPU code runs an L4 shake-out before registered hardware; for H100/A100 confirmation runs, one canary on H100 *or* A100 suffices (never block on an H100 proof when H100 returns 503).
4. **Automatic mechanical retries.** Verified stop + no outputs + known mechanical class (transport timeout, upload cap, allocation refused) → retry automatically with a fresh identity up to the package's limit, no new review/packet cycle; report afterwards.
5. **Transport registry holds several qualified trees.** Activating a fixed transport must never block or race jobs queued/running on the previous qualified tree.
6. **Dispatch.** Keep all Colab runtimes busy (measured account cap 6 total, L4 sub-cap 3 in `tools/ops/ready_queue.py` since 2026-10-31; poll scarce GPU stock `~1/min`, not every `20 min`) from a ready queue; don't ask the owner about operational steps; do multiple requested things in parallel.
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
