# T34 Watch6 exact-stop efficacy ceiling V3 (confirmation lane) — 2026-10-31

Status: PROPOSED_PRE_DATA_PENDING_ASTRA_RULING (`research/watch6_search_efficacy_v3`,
implementer: Argon; independent pre-data judge: Astra). Live dispatch is mechanically
blocked by `research/watch6_search_efficacy_v3/dispatch.py` until Astra's ratified
pre-data ruling receipt (`astra-ruling.json`, binding `REGISTRATION_V3.md` SHA256,
`science` identity, and Astra's `RULING_V3.md` SHA256) is recorded on the Mac via
`python -m research.watch6_search_efficacy_v3.ratify`.

Context: Watch6 V2 (`t34-watch6-exact-stop-efficacy-20261030-v2`) closed as
`INVALID_OR_INCOMPLETE` (`88/96` paired sessions exported; shard 5 lost all three
attempts to pre-start staging/RPC transport failures without starting compute, leaving
its 8 pairs missing under V2's rigid 2-retry ceiling). All 88 V2 pair payloads remain
sealed (`exports/` and `pair-*.json` untouched and never opened by V3). Per owner-adopted
`OPUS_META.md` §2 item 1 and §3 row 1 (2026-10-31), Watch6 V3 re-runs the identical
confirmation science on a completely fresh, disjoint seed bank with hardened,
outcome-blind failure handling.

## 1. Unchanged science, controller, product, and statistical decision rule

All scientific, product, simulation, controller, and statistical elements are
**unchanged** from `research/watch6_search_efficacy/REGISTRATION_V2.md` and reuse
`research.watch6_search_efficacy.{engine,product,run,analyze}` directly without editing
any V2 file:

- **Question:** Does the served L2-prime exact-stop action rule (`L >= 600`, `H = 300`)
  have worthwhile product efficacy when delivered synchronously at every eligible root
  in default Watch6 mechanics v2?
- **Hero & opponents:** Bare B8 s3 hero in slot 0
  (`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`), rowwise fp32
  CPU inference with advisory masking and normal-action fallback; five `frp3-s12`
  opponents in slots 1..5
  (`eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723`), each with its
  own `v8(lambda=8)` carry.
- **World & mirror:** Actual default `GameSession` Watch6 mechanics v2
  (`configs/mechanics_v2.yaml`, `SNAKE_WATCH_SOURCE=live`, `SNAKE_SERVE_PATHB=off`,
  `SNAKE_MECHANICS_V2=1`), continuous respawn for all six snakes, rectangular arena,
  no long-body injection, no terminal hero. Independently stepped `GridBatchSim` mirror
  verifies physical state, RNG, `vector61` carry, B8 Q-values, delivered actions, death
  events, and native `v8` kernel counters every frame (`mirror_identity_frames == 14400`,
  `0` mismatches; any mismatch or fallback row is `INVALID_OR_INCOMPLETE`).
- **Controller:** `DenseTrigger(min_length=600, interval=5)` resetting each respawn;
  at each eligible root with nonempty advisory mask, `exact_stop.run` evaluates full
  `H = 300` continuations (bare first, then descending raw Q / ascending action index,
  stopping at first survivor; all-die tie-break via unchanged `ceiling._choose`); bare
  B8 inside branch continuations; opponent `v8` active in every branch; root RNG/carry
  cloned and verified unmutated.
- **Panel & unit of inference:** Exactly `N = 96` complete paired sessions (`df = 95`),
  `H = 14400` frames per arm (10 min at 24 fps). Fixed alternating arm execution order
  (`bare` then `exact_stop` for even slot index; `exact_stop` then `bare` for odd slot
  index). One fixed-`N` Mac look after all 96 paired units are verified and all V3
  runtimes have `stop_verified == true` and `cleanup_error == null`.
- **Co-primary intersection-union pass bars (`SUCCESS_CONTRACT.md` §3, `P7_STATISTICS.md`):**
  1. Candidate minus bare `deaths / 1000 frames` two-sided paired Student-t 95% CI
     (`df = 95`) wholly below zero (`CI95_upper < 0`, nondegenerate `ptp > 0`).
  2. Fieller paired two-sided 95% lower limit on the ratio of pooled full-horizon means
     `(candidate - bare) / bare` `>= -0.05` (dead frames counted as `0` mass across all
     respawned lives; unbounded or disconnected Fieller set cannot pass).
- **Verdicts:**
  - `GO_EFFICACY_ONLY`: both co-primary bars pass (`ELIGIBLE_FOR_REPLAY_QUALIFICATION`).
  - `FAIL_EFFICACY`: nondegenerate death-delta `CI95_lower >= 0` (`CI95_upper > 0`), or
    bounded Fieller `CI95_upper < -0.05`
    (`NO_GO_FOR_REPLAY_SERVING_UNDER_THIS_REGISTRATION`).
  - `INCONCLUSIVE`: valid 96-pair union meeting neither `GO_EFFICACY_ONLY` nor
    `FAIL_EFFICACY` (`NO_GO_FOR_REPLAY_SERVING_UNDER_THIS_REGISTRATION`).
  - `INVALID_OR_INCOMPLETE`: any missing, duplicate, foreign, unstopped, hash-corrupt,
    mirror-mismatched, or unlawfully substituted unit (precedes science; no statistical
    look is taken).

## 2. Fresh deterministic seed allocation and disjointness audit

- **Namespace:** `NAMESPACE = "t34-watch6-exact-stop-efficacy-20261031-v3"`.
- **Primary bank (`i in 0..95`):**
  `primary_seed(i) = uint32_be(SHA256(f"{NAMESPACE}/primary/{i}")[:4])`.
- **1-to-1 reserve bank (`i in 0..95`):**
  `reserve_seed(i) = uint32_be(SHA256(f"{NAMESPACE}/reserve/{i}")[:4])`.
- **Excluded operational fixtures:** `FIXTURE_SEEDS = [34890123, 34890124, 34890125]`.
  Fixtures are used solely in Mac mirror/preflight and the short L4 canary (which
  exports `fixture-pair-00..02.json`, never `pair-XX.json`) and can never enter the
  96-unit scientific bank.
- **Pre-data freshness audit (`freshness.py`):** Verifies before preparation that all
  195 seeds (`96` primary + `96` reserve + `3` fixtures) are pairwise distinct and that
  all 192 scientific seeds (`primary | reserve`) are completely disjoint from:
  - Watch6 V1 (`t34-watch6-exact-stop-efficacy-20261030-v1`),
  - Watch6 V2 (`t34-watch6-exact-stop-efficacy-20261030-v2`),
  - Watch replay banks 01–04 (and 01–32 across `t34-watch-replay-fix-20261030-bank*`,
    `t34-watch-replay-20261030-bank*`, `t34-watch-replay-20261030-v1`, and parity
    fixtures),
  - All exploration and historical namespaces (`bl-plan-v3-c1-served-watch6-20261022/v1`,
    `bl-served-search-l2-20261021/v1`, `bl-c15-*`, `t15-*`, `t12*`, `t13*`, `t34-*`,
    `s1-*`, `q0-*`, `q1-*`, `t34-watch6-alllen-screen-20261031*`),
  - Committed JSON world banks (with positive controls) and `>= 7` historical FRP
    `training_exclusions.json` inventories, and visible repo/artifact text while
    **strictly excluding and refusing to open** any `**/exports/**` or `pair-*.json`.

## 3. Hardened failure handling (the only changes from V2)

### 3.1 Shard layout and empirical L4 W4 sizing (`24` shards × `4` pairs)

- Partition: `SHARDS = 24`, `WORKERS = 4` (`min_vcpu = 12`, L4 host, CPU fp32). Shard
  `s in 0..23` owns the 4 fixed slots `i in {s, s + 24, s + 48, s + 72}` (a single 4-worker
  wave per shard, cutting blast radius 4× relative to V2's 16-pair shards).
- Sizing (`costs.py`, outcome-blind reuse of verified L4 W4 bare-loop `0.0442842855 s/frame`
  and linear p95 `2883.4574 s` across 24 verified H6000 W4 replay sessions scaled to
  H14400 = `7557.9914 s/pair`, `1.5x` forecast margin, `20` min setup/export):
  - `base_minutes_per_shard = ceil(20 + 1.5 * 1 * 7557.9914 / 60) = 209` min (`3h 29m`).
  - Owner's 2× runaway wall guard: `timeout_minutes = 418` min (`6h 58m`, well below the
    660 base / 1320 effective ceiling).
- Canary qualification: one short same-pin L4 canary (`base_minutes = 15`, effective
  `timeout_minutes = 30`, identical uploaded input inventory, 3 tiny H12 fixture pairs on
  `W4` + 1 long-body H300 `branch_fixture` verifying native `v8` kernel execution)
  qualifies all 24 initial shard manifests via `finalize.py`.

### 3.2 Account-wide serialized staging (Astra `RULING_SHARD6` §2)

- `research/watch6_search_efficacy_v3/dispatch.sh` acquires an exclusive host `flock` on
  `<ops_dir>/watch6-staging.lock` before live dispatch:
  1. Waits silently until **zero** active jobs across `<ops_dir>/manifests/*.json` are in
     their staging phase (every active job has written `upload-measurement.json`,
     `receipt.json`, or terminal `EXIT=` in `<ops_dir>/logs/<name>.log`).
  2. Invokes the qualified operator `<operator_root>/tools/ops/dispatch.sh` silently,
     preserving exact stdout/stderr and exit codes so `ready_queue.py` capacity/lock
     backoff (`ACCOUNT_ASSIGNMENT_CAP = 6`) behaves identically.
  3. If a runtime was allocated (`returncode == 0`), **continues holding
     `watch6-staging.lock`** until this job's active output directory (`<output_dir>-retry-1`
     if accounting retry fired, else `<output_dir>`) writes `upload-measurement.json` or
     `receipt.json` (or its log records `EXIT=`, bounded by the 300s staging cap + margin).
- Thus **at most one runtime stages/uploads at a time across the account**, while
  already-staged computing shards run concurrently up to the account cap of 6.

### 3.3 Tier 1 — Pre-start zero-export transport failures (no 2-completion cap; bounded total cap)

- **Exact definition:** An attempt is a *pre-start transport failure* iff no scientific
  pair payload or child terminal summary exists (`exports == []`, no `exports/efficacy/pair-*.json`,
  `terminal is None`, no `terminal.json`) AND either:
  1. Dispatch refused pre-allocation or recorded a verified `colab-allocation-unavailable/v1`
     `LIVE` unallocated proof with zero leaked session, OR
  2. `receipt.json` records `status == "INCOMPLETE"`, `stop_verified is True`,
     `cleanup_error is None`, and a pre-start transport/bootstrap/staging/initial-lease/RPC
     failure before child scientific execution (`CHILD_NONZERO_EXIT` and any scientific /
     mirror / Q-value error are **never** pre-start transport failures).
- **Rule:** Pre-start transport failures have zero interaction with world seeds or snake
  dynamics (strictly MCAR before world instantiation). They **do not** consume the 2
  compute-started completion attempts and **do not** trigger reserve seeds.
- **Sane total cap:** Automatic queue `auto_retry` handles up to 2 consecutive zero-export
  attempts per queue lineage; `recover.py` issues fresh pre-start retry packets
  (`completion_attempt = 0`, `prestart_retries_used = k`, reusing the original passing
  canary receipt because uploaded input bytes are unchanged) up to
  **`MAX_PRESTART_RETRIES_PER_SHARD = 6`** per shard, **`MAX_PRESTART_RETRIES_STUDY = 72`**
  study-wide, and **`MAX_TOTAL_DISPATCHES_PER_SHARD = 10`** total dispatches of all kinds
  per shard. Exceeding any cap halts the study as `INVALID_OR_INCOMPLETE`.

### 3.4 Tier 2 — Partial-shard resume from verified pairs (`skip_verified: true`)

- Every completed pair is published atomically (`pair-XX.json`, `.sha256`, `.complete`,
  followed by outcome-blind `pair-XX.meta.json`, `.sha256`, `.complete`). On `SIGUSR1`
  or abrupt host disconnect, in-flight pairs are abandoned and never exported.
- If a shard **starts compute** (evidenced by `upload-measurement.json` passing staging
  and entering child execution, or any exported file / `terminal.json`) and fails with
  `stop_verified is True`, `cleanup_error is None`, and **external transport/VM loss**
  (`RPC_EXEC_POLL_TIMEOUT`, `TRANSPORT_TIMEOUT`, `RUNTIME_LOST`, or external `SUPERVISOR`
  transport/poll loss without child scientific fault):
  1. `resume.py` verifies all completed `pair-XX.json` + `pair-XX.meta.json` units from
     stopped receipts on the Mac using path, index, size, SHA256, and `.meta.json` only
     (never parsing outcomes) and records them in `verified-ledger.json`.
  2. `recover.py` prepares a fresh resume attempt (`compute_completion_attempt in {1, 2}`,
     up to **`MAX_COMPUTE_COMPLETIONS_PER_SHARD = 2`** compute-started retries after the
     initial compute-started attempt = **3 compute-started primary attempts** maximum)
     that uploads the updated `verified-ledger.json` (`skip_verified: true`, qualified by
     a ~2-minute short canary on the updated `input-metadata.tar.gz`) and runs **only**
     the unverified slots of that shard.

### 3.5 Tier 3 — Pre-registered outcome-blind 1-to-1 reserve seeds (strict MCAR boundary)

- **Exact activation rule:** Primary slot `i` (`i in 0..95`) is replaced 1-to-1 by its
  pre-registered reserve seed `reserve_seed(i)` (`SHA256(f"{NAMESPACE}/reserve/{i}")[:4]`)
  **if and only if all five conditions hold** on Mac outcome-blind inspection (`recover.py`):
  1. **Zero exported payload for slot `i`:** No verified `pair-{i:02d}.json` exists across
     any prior attempt (verified pairs are immutable and can **never** be replaced).
  2. **Primary compute attempts exhausted:** Slot `i`'s shard has completed `1 + 2 = 3`
     compute-started primary attempts, every one with `stop_verified is True` and
     `cleanup_error is None`.
  3. **Pure external infrastructure/transport failure on every failed compute attempt:**
     Every one of those 3 failed compute-started attempts terminated with external
     transport/VM loss (`failure_class in {"RPC_EXEC_POLL_TIMEOUT", "TRANSPORT_TIMEOUT", "RUNTIME_LOST"}`
     or `failure_class == "SUPERVISOR"` with transport/poll/session-loss error,
     `terminal is None`, and zero child traceback).
  4. **Strict exclusion of informative / duration-correlated / child failures (MCAR guard):**
     Reserve substitution is **FORBIDDEN** (and `recover.py` / `analyze.py` fail closed
     to `INVALID_OR_INCOMPLETE`) if **any** attempt on that shard recorded:
     - `CHILD_NONZERO_EXIT`, mirror mismatch, non-finite Q, illegal action, or rolling
       action-collapse fault; OR
     - Shard wall-clock timeout (`WALL_TIMEOUT` / `TIMEOUT` at the 418-minute shard wall)
       or the 900-second `faulthandler` stall watchdog (`TimeoutError` / watchdog exit).
       *(Rationale: runtime duration correlates with snake survival length and search-root
       count; replacing wall-timed-out seeds would censor long-body worlds and bias both
       co-primary endpoints.)*
  5. **Strict substitution caps:** At most **1 reserve seed per slot** (`reserve/{i}` gets
     at most **1 compute-started attempt**, plus normal pre-start zero-export transport
     retries within `MAX_TOTAL_DISPATCHES_PER_SHARD = 10`; there is no second reserve seed
     and no multi-compute retry if reserve compute fails) and at most
     **`MAX_RESERVE_SUBSTITUTIONS = 16` reserve-substituted slots study-wide** across
     `N = 96`. Exceeding either cap or failing during reserve compute closes V3 as
     `INVALID_OR_INCOMPLETE`.
- ** Provenance binding:** Every reserve-substituted slot is explicitly declared in the
  uploaded `runtime-contract.json` (`reserve_slots: [i, ...]`) alongside a Mac-verified
  `reserve-lineage.json` binding the 3 stopped external-transport receipts for that shard.
  Both `resume.py` and `analyze.py` re-verify those 3 receipts before accepting any unit
  with `seed_kind == "reserve"`.

## 4. Final-look integrity contract (`final_look.py`)

The single Mac final look (`python -m research.watch6_search_efficacy_v3.final_look`)
enforces all five lessons from the V2 post-mortem (`watch6-analysis-ready-1031/answer.md`):
1. Canary/fixture exports use `fixture-pair-*.json` and are structurally rejected from
   scientific reduction.
2.Attempt discovery scans `<ops_dir>/manifests/*.json` (covering initial queue jobs,
   `ops/auto-retry/*`, `-retry-1` accounting retries, and `recover.py` packets) rather
   than fragile single-directory globs.
3. Refuses to run if Astra's `astra-ruling.json` is absent/invalid or if **any** launched
   V3 manifest lacks a terminal `receipt.json` with `stop_verified is True`,
   `cleanup_error is None`, and `manifest_sha256 == sha256(executed_manifest)` (except
   proven unallocated pre-dispatch refusals with zero runtime created).
4. Builds `final-verified-ledger.json` via `resume.py` and re-verifies every unit's
   immutable `.sha256`/`.complete` sidecars, `.meta.json`, bound stopped `receipt.json`
   export inventory, and (for any `seed_kind == "reserve"`) the 3 stopped external-transport
   primary compute receipts and `<= 16` study-wide cap.
5. Executes `research.watch6_search_efficacy.analyze.decide` exactly once on the complete
   verified 96-pair union (`df = 95`) and publishes `efficacy-decision.json`.
