# Watch6-L1-V1 Pre-Registration: 3-Arm Fixed-Sequence Confirmation of All-Length Exact-Stop (`L >= 1`) vs Confirmed `L >= 600` (`l2_prime`) and Bare `b8-s3`

- **Study ID:** `t34-watch6-l1-confirm-20261101-v1`
- **Lane:** Confirmation (T34 ExIt-B8; binding under `SUCCESS_CONTRACT.md`, `P7_STATISTICS.md`, `ROADMAP.md` §§2–2.3, and Fast-Process Rule #1/#7)
- **Implementer:** Argon (independent pre-data review & ratification required from **Astra**, or Fable stand-in if unavailable, before any live primary Colab dispatch)
- **Date:** 2026-11-01
- **Status:** `PROPOSED_PRE_DATA_PENDING_JUDGE_RULING` (`0` primary outcomes generated; `0` Colab spend)
- **Package:** `research/watch6_l1_confirm/` (self-contained; sealed Watch6 V2 and V3 packages untouched)

---

## 1. Scientific Motivation & Pre-Data Context

1. **Watch6 V3 Outcome (`GO_EFFICACY_ONLY`, 2026-11-01):**
   Watch6 V3 (`research/watch6_search_efficacy_v3/`, `N = 96`, `H = 14,400`, `df = 95`) confirmed synchronous `exact-stop-l2-prime` (`DenseTrigger(min_length=600, interval=5)`, `exact_stop.run` at `H = 300`) over bare `b8-s3` on fresh Watch6-v2 worlds:
   - Hero death rate difference `(l2_prime - bare)`: **`-0.1250` deaths / 1k frames** (95% paired Student-t CI `[-0.1532, -0.0968]`, `UB95 < 0`).
   - Full-horizon pooled mass relative difference: **`+63.74%`** (95% paired Fieller CI `[+49.84%, +78.52%]`, `LB95 >= -5.0%`).
2. **All-Length Exploratory Screen (`research/watch6_alllen_screen/`, `N = 48`, `H = 6,000`):**
   On a disjoint `N = 48`, `H = 6,000` exploration screen (`audit_live_mirror=False`), removing the `L >= 600` gate so `exact_stop.run` (`H = 300`, `interval = 5`) triggers at all lengths `L >= 1` (`exact-stop-l1` / `l1`) eliminated `96%` of short-body (`L < 600`) deaths (`26 -> 1`), cutting total deaths/1k from `0.156` (`l2_prime`) to `0.052` (`-0.104/1k`, `-66.7%`) and raising mean mass from `605.7` to `703.4` (`+97.7`, `+16.1%`), with `1,268.5` mean branches/session (`2.82x` `l2_prime` at `H = 6,000`).
3. **Why Full-Horizon (`H = 14,400`) 3-Arm Confirmation Is Required:**
   The `H = 6,000` screen is exploration-lane evidence without per-frame live `GameSession` mirror auditing and does not observe frames `6,001..14,400`, where `l1` snakes spend more time at mature/large lengths (`L >= 800`). Per `docs/research/ROADMAP.md` §2 ("On `GO_EFFICACY_ONLY`", item 2) and §2.1 item 3, **Watch6-L1-V1** tests `l1` against both `bare` (Gate 1) and the newly confirmed `l2_prime` baseline (Gate 2, including a session-denominator late large-body `L >= 800` death non-inferiority guard) on `N = 96` fresh paired worlds at full `H = 14,400` (`10 min` at `24 fps`) with full `14,400`-frame live/`GridBatchSim` mirror checks and `record=None`.

---

## 2. Fresh Deterministic Seed Namespace & Zero-Reserve MCAR Rule

1. **Frozen Namespace:** `NAMESPACE = "t34-watch6-l1-confirm-20261101-v1"`.
2. **Primary Panel (`N = 96`, `index in 0..95`):**
   $$\text{primary\_seed}(i) = \text{uint32\_be}\!\left(\text{SHA256}(\texttt{"t34-watch6-l1-confirm-20261101-v1/primary/"} \,\|\, i)_{0..3}\right)$$
3. **Excluded Operational Fixture Seeds (Mac Mirror, Mac Preflight, and Colab Canaries Only):**
   `FIXTURE_SEEDS = [34910101, 34910102, 34910103]`. Fixture runs emit `fixture-world-XX.json` (`seed_kind = "fixture"`, `fixture = true`) and are unconditionally rejected by `resume.py`, `analyze.py`, and `final_look.py`.
4. **Pre-Data Disjointness Audit (`research/watch6_l1_confirm/freshness.py`):**
   Verifies internal uniqueness across all `99` (`96` primary + `3` fixture) seeds and exact zero overlap against Watch6 V1, Watch6 V2, Watch6 V3 (`primary` and `reserve`), `watch6_alllen_screen` (`main`, `canary`, `preflight`), `r6_l600_census`, `t4_gate1`, Watch replay banks `01..32` and L1 banks `14l1..32l1`, committed JSON world banks (`BANK_L2`, `BANK_ROUND0`, `BANK_ROUND0D`, `RETENTION_BANK_V2`, `WORLD_BANK`), `>= 7` FRP `training_exclusions.json` files, and single-threaded `rg --threads 1` scans of repository and `snake-dqn-artifacts` metadata without ever opening `**/exports/**`, `**/pair-*.json`, `**/world-*.json`, or `**/*.watch.gz`.
5. **Zero Reserve Seed Pool (`0` Substitutions):**
   Per Watch6 V3's ratified MCAR rule (`RULING_V3.md` §2), no reserve seed pool exists. Any primary slot `0..95` still missing after bounded outcome-blind mechanical retries forces `INVALID_OR_INCOMPLETE` with `0` outcome payloads opened.

---

## 3. Arms, Panel, Paired-World Atomic Units, and Live/Grid Mirror Contract

### 3.1 Three Frozen Arms (`ARMS = ("bare", "l2_prime", "l1")`)

| Arm key | Controller label | Hero checkpoint | Trigger (`DenseTrigger`) | Continuation search |
|---|---|---|---|---|
| `bare` | `bare-b8-s3` | `b8_s3_t7_20261016.pth` (`3e649783...`) | None (`min_length=None`) | None (`0` roots, `0` branches) |
| `l2_prime` | `exact-stop-l2-prime` | `b8_s3_t7_20261016.pth` (`3e649783...`) | `min_length=600`, `interval=5`, reset per life | Synchronous `exact_stop.run`, `H = 300`, native `v8` kernel |
| `l1` | `exact-stop-l1` | `b8_s3_t7_20261016.pth` (`3e649783...`) | `min_length=1`, `interval=5`, reset per life | Synchronous `exact_stop.run`, `H = 300`, native `v8` kernel |

### 3.2 Product World, Horizon, and `record=None`

- **World:** Shipped Watch6 product configuration (`configs/mechanics_v2.yaml`, `SNAKE_WATCH_SOURCE=live`, `SNAKE_SERVE_PATHB=off`, `SNAKE_MECHANICS_V2=1`, `rowwise` forward, continuous respawn, `max_capacity=14704`), `b8-s3` hero in slot `0` vs `5 x frp3-s12` (`frp3_m3_s12_u60000_20261005.pth`, SHA-256 `eec144bf2bc6e5a7f61fd9ff7d6fb5be3c902496d7be75a16552ab0b5cb6f889`) + `v8(lambda=8.0)` opponents in slots `1..5`.
- **Horizon & Panel:** `N = 96` paired worlds (`df = 95`), `H = 14,400` frames per arm (`10 minutes` at `24 fps`; `43,200` mirrored frames per paired world; `4,147,200` mirrored frames total).
- **`record=None`:** No `.watch.gz` replay frames are recorded during efficacy confirmation. When `record is None`, read-only WebSocket DTO serialization (`session.live.snapshot()`) is omitted while preserving 100% of `session._step_live(actions)` and per-tick live/`GridBatchSim` mirror verification (verified bit-identical to `snapshot()`-enabled execution by automated test).

### 3.3 Paired-World Atomic Units & Balanced Arm Execution Order

- **Atomic Unit (`world-{index:02d}.json`):** One completed unit is a single paired world `index in 0..95` containing all three arms `{bare, l2_prime, l1}` executed sequentially inside the **same worker process on the same host**. A paired world is **never** split across hosts or hardware tiers, ensuring within-world paired differences `(l1 - bare)` and `(l1 - l2_prime)` are immune to cross-CPU FP32 tie differences between A100 and L4 VMs.
- **Cyclic 6-Permutation Latin Execution Order:** Arm execution order within world `index` is `ARM_ORDER_PERMUTATIONS[index % 6]`, cycling through all `3! = 6` permutations of `("bare", "l2_prime", "l1")`. Across `N = 96 = 16 x 6` worlds, each permutation runs `16` times total (`12` times across A100 worlds `0..71` — once per 6-world A100 shard — and `4` times across L4 worlds `72..95`).
- **Per-Worker Watchdog:** Each worker process arms `faulthandler.dump_traceback_later(900, exit=True)` renewed on every frame (`on_frame`) and cancelled on unit exit; any 900 s hang aborts with a `faulthandler` trace that is strictly non-retryable.

### 3.4 Full 14,400-Frame Live / `GridBatchSim` Mirror & Kernel Integrity Checks

On **every tick `1..14,400` of every arm**, `engine.run_arm` steps both `GridBatchSim` and live `GameSession` (`session._step_live`) and fails closed (`ScientificFault`, exit `86`, non-retryable) on any violation of:
1. Physical world state, Python RNG state, `vector61` carry, and `v8` veto state digest equality (`_step_live`).
2. Hero B8 dispatch Q-row exact bitwise equality (`np.array_equal`) and pre-step decision length equality.
3. Hero death event parity (`hero.id in frame_death_causes` vs `transition_valid & ~alive`), delivered `hero.last_action`, post-step `hero.is_alive`, post-step `hero.length`, and all-6-slot respawn counts (`respawns_per_slot`).
4. `mirror_identity_frames == 14,400` on all three arms.
5. Native `v8` kernel attestation on every search root (`kernel_rows > 0`, `stock_rows == fallback_rows == noncandidate_rows == 0`) and exact root-world immutability across `exact_stop.run`.
6. Finite hero/opponent neural Q-values, legal action selection, and `< 90%` single-action non-collapse at `>= 100,000` decisions.

---

## 4. Fixed-Sequence Gates, Statistical Estimators, and Verdict Set

### 4.1 Multiplicity Control & Parameterized `alpha`

- **Unit of inference:** Paired world `i in 0..95` (`N = 96`, `df = 95`).
- **Confirmation level:** Two-sided `alpha = 0.05` (`1 - alpha = 0.95`, `t_{0.975, 95} = 1.985251...`), implemented via parameterized `alpha: float = 0.05` in `research/watch6_l1_confirm/analyze.py`.
- **Fixed-Sequence Hierarchy (`P7_STATISTICS.md` §3):**
  1. **Gate 1 (`l1` vs `bare`)** is tested first at full two-sided level `alpha = 0.05`.
  2. **Gate 2 (`l1` vs `l2_prime`)** is tested for confirmation **if and only if Gate 1 passes**, also at full two-sided level `alpha = 0.05`.
  Because Gate 2 cannot pass unless Gate 1 passes, family-wise Type I error across the hierarchy is strongly controlled at `alpha = 0.05` without alpha splitting.
- **Descriptive Replication (`l2_prime` vs `bare`):** Reported as a non-gating diagnostic only (Watch6 V3 already confirmed `l2_prime` vs `bare` on `N = 96`, `H = 14,400`).

### 4.2 Gate 1 (`l1` vs `bare`, `alpha = 0.05`, `df = 95`) — Both Bars Required

1. **Hero Death Rate Superiority (`deaths/1k UB95 < 0`):**
   Let $d_i^{(1)} = (\text{deaths}_{l1,i} - \text{deaths}_{\text{bare},i}) \times \frac{1000}{14400}$.
   Pass iff sample differences are nondegenerate ($\max_i d_i^{(1)} - \min_i d_i^{(1)} > 0$) and the two-sided 95% paired Student-t upper bound satisfies $\text{UB}_{95}(d^{(1)}) < 0.0$.
2. **Full-Horizon Pooled Mass Non-Inferiority (`Fieller LB95 >= -0.05`):**
   Let $M_{a,i} = \frac{1}{14400}\sum_{t=1}^{14400} L_{a,i,t} \cdot \mathbb{1}[\text{alive}_{a,i,t}]$ (dead/respawn-cooldown frames contribute `0` mass).
   Pass iff the two-sided 95% paired Fieller confidence set on $\frac{\bar{M}_{l1} - \bar{M}_{\text{bare}}}{\bar{M}_{\text{bare}}}$ is bounded and connected ($a > 0$, $\text{disc} \ge 0$) and its lower bound satisfies $\text{LB}_{95} \ge -0.05$ (`-5.0%`).

### 4.3 Gate 2 (`l1` vs `l2_prime`, Tested ONLY IF Gate 1 Passes, `alpha = 0.05`, `df = 95`) — All Three Bars Required

1. **Hero Death Rate Superiority Over `l2_prime` (`deaths/1k UB95 < 0`):**
   Let $d_i^{(2)} = (\text{deaths}_{l1,i} - \text{deaths}_{l2',i}) \times \frac{1000}{14400}$.
   Pass iff sample differences are nondegenerate and $\text{UB}_{95}(d^{(2)}) < 0.0$.
2. **Full-Horizon Pooled Mass Non-Inferiority Over `l2_prime` (`Fieller LB95 >= -0.025`):**
   Pass iff the two-sided 95% paired Fieller confidence set on $\frac{\bar{M}_{l1} - \bar{M}_{l2'}}{\bar{M}_{l2'}}$ is bounded and connected and satisfies $\text{LB}_{95} \ge -0.025$ (`-2.5%`).
3. **Session-Denominator Late Large-Body Death (`L >= 800`) Non-Inferiority Guard (`Δd_ge800/1k UB95 <= +0.05`):**
   For each session $i$ and arm $a$, let $c_{a,i}^{(800)} = \sum_{k} \mathbb{1}[\text{death\_length}_{a,i,k} \ge 800]$ (number of hero deaths in session $i$ occurring from pre-step length $\ge 800$; `0` if none occur) and $g_i = (c_{l1,i}^{(800)} - c_{l2',i}^{(800)}) \times \frac{1000}{14400}$ (clustered by complete `14,400`-frame session, avoiding post-treatment conditioning on time spent at $L \ge 800$).
   Pass iff the two-sided 95% paired Student-t upper bound satisfies $\text{UB}_{95}(g) \le +0.05$ deaths per `1,000` session frames (if $g_i = 0$ for all $i = 0..95$, the zero-variance interval is $[0.0, 0.0]$, which satisfies $0.0 \le +0.05$).

### 4.4 Exhaustive Pre-Registered Verdict Set

| Verdict | Condition | Operational / Product Action |
|---|---|---|
| `INVALID_OR_INCOMPLETE` | Pre-outcome gate failure (unratified, missing/unstopped runtime, canary missing, forked or `> 2` compute-retry lineage, forbidden failure class, hash/mirror mismatch, or `< 96/96` primary worlds); `0` outcome payloads opened | No scientific claim; diagnose mechanical/integrity fault |
| `GO_L1_SUPERIOR` | Gate 1 PASS **and** Gate 2 PASS | `exact-stop-l1` confirmed superior to both `bare` and `exact-stop-l2-prime` at `H = 14,400`; eligible for `exact-stop-l1` replay bank expansion + Mac `24 fps` serving qualification + owner release sign-off (`SUCCESS_CONTRACT` §4) |
| `KEEP_L2_PRIME_GATE1_ONLY` | Gate 1 PASS **and** Gate 2 does NOT pass (`FAIL_GATE2` or `INCONCLUSIVE_GATE2`) | `l1` beats `bare` but does not confirm superiority + `L >= 800` safety over `l2_prime`; retain confirmed `exact-stop-l2-prime` bank |
| `FAIL_GATE1` | Gate 1 fails by demonstrated non-efficacy/harm ($\text{LB}_{95}(d^{(1)}) \ge 0$ with $\text{UB}_{95} > 0$, or bounded Fieller $\text{UB}_{95} < -0.05$); Gate 2 not tested | Reject `exact-stop-l1` at `H = 14,400`; retain confirmed `exact-stop-l2-prime` bank |
| `INCONCLUSIVE_GATE1` | Gate 1 neither passes nor meets `FAIL_GATE1`; Gate 2 not tested | `exact-stop-l1` unconfirmed; retain confirmed `exact-stop-l2-prime` bank |

---

## 5. Six-Runtime Sharding Plan & Measured Receipt Cost Forecast

### 5.1 Measured Receipt Basis (`research/watch6_l1_confirm/costs.py`)

All timing and billing inputs come from verified Colab receipts (`0` `exports/` or `pair-*.json` files opened):
1. **Watch6 V3 (`24/24` PASS `L4 W = 4` single-wave shards, `4` pairs/shard, `H = 14,400`, 2-arm `{bare, l2_prime}` with live mirror):**
   - `snake-dqn-artifacts/ops/manifests/watch6-v3-fix3-shard-*.json`: whole-shard `elapsed_seconds` `min = 6,755.3 s`, `mean = 10,312.4 s` (`2.865 h`), `median = 10,334.1 s`, `p95 = 12,058.8 s` (`3.350 h`), `max = 12,313.7 s` (`3.420 h`).
   - Calibrated L4 provider rate across all 24 `rate-calibration.json` files: **`1.94 u/h`** (manifest priced ceiling: **`2.5 u/h`**).
   - Net `{bare + l2_prime}` `W = 4` compute (subtracting `450 s` VM setup): **`9,862.4 s` (`2.740 h`) mean**, **`11,608.8 s` (`3.225 h`) p95**.
2. **All-Length Screen (`3/3` PASS `A100 W = 6` 2-wave shards, `H = 6,000`, `t34-sol/argon-alllen-a100/`):**
   - `s0-01` (`12` worlds): `15,907.4 s` (`29.75 u`); `s1-01` (`12` worlds): `15,557.2 s` (`29.08 u`); `s2-02` (`11` worlds): `16,059.8 s` (`30.02 u`).
   - Calibrated A100 provider rate: **`6.73 u/h`** (manifest priced ceiling: **`8.0 u/h`**).
   - Measured 12-vCPU `W = 6` vs `W = 4` contention factor: **`1.03128x`** (`415.72 s / 403.11 s`).
3. **L1 Replay Banks `14l1–16l1` (`3/3` PASS `A100 W = 6` 2-wave shards, `12` units/shard, `H = 6,000`, 1-arm `exact-stop-l1` WITH live mirror):**
   - `bank14l1`: `11,410.0 s`; `bank15l1-retry-1`: `11,173.8 s`; `bank16l1`: `11,098.9 s` (mean `11,227.6 s` -> **`5,388.8 s/wave`** net compute on `A100 W = 6`, or **`5,225.3 s/wave`** on `L4 W = 4`).
   - Horizon scaling of `l1` from `H = 6,000` to `H = 14,400`: because `l1` already triggers at the physical 5-tick ceiling during `0..6,000` (`1,200.1` roots in `6,000` frames), `l1` root count scales strictly linearly by `14,400 / 6,000 = 2.40x` (`<= 2,880` roots), while mature-coil branch weight in `6,000..14,400` yields **`2.65x` central** (`2.40x` linear lower bound to `3.05x` conservative p95 upper bound).

### 5.2 Registered Single-Wave 18-Shard Layout for 6 Runtimes (`3 A100 W=6 + <=3 L4 W=4`)

Single-wave shards (`6` worlds/shard on `A100 W = 6`; `4` worlds/shard on `L4 W = 4`) keep expected runtime at `~6.7–6.9 h/shard` (`p95 ~7.8–8.0 h/shard`) and Owner's 2x wall timeout at `18–19 h` (safely below Colab's `24 h` VM lifetime cap):

| Shards | Target GPU | Workers (`W`) | Worlds / shard | Total worlds | World indices | Permutations (`i % 6`) | Expected wall / shard | P95 wall / shard | `base_minutes` (`est_minutes`) | 2x `timeout_minutes` | Priced ceiling (`u/h`) | 2x `unit_cap` (`u`) | Expected spend / shard (`@6.73` / `@1.94`) |
|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| `00..11` (`12` shards) | `A100` | `6` | `6` (`1` wave) | `72` | `6*s .. 6*s+5` (`0..71`) | Exact `1` of each `0..5` per shard (`12` each) | `6.92 h` (`415 min`) | `8.02 h` (`481 min`) | `570 min` (`9.5 h`) | `1,140 min` (`19.0 h`) | `8.0` | `152.0` | `46.55 u` |
| `12..17` (`6` shards) | `L4` | `4` | `4` (`1` wave) | `24` | `72+4*(s-12) .. +3` (`72..95`) | Exact `4` of each `0..5` across L4 (`4` each) | `6.71 h` (`403 min`) | `7.78 h` (`467 min`) | `540 min` (`9.0 h`) | `1,080 min` (`18.0 h`) | `2.5` | `45.0` | `13.02 u` |
| `canary-a100`, `canary-l4` | `A100`, `L4` | `6`, `4` | `2` fixture (`H=60`) | `0` primary | Excluded `FIXTURE_SEEDS` | Excluded | `~0.15 h` | `~0.22 h` | `15 min` | `30 min` | `8.0`, `2.5` | `4.0`, `1.25` | `~1.4 u` combined |

- **Aggregate Forecast (Registered `12 A100 + 6 L4` Plan across `3 A100 + <=3 L4` Concurrent Runtimes):**
  - **Expected observed spend:** `12 x 46.55 u` (`558.5 u` A100) + `6 x 13.02 u` (`78.1 u` L4) + `~1.4 u` canaries = **`~638.1 u`** (linear-to-p95 range **`598.1–739.5 u`** before retries).
  - **Expected calendar wall clock:** `4` A100 waves on `3` concurrent A100s = **`~27.7 h`**; `2` L4 waves on `3` concurrent L4s = **`~13.4 h`** (or `3` waves on `2` L4s = `~20.1 h`, leaving L4 headroom for other lanes).
  - *(Pre-ratification alternative mixes computed in `costs.py`: `10 A100 + 9 L4` [`19` shards] = `~584.1 u`, `~27.7 h`; `8 A100 + 12 L4` [`20` shards, wall-matched] = `~530.1 u`, `~26.8 h`; `24 L4` [`24` shards, pure-L4 fallback] = `~312.8 u`, `~53.7 h`.)*

---

## 6. Linear Single-Retry Lineage, Input Staging, and Sole-Final-Look Rules

1. **Single-Archive Staging (`<= 4` Transfers, `lease_seconds = 600`):**
   `uploads.py` bundles all 7 small immutable inputs + sidecars (`runtime-contract.json`, `verified-ledger.json`, `judge-approval.json`, `mac-mirror.json`, `cost-evidence.json`, `release-evidence.tar.gz`, `public-inputs.json`) into one `< 2 MiB` `input-metadata.tar.gz`. The VM unpacks atomically and fetches `b8-s3` and `frp3-s12` from `s1-inputs-20261019` with SHA-256 verification against `web/backend/served_checkpoint_pins.json`.
2. **Decoupled Science Identity:**
   `protocol.identity()` hashes `REGISTRATION_L1_V1.md`, `research/watch6_l1_confirm/` science modules, `job.sh`, `src/`, `configs/mechanics_v2.yaml`, `served_checkpoint*.py/.json`, `ego2s_policy.py`, `session.py`, and upstream search/simulation modules, while excluding `tools/ops/*`, `research/colab_cli/*`, `web/backend/{app,watch_replay,serialize}.py`, and `__pycache__`.
3. **Hardware-Matched Canary Gate (`canary.py:302`):**
   Every A100 main shard (`00..11`) binds a passing same-pin `A100` canary receipt and every L4 main shard (`12..17`) binds a passing same-pin `L4` canary receipt, each attesting `H = 300` native `v8` kernel execution (`kernel_rows > 0`, `fallback_rows == 0`).
4. **Unbroken Linear Per-Shard Attempt Lineage (`MAX_COMPUTE_COMPLETIONS_PER_SHARD = 2`):**
   For each shard `s in 0..17`, all attempts must form a single unbranched chain $A_1 \to A_2 \to \dots \to A_M$ (`1` root, `0` forks, `0` attempts after `PASS`, terminal $A_M$ `PASS`, verified stop on every assigned VM):
   - **Pre-start failures (unlimited outcome-blind mechanical retries):** Positively proven pre-`ready.json` failures (including pre-assignment `412`/`503` refusals with `endpoint is None` and `ALLOCATION_UNRESOLVED: retained pending record`, warmup/calibration/staging/`UPLOAD_CAP` failures) or pre-`workload-started.json` VM checkpoint-fetch `rc = 74` failures with `0` files in `exports/`.
   - **External-transport / lease compute failures (`<= 2` retries per shard; `<= 3` compute-started attempts total):** Compute started (`hardware.json` present) and stopped cleanly (`stop_verified is True`, `cleanup_error is None`) purely on verified host/CLI transport loss (`RPC_EXEC_POLL_TIMEOUT`, `SUPERVISOR`, `TRANSPORT_TIMEOUT`, `RUNTIME_LOST`, provider-lag `ACCOUNTING`, renewal-WebSocket drop, or `CHILD_NONZERO_EXIT` solely with `terminal.reason in {"lease_expired", "release_credential_expiring"}`) with `0` scientific, mirror, numerical, `faulthandler` watchdog, or wall-timeout faults.
   - **Forbidden non-retryable failures (`0` retries -> `INVALID_OR_INCOMPLETE`):** Any `ScientificFault` (`rc = 86`), live/`GridBatchSim` mirror mismatch, `NaN`/non-finite Q, fallback row, Python exception in `src/`/`web/`/`research/`, `faulthandler` 900 s watchdog, or `2x` wall/unit timeout (`terminal.reason == "timeout"`).
   - **Study-Wide Union `verified-ledger.json`:** Recovery waves build one shared union `verified-ledger.json` from path/size/SHA-256/sidecars only (`resume.py`), and each shard skips verified indices in `shard_indices(s)` without ever recomputing a verified unit.
5. **Sole-Final-Look Rule (`final_look.py`):**
   `final_look.py` activates a runtime `sys.addaudithook` (`pre_outcome_open_guard`) that blocks opening any `world-XX.json` payload (allowing only raw `"rb"` SHA-256 hashing inside `tools/ops/package_spec.py:digest`) until pre-data ratification, hardware-tier canaries, verified stops, linear per-shard retry lineages, sidecars, and exact `96/96` primary slot bijection all pass and `pre-outcome-accounting.json` + `final-verified-ledger.json` are sealed on disk. Only then is the guard lifted for the single confirmation evaluation of `analyze_verified_union(..., alpha=0.05)`.
