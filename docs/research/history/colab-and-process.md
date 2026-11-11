# Colab and Process

<a id="project-status-and-indexes"></a>
## Project Status, Top-Level Indexes, and Compendium Provenance (`project-status-and-indexes`)

**Dates:** 2026-06 to 2026-11-01
**Verdict:** Active index and status lineage (`21` kept entry-point files) centralizing live campaign and serving truth in `AGENTS.md`, `docs/STATUS.md`, and `docs/INDEX.md`, supported by the `2026-10-25` `10`-fact-sheet / `14`-writer `docs/notes/` compendium (`4` intermediate staging files consolidated).

### What was tried
- Bannered early June–September compendia (`project_history_and_findings.md`, `research_findings_2026-09-24.md`) as historical on `2026-10-08` and centralized live state in `AGENTS.md`, `docs/STATUS.md`, and `docs/INDEX.md`.
- Recorded sequential serving promotions: `frp3-s12 + v8` (`2026-10-06`), live bare `b8-s3` (`2026-10-17`), and the `132`-session confirmed `exact-stop-l2-prime` Watch replay bank (`watch-replay-bank-live-132`, `2026-11-01` after Watch6 V3 `GO_EFFICACY_ONLY`).
- Built the `docs/notes/` topic compendium (`notes-cleanup-20261025`, `2026-10-25`): `10` Opus agents extracted cited `_facts/` sheets, reviewed by Fable, before `14` Astra writers produced `13` topic notes + `GLOSSARY.md`, alongside `MAP.md` and `TIMELINE.md` (`2026-06-16` to `2026-11-01`).

### What happened
- **Play & rollback serving state (as of `2026-11-01`):** Play serves live bare **`b8-s3`** (`Ego2sNet`, `ego2s-bv8`, `~1.64M` params, `1` human vs `1–11` AI, default `5`), with `frp3-s12 + v8` (`@60000`) and `champion_a5 + v8` (`61 → 512 → 256`) retained as rollbacks.
- **Watch replay bank (`2026-11-01`):** Watch serves `132` `6,000`-frame `exact-stop-l2-prime` sessions (`L >= 600`, interval `5`, `H = 300` vs `5× frp3-s12 + v8`; live bare `b8-s3` fallback) backed by Watch6 V3 (`96` fresh paired `H = 14,400` sessions: **`−0.125` deaths/1k**, `CI95 −0.153 .. −0.097`; **`+63.7%` pooled mass**, Fieller `CI95 +49.8% .. +78.5%`).
- **Compendium consolidation:** Kept all `21` top-level navigation/topic indexes while consolidating `4` `2026-10-25` worker briefs and raw fact sheets (`STYLE.md`, `_facts/overview-and-serving.md`, `ASTRA_BRIEF.md`, `COMMON_BRIEF.md`).

### Why
Maintaining one canonical current-truth trio (`AGENTS.md`, `docs/STATUS.md`, `docs/INDEX.md`) while bannering older summaries prevents agents from following superseded instructions, and separating cited fact extraction (`_facts/`) from synthesis (`docs/notes/`) preserved exact hashes across `14` parallel writers.

### What to reuse / what not to repeat
- Always treat `docs/STATUS.md` + `docs/INDEX.md` + `AGENTS.md` as current truth (noting post-`2026-11-01` Watch upgrades such as `exact-stop-l1-os` in `STATUS.md`), and banner superseded summaries explicitly.
- For large multi-agent documentation syntheses, stage cited fact sheets first so narrative writers never invent numbers or paths.

### Full record
Original: `git show 90b3db94:docs/research/history/project-status-and-indexes.md` | Primary sources: [STATUS.md](../../STATUS.md), [INDEX.md](../../INDEX.md), [README.md](../../notes/README.md)

<a id="architecture-and-specs"></a>
## Architecture, Specifications, and Test-Suite Hygiene (`architecture-and-specs`)

**Dates:** 2026-07 to 2026-10-28
**Verdict:** Active reference lineage (`ml_algorithm.md`, `simd_env_spec.md`, `WEB_MAP.md`, `TESTS_MAP.md`, `testing_openmp_macos.md`, `tests/fixtures/`) and `PASS` on the `2026-10-20` test-suite hygiene repair (`9,082` passed, `0` non-Inductor failures, `0` production/research source bytes modified).

### What was tried
- Documented the Apex vector stack (`ml_algorithm.md`), file:line v1/v2 `GameState` parity spec for `src/simd_env/` (`simd_env_spec.md`), September codebase assessment (`codebase_quality_2026-09.md`), and read-only maps of `web/` (`WEB_MAP.md`) and `tests/` (`TESTS_MAP.md`, `556` entries / `545` `test_*.py` files).
- Repaired duplicate macOS `libomp.dylib` loading (torch vs scikit-learn) in `tests/conftest.py` and `tests/openmp_runtime.py` (`testing_openmp_macos.md`) and pinned `5` frozen regression fixtures under `tests/fixtures/`.
- Executed a test-only hygiene repair (`t34-test-hygiene-20261020`, `73a2d794 -> e301cfeb`) across `6` test files while verifying byte invariance across `936` protected source digests.

### What happened
- **macOS OpenMP contract (`ACTIVE`):** Routing torch's `lib` first in `DYLD_LIBRARY_PATH` via single-restart `tests/conftest.py` + dyld image guard eliminated duplicate `libomp.dylib` crashes without setting `KMP_DUPLICATE_LIB_OK` (and dropped `--showlocals` from `pytest.ini`).
- **Repaired `20` non-Inductor failures:** Fixed `5` `time.perf_counter()` parity diffs (`test_perf_actor.py`, `test_perf_opp.py`, `test_profile_m3c.py`), `13` wall-clock-expired Q1 A5 fixtures (`test_q1_admission_integration.py`), and `2` `membership` signature mismatches against the frozen `v8_decide` oracle (`test_t22_frozen_pool.py`).
- **Full-suite outcome (`0` source bytes changed):** Lifted the full suite from `8,971` passed / `35` failed (`2,681.97 s`) to **`9,082` passed, `15` Inductor-only failed (`@rpath/libc++.1.dylib` in `test_perf_compile.py`, left visible), `30` skipped (`2,885.06 s`)** with `936` protected source digests unchanged.

### Why
Lockstep actor parity tests had compared live `time.perf_counter()` stamps across runs, Q1 admission tests compared a frozen historical dispatch window against real wall time, and `v8` feature adapters added `membership` after the `t22` oracle was pinned.

### What to reuse / what not to repeat
- Strip wall-clock telemetry (`WALL_CLOCK_ACTOR_FIELDS`) from deterministic payload parity assertions and pin synthetic test clocks so historical admission tests never expire.
- Route macOS OpenMP explicitly via `DYLD_LIBRARY_PATH` + dyld image checks rather than `KMP_DUPLICATE_LIB_OK`, and keep environment-specific TorchInductor failures visible rather than blanket-skipping.

### Full record
Original: `git show 90b3db94:docs/research/history/architecture-and-specs.md` | Primary sources: [TESTS_MAP.md](../../architecture/TESTS_MAP.md), [WEB_MAP.md](../../architecture/WEB_MAP.md), [simd_env_spec.md](../../simd_env_spec.md)

<a id="governance"></a>
## Research Governance, Statistical Amendments, and CI/Rules Audits (`governance`)

**Dates:** 2026-09-26 to 2026-10-30
**Verdict:** Active governance record: established 3-tier evidence gates, ratified sequential OBF strict gates, survival band `v2` (`pooled_ni_continue`), `LH-1`, and sequential Phase R, pinned strict gates and Python CI (`macos-15` ARM64, `T14`) to Darwin ARM64 after cross-platform float/`erf` drift, and ratified the `2026-10-19` Campaign Authority + `2026-10-30` Fast-process rule.

### What was tried
- Defined Tier `0` (saved-data), Tier `1` (dev screen), and Tier `2` (strict promotion) gates (`governance_tiers_2026-09-26.md`) and CPU-slot thermal rules (`compute_policy_2026-10-02.md`).
- Ratified statistical amendments (`2026-10-02` to `10-08`): `4`-look OBF strict gates, `paired_ni_at_stop` -> band `v2` (`pooled_ni_continue`, `0.05` pooled NI, `0.075` per-mix floor, `0.30` floor), `LH-1` (`48` H10000 worlds/mix, sibling erratum preserving hash `85c3b851...`), and opt-in `sequential-phase-r-obf-hk-v1` (`strict_on_runpod` superseded).
- Repaired CI on `macos-15` ARM64 (`T14`, `fe69d208`: `torch==2.9.1`, pytest `9`, isort `8`, `76`-test numerical preflight).
- Conducted the `2026-10-19` `3`-lane Astra + Fable rules audit (`P7_STATISTICS.md`) and adopted the `2026-10-30` Fast-process rule (*Exploration* vs *Confirmation* lanes).

### What happened
- **Mac-only strict gates (`strict_on_runpod` SUPERSEDED):** Mac Accelerate vs x86 MKL AVX2 broke exact float32 Q ties differently in **`~1` in `50` H10000 episodes (`1/54` in `FRP-v5-S2`)**, keeping strict and serving gates on Mac.
- **`T14` CI repair (`PASS`):** Linux vs Darwin `erf` differed in the **last bit** (`1.941938163774193e-06` vs `1.941938163829704e-06` at `4.617501629587915`) and macOS `26` failed `6` Metal tests (issue `14380`); `macos-15` ARM64 passed **`6,945` Python + `178` frontend tests**.
- **Owner-ratified campaign & fast-process rules (`2026-10-19`, `2026-10-30`):** Ratified `15` audit fixes (paired world = fixed-controller unit, released bare `B8 s3` comparator, survival guard A/B, fail-closed provider lag, `CODEX_MODEL` pin) and split *Exploration* (no pre-registration ceremony, `1` reviewer) from *Confirmation* (pre-data bars, fresh paired worlds, verified stops).

### Why
Hardware-specific BLAS reductions (`arm64` Accelerate vs `x86` AVX2) and OS math libraries (`erf` last bit, macOS `26` Metal) break bit-exact trajectory and SHA-table parity across platforms, while fixed controllers over frozen weights have zero training-seed variance (making paired worlds the true unit of inference).

### What to reuse / what not to repeat
- Keep release confirmation gates on Mac, Python CI on `macos-15` ARM64 (`torch==2.9.1`), and SHA-bound governance docs immutable (use sibling errata, as on `LH-1`).
- Use the *Exploration* lane for screens/pilots/speed work and reserve *Confirmation* ceremony for releases (`AGENTS.md`, `P7_STATISTICS.md`).

### Full record
Original: `git show 90b3db94:docs/research/history/governance.md` | Primary sources: [governance-and-owner-rules.md](../../notes/governance-and-owner-rules.md), [P7_STATISTICS.md](../final_run_20261019/P7_STATISTICS.md), [t14_ci_validation_2026-10-17.md](../t14_ci_validation_2026-10-17.md)

<a id="colab-ops"></a>
## Colab CLI Supervisor, Legacy Queue Worker, RunPod/H100 Operations, and Small-Bench Fast-Path

**Dates:** 2026-10-02 to 2026-10-31
**Verdict:** Qualified and live (`research/colab_cli/` + `tools/ops/`, `ACCOUNT_ASSIGNMENT_CAP = 6`, `ACCOUNT_L4_CAP = 3`): retired RunPod strict gates after `~1` of `50` (`1/54`) `H10000` float32 Q-tie flips, lifted A100 self-play to `15,927 tr/s` (`3.0×` Mac) and L4 `N=3` to `7,625 tr/s` (`T17`/`T22`/`T24`), and repaired `UPLOAD_CAP` staging via `metadata_archive.py` and kernel-UUID supervision.

### What was tried
- Tested RunPod `cpu3c` x86 vs Mac M5 Pro consistency (`28` H5000 episodes + `diag-divergence`) and `runpod_fanout`, then froze legacy Drive queue `research/colab_queue` (`2026-10-18`).
- Built `uv` Python `3.12` Colab bundles (`T16`/`T16b`), added multi-thread learner gather (`--gather-threads 3/4 --no-obs-pack`, `T17`/`T17b`), and wired Numba `v8` + shared body-release maps into static-pool actors (`T22`/`T24`) across Mac MPS, L4 (`12` vCPU), and A100.
- Built and qualified the Mac OAuth2 supervisor around `google-colab-cli 0.7.4` (`research/colab_cli`, `2`-poll grace & carry-over, kernel-UUID readiness, multi-tree qualification registry, and `ready_queue.py` auto-retry).
- Diagnosed `4` Oct 23–24 `UPLOAD_CAP` staging failures (`colab_upload_cap_diagnosis_2026-10-24.md`), built `research/metadata_archive.py` (`<= 2 MiB` tar), and audited `249.27 GiB` across `250` `snake-dqn-artifacts/` folders.

### What happened
- **RunPod retired for strict gates:** `cpu3c` ran `3.66×` slower (`1,813 s` vs `508 s`) with `4.8e-5`–`8.4e-5` (`2,176–3,072` ULP) Q-diffs, and `diag-divergence` proved `arm64` vs `x86_64` float32 Q ties flip in **`~1` of `50` H10000 episodes (`1/54` in S2)**.
- **A100 learner & host-gather optimization (`T17b`):** Cut `B=256` gather `3.024 -> 1.214 ms` and lifted A100 self-play `11,341 -> 15,927 tr/s` (`62.2 upd/s`, `3.0×` Mac MPS `5,459 tr/s`), missing the `84 upd/s` (`4×`) bar because isolated fp32 learner updates topped out at `78 upd/s`.
- **Static-pool speedups (`T22`+`T24`) & L4 economics:** Cut frozen-pool CPU `-30.2%` (`41.109 -> 28.687 ms`) and `L=1200` `v8` `-26.5%` (`22.155 -> 16.280 ms`), giving `12`-vCPU L4 **`7,625 tr/s` (`1.46×` Mac, `~3.8k` vs A100 `~2.1k tr/s` per unit-hour)**; `S1` L4 runs closed `KILL` (`-12.5` / `-14.9` mass).
- **`UPLOAD_CAP` staging & disk audit:** In P1-04 (`28` files, `74,138 B` in `113.760 s`), **`22` tiny sidecars (`1.6%` of bytes) took `80.6%` (`91.679 s`)** of transfer time alongside slow kernel/control RPCs (`0` PUT failures), fixed by single-tar `metadata_archive.py` and `STAGING_REPAIR_20261024.md`. Disk audit flagged `34` copies of one `4.06 GB` demo bundle (`~138 GB` of `249.27 GiB`).

### Why
Host Python/RPC overhead—single-thread replay gather, CPU `v8` wrappers, and per-file Colab kernel/Contents RPCs after `google-colab-cli 0.7.4` dropped its keep-alive daemon—dominated Colab runtimes rather than GPU FLOPs.

### What to reuse / what not to repeat
- Dispatch all Colab work via `research/colab_cli/` + `tools/ops/ready_queue.py` (`ACCOUNT_ASSIGNMENT_CAP = 6`, `L4 <= 3`), using L4 (`~3.8k tr/s` per unit-h) first for `12`-vCPU jobs and tar-bundling small sidecars (`metadata_archive.py`).
- Keep strict release gates on Mac (`1/54` x86 float32 tie flips), and never upload dozens of tiny metadata sidecars individually or retain duplicate `4.06 GB` staging bundles.

### Full record
Original: `git show 90b3db94:docs/research/history/colab-ops.md` | Primary sources: [colab-cli.md](../../colab-cli.md), [QUALIFICATION.md](../../../research/colab_cli/QUALIFICATION.md), [README.md](../../../tools/ops/README.md)

<a id="agent-skills"></a>
## Codex and Claude Operational & Research Skills (`agent-skills`)

**Dates:** 2026-10-18
**Verdict:** Active repository backup mirror (`18` kept files = `169,999` bytes under `docs/codex-skills/`) of procedural `~/.codex/skills/snake-dqn-*` runbooks, with campaign-specific budgets (`Q0/Q1`), legacy queues (`colab_queue`), and pre-`2026-10-30` ceremony superseded by `AGENTS.md` and `docs/STATUS.md`.

### What was tried
- Built and reviewed (`2026-10-18` Fable + Claude pass) a `docs/codex-skills/` mirror (`README.md` + `17` `SKILL.md`s; `snake-dqn-colab-jobs` tracked under `colab-ops`) synced from `~/.codex/skills/snake-dqn-*`.
- Codified codebase, `58`-D state / `6`-action mechanics, `GameState` / `BatchSim` / `GridBatchSim` parity, Mac-vs-x86 float32 tie divergence (`~1` in `50` H10000 episodes), and immutable SHA-256 governance docs.
- Codified M5 Pro (`18` cores, `20`-core GPU, `64` GB) thread caps, slot locks, AC/`caffeinate -dims` rules, served-checkpoint pins (`b8-s3`, `frp3-s12`, `champion`), `ego2s` fps ceilings, sequential OBF/Phase R gates, and throughput baselines.

### What happened
- **Durable operational baselines (`ACTIVE`):** Preserves M5 Pro single-thread env exports (`OMP`/`OPENBLAS`/`MKL`/`NUMBA`/`VECLIB=1`), `2`-slot strict / `3`-slot Tier-1 CPU locks, and `ego2s` speed ceilings (`24` fps for `<= 6` driven AI, `12` fps for `7–12`).
- **Gate & throughput references:** Codifies survival band `v2` (`0.05` pooled NI, `0.075` per-mix, `0.30` floor) and reference speeds (`~5.2–5.5k tr/s` at `~21 upd/s` Mac MPS, `78–83 upd/s` isolated / `~62 upd/s` = `3.0x` full A100, `14.9 -> 8.4 ms` = `1.74x` RTX `4090`).
- **Aged campaign parameters (`SUPERSEDED`):** Hard-coded `Q0+Q1` unit caps (`<= 150` +`25` units, `Q0` `40`, `Q1` `<= 60`), legacy Drive `research/colab_queue/`, RunPod-for-T34, and pre-`2026-10-30` pre-registration ceremony were overridden by `research/colab_cli` and the Fast-process rule in `AGENTS.md`.

### Why
Procedural skills encoding timeless codebase/hardware contracts (`6` relative actions, parity tests, M5 Pro thread caps, SHA-256 immutability) remain durable across campaigns, whereas skills embedding point-in-time credit budgets or transport queues go stale whenever operating rules advance.

### What to reuse / what not to repeat
- Keep procedural invariants (state/action specs, parity checks, Mac thread caps, serving fallbacks) in `SKILL.md` files, while treating `AGENTS.md` and `docs/STATUS.md` as the sole authority on active campaign rules, budgets, and ceremony.

### Full record
Original: `git show 90b3db94:docs/research/history/agent-skills.md` | Primary sources: [README.md](../../codex-skills/README.md), [SKILL.md](../../codex-skills/snake-dqn-mac-ops/SKILL.md), [SKILL.md](../../codex-skills/snake-dqn-evaluation-gates/SKILL.md)

<a id="t34-colab-lanes"></a>
## T34 Parallel Colab Lanes, 1,160-Unit Compute Portfolio, and Speed-vs-Constraints Rulings

**Dates:** 2026-10-20
**Verdict:** Three independent judge rounds ratified the gated `L0–L9 + Reserve` Colab portfolio (`~1,002 u` GO-branch ceiling + `~90 u` `L0`), bound Round 0 routing to `$E/amendment-boost-floor/decision.json`, approved the split-stream Claim A (`71de7645`) and straggler-charged `Round 0-D` (`9730fed2`) amendments, and launched the `S0–S4` parallel CPU speed program (`DO NOT STOP` active runs).

### What was tried
- Audited `A5 #2` live L4 throughput (`2` workers × `5` threads, `256` roots) vs `W1` `12`-process fanout (`1` world/process) to price true teacher cost across `~1,154–1,160` prepaid Colab Pro+ units (`1,153.84 u` at `19:18 PT`).
- Allocated gated Colab lanes (`L0–L9 + Reserve`), adjudicated which Round 0 `decision.json` governed `SUCCESS_CONTRACT.md` §5, and reviewed pre-data fixes for Claim A (`L1 D0` / `L2 R1` / `L3`) and `Round 0-D` (`L7` straggler makespan charging).
- Evaluated whether to pause T34 for a GPU simulator rewrite or relax parity/hashing/replay constraints (`t34-speed-vs-constraints`).

### What happened
- **Teacher placement & unit cost:** `A5 #2` stalled at `~30–31` row-steps/s/lane (`0/256` worlds in `1,500 s`); after the `1.7139×` digest fix (`6.1605 → 3.5945 ms`) and `12` 1-thread `W1` processes/L4, `A5` hit **`57.2` and `54.4` worlds/lane-h** (`31–36` row-steps/s per L4 vCPU vs `208` on `1` Mac M5 Pro core = `1/6` Mac core, **`~0.036 u/teacher world`** at `1.54 u/h`).
- **Gated `L0–L9` portfolio:** Ordered `L1 D0` (`12 u`, split-stream `128` decisive + `896` anchor batch after v1 flipped `0.4%` of `225` rescues) and `L3` (`120 u`) ahead of bulk labels (`L2` `60 u` for `800 + 200` worlds, `L4` `150 u`, `L5` `280 u`, `L6` `80 u`, `L7` `120 u + 200 u`, `L8` `100 u`, `L9` `100 u`, `Reserve` `100 u`; expected `~500 u` on `R0 GO`, `~195 u` if premises fail).
- **Exclusions & Round 0 routing:** Excluded `T35` (`KILL`), `A100/H100/G4`, and 5-seed checkpoint studies, and bound Round 0 routing exclusively to `$E/amendment-boost-floor/decision.json`.
- **Speed vs constraints (`DO NOT STOP`):** Kept active T34 runs unchanged, rejected GPU sim rewrites (sim was `5.7%` Mac / `12.3%` x86 step time vs `35.5%` digest + `22.2%` `v8` = `57.7%`), and launched `S0` baseline, `S1` flat `full_digest v4`, `S2` `v8_nb` (`nogil`), and `S4` multi-session Mac pacing design.

### Why
Multi-threaded `256`-root workers contended on host CPU and cold-fill latency (`1` L4 vCPU `≈ 1/6` M5 Pro core), `full_digest` (`35.5%`) is active controller logic in `RollingShadow` rather than a removable receipt, and `24 fps` paced Mac runs sleep `~75%` of each tick (wall-clock bound, not engine bound).

### What to reuse / what not to repeat
- Price L4 vCPUs at `~1/6` of a Mac M5 Pro P-core (`31–36` vs `208` row-steps/s) and charge multi-worker job budgets by wave-quantized straggler makespan (`setup + ceil(n/12)·max(slowest, wave)`).
- Optimize `full_digest` (`S1`) and Numba `v8` (`S2`) with exact transition parity rather than subsampling controller state digests, and never spend unallocated credits to pad killed hypotheses.

### Full record
Original: `git show 90b3db94:docs/research/history/t34-colab-lanes.md` | Primary sources: [2026-10-20_colab_lanes/JUDGMENT.md](../advice/2026-10-20_colab_lanes/JUDGMENT.md), [2026-10-20_speed_vs_constraints/JUDGMENT.md](../advice/2026-10-20_speed_vs_constraints/JUDGMENT.md), [CLAIMA_AMENDMENT_JUDGMENT.md](../advice/2026-10-20_colab_lanes/CLAIMA_AMENDMENT_JUDGMENT.md)

<a id="t34-a100-advice"></a>
## T34 Colab A100/H100 Consultations, Label Ordering Ruling, and Round-20 Parallel A100 Program

**Dates:** 2026-10-20 to 2026-10-22
**Verdict:** Unanimous `NO A100` (`10/10` Astra, `3/3` Fable reviews, `3/3` GPU experts) kept host-bound `teacher_v2` on L4 (`12` vCPUs, `54.4–57.2` worlds/h), `RULING_label_ordering.md` permitted parallel Colab `R1` collection while reserving release sign-off to the owner, and `JUDGMENT_round20.md` diagnosed `E1b`'s `0.217 steps/s` (`4.6 s/tick`) loop and funded `PLAN_A100.md` (`A0`–`A4`).

### What was tried
- Audited whether Colab A100-40GB (`~5.4 u/h`) or A100-80GB (`~7.5 u/h`) beat L4 (`~1.7 u/h`) for the CPU-bound T34 teacher (`1,157` prepaid units) and audited `t34_exit_b8_fit/loss.py` and Path B critical-path timing (`2026-10-20_a100`).
- Adjudicated `SUCCESS_CONTRACT.md` §5 vs `DESIGN_V2_SHADOW.md` on overlapping Colab `R1` collection with Mac Path B, plus §4 release authority (`RULING_label_ordering.md`).
- Audited `E1b`'s `0.217 steps/s` (`4.6 s/tick`) stall in `research/gpu_sim_spike/device_sim.py` and structured `PLAN_A100.md` (`WP0`–`WP6`, `A0`–`A4`) with a 3-tier prediction doctrine and CPU action-tape rescoring (`2026-10-22_a100_program`).

### What happened
- **Oct 20 A100 audit (`NO A100`):** Both L4 and A100 expose `12` vCPUs while `B8` forward+transfer took `7.54%`–`7.81%` of A5 time (`13.3%` on 2-vCPU profile vs `35.4%` digest, `19.9%` `v8`), capping infinite-GPU gain at `1.15x` for `~3.14–3.51x` unit cost. Caught `~11–12` Mac hours `time.sleep`-pacing Path B `bare` (`fps=0`) and missing `tau^2` in `loss.py` (`+0.04686`..`+0.14712` gap).
- **Label ordering & release authority (`APPROVE WITH CONDITIONS A–H`):** Permitted Colab `R1` collection after published Round 0 `GO` in parallel with Path B (fits held), and ruled that **every serving or release change requires the owner's explicit OK**.
- **Round 20 (`FUND PARALLEL A100 LANE`):** Proved `E1b` spent `8,296.32 s` host CPU (`8,294.06 s` loop over `1,800` steps) in a Python `for col in range(n)` loop over `n = 6 * 10,400 = 62,400` corpse slots/tick (`~1.2–1.9M` ops/tick) plus `160` BFS passes in `vector_device.py`.
- **Scale & prediction rules (`PLAN_A100.md`):** Noted `89%` of `Ego2sNet`'s `1,638,263` params sit in two flatten `Linear` layers (`~60k` conv); funded `A0`–`A4` (`G1 >= 2,000 steps/s` at `E=1,024`) with CPU action-tape rescoring (`0.305 s`/H300 branch) and barred learned surrogates from certifying safety (`<= 5%` lost rescues on `H64`-prefix pruning).

### Why
Host-bound teachers (`7.54%`–`13.3%` GPU inference share) gain nothing from A100 over L4 (`12` vCPUs each), while `E1b`'s `4.6 s/tick` was caused by `62,400` eager Python tensor launches per tick on `1` CPU core rather than GPU compute.

### What to reuse / what not to repeat
- Keep CPU-bound jobs on `3` parallel L4 lanes (`~5.1 u/h` combined) rather than `1` A100 (`~5.4–7.5 u/h`), and check `host_launch_cpu_seconds` whenever a PyTorch GPU loop stalls (`62,400`-iteration Python loops).
- Never allow learned surrogates (`~0.74–0.77` H300 AUC) to certify branch safety without an exact prefix (`<= 5%` lost rescues) and CPU action-tape rescoring, and always get explicit owner sign-off before changing served defaults.

### Full record
Original: `git show 90b3db94:docs/research/history/t34-a100-advice.md` | Primary sources: [RULING_label_ordering.md](../advice/2026-10-20_a100/RULING_label_ordering.md), [JUDGMENT_round20.md](../advice/2026-10-22_a100_program/JUDGMENT_round20.md), [PLAN_A100.md](../advice/2026-10-22_a100_program/PLAN_A100.md)

<a id="repo-reorg"></a>
## October 2026 Repository Reorganization & Worktree Hygiene Audit (`repo-reorg` / `org30`)

**Dates:** 2026-10-20 to 2026-11-02
**Verdict:** Closed and superseded (`2026-11-02`): the read-only/additive `2026-10-20` `org30` audit inventoried `145` worktrees (`94` `SAFE-TO-REMOVE` = `12.17 GB` of `25.47 GB`) and `800` markdown files without moving SHA-bound `research/**` or `tests/**` paths, before `argon-docs-prune` consolidated lineages and deleted all `29` `org30` proposal files.

### What was tried
- Ran a read-only inventory (`WORKTREE_INVENTORY_2026-10-20.md` against `origin/main` @ `e301cfeb`) of all git worktrees, local branches (`211` merged, `58` unmerged), and live `tmux` sessions without executing removals.
- Drafted `org30` `reorg_proposals/01`–`30` across `docs/`, `research/`, `tests/`, `web/`, and `README.md`, adding series/era catalogs (`SOLO_FOOD_SERIES.md` for `78` folders, `docs/research/catalog/*`) and maps (`TESTS_MAP.md`, `WEB_MAP.md`, `MAP.md`, `TIMELINE.md`) while rewriting `59` absolute in-repo links across `34` `solo_food_*` READMEs to relative links.
- Replaced the additive `org30` proposals on `2026-11-02` with `argon-docs-prune` (`PLAN.json`), consolidating lineages under `docs/research/history/` and deleting all `29` `repo-reorg` proposal/inventory files.

### What happened
- **Worktree & branch audit (`145` worktrees, `25.47 GB`):** Classified `12` `PROTECTED` (`6.69 GB`, incl. `4.73 GB` primary + Round 0/Path B/T35 frozen worktrees), `15` `ACTIVE` (`2.68 GB`), `24` `NEEDS-OWNER` (`3.94 GB`), and **`94` `SAFE-TO-REMOVE` (`12.17 GB` reclaimable)**.
- **Hash-bound path protection & doc-count outcome:** Confirmed `research/**` and `tests/test_*.py` could not be moved because ratified study receipts hash their exact source closures; because `org30` was strictly additive, it increased markdown file count until `2026-11-02` consolidation deleted its `29` proposal files (`0` `KEEP` files).

### Why
Ratified experiment receipts bind cryptographic hash digests over `research/**` and `tests/test_*.py` file paths, preventing directory moves, while an additive-only doc pass (`org30`) added catalogs without pruning redundant per-run markdown until `argon-docs-prune`.

### What to reuse / what not to repeat
- Never move or rename `research/**` or `tests/test_*.py` paths bound by ratified study receipts, and audit active `tmux`/agent processes before pruning git worktrees.
- Replace superseded per-run notes directly (with git history preserving full originals) rather than piling on additive reorganization proposals.

### Full record
Original: `git show 90b3db94:docs/research/history/repo-reorg.md` | Primary sources: [STATUS.md](../../STATUS.md), [TIMELINE.md](../../TIMELINE.md), [GLOSSARY.md](../GLOSSARY.md)
