# T34 Watch6 exact-stop efficacy ceiling V3 (confirmation lane, rev 2) — 2026-10-31

Status: PROPOSED_PRE_DATA_PENDING_ASTRA_RULING (`research/watch6_search_efficacy_v3`,
implementer: Argon; independent pre-data judge: Astra). Live dispatch is mechanically
blocked by `research/watch6_search_efficacy_v3/dispatch.py` until Astra's ratified
pre-data ruling receipt (`astra-ruling.json`, binding `REGISTRATION_V3.md` SHA256,
`science` identity, and an explicit `Verdict: APPROVE` or `Verdict: APPROVE-WITH-CONDITIONS`
parsed from Astra's `RULING_V3.md` by `ratify.py`) is recorded on the Mac.

Context: Watch6 V2 (`t34-watch6-exact-stop-efficacy-20261030-v2`) closed as
`INVALID_OR_INCOMPLETE` (`88/96` paired sessions exported; shard 5 lost all three
attempts to pre-start staging/RPC transport failures without starting compute). All 88
V2 pair payloads remain sealed (`exports/` and `pair-*.json` untouched and never opened
by V3). Following Astra's pre-data rejection of V3 rev 1 (`RULING_V3.md`, `Verdict: REJECT`,
`STATUS: NOT_RATIFIED`, zero V3 data run), V3 rev 2 fixes all five implementation-integrity
findings while keeping the identical V2 confirmation science and V3's audited fresh
96-seed primary bank unchanged.

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
  index). One fixed-`N` Mac look after all 96 primary paired units are verified and all
  V3 runtimes satisfy verified stop and complete retry-cap lineage.
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
    mirror-mismatched, forbidden-failure, or retry-cap-violating unit/attempt (precedes
    science; no statistical look is taken).

## 2. Fresh deterministic primary seed bank and removal of reserve substitution

- **Namespace:** `NAMESPACE = "t34-watch6-exact-stop-efficacy-20261031-v3"`.
- **Fixed prospective primary bank (`i in 0..95`, unchanged from V3 rev 1):**
  `primary_seed(i) = uint32_be(SHA256(f"{NAMESPACE}/primary/{i}")[:4])`.
- **No reserve substitution (prospective justification per `RULING_V3.md` §2):**
  Reserve seed replacement has been **dropped entirely**. As Astra's ruling noted,
  execution duration in Watch6 exact-stop depends on hero survival length and eligible
  search-root frequency (`L >= 600`), so even exogenous mid-compute VM/transport drops
  preferentially leave slower, longer-surviving pairs unexported; replacing unfinished
  slots after `k` failed compute attempts selects on duration and breaks MCAR for
  ordinary `df = 95` inference. Retaining the fixed 96-seed primary bank (`seed_kind == "primary"`
  only; any reserve metadata or `seed_kind != "primary"` is rejected by `admission.py`,
  `resume.py`, `analyze.py`, and `final_look.py`) eliminates all duration-selective
  inclusion bias: V3 either observes all 96 prospective primary seeds (`df = 95`) via
  unlimited positively-proven pre-start retries plus capped partial-shard resume, or
  closes outcome-blind as `INVALID_OR_INCOMPLETE` with zero outcome look.
- **Excluded operational fixtures:** `FIXTURE_SEEDS = [34890123, 34890124, 34890125]`,
  used solely in Mac mirror/preflight and the short L4 canary (`fixture-pair-00..02.json`).
- **Pre-data freshness audit (`freshness.py`):** Verifies that all 99 seeds (`96` primary
  + `3` fixtures) are pairwise distinct and disjoint from Watch6 V1, Watch6 V2, Watch
  replay banks 01–32, all exploration/historical namespaces (including
  `t34-watch6-alllen-screen-20261031*`), committed JSON banks (with positive controls),
  visible repo/artifact text, and `>= 7` historical FRP `training_exclusions.json`
  inventories while strictly refusing to open any `**/exports/**` or `pair-*.json`.

## 3. Hardened operational and failure-handling design

### 3.1 Shard layout, empirical L4 W4 sizing, and `<= 12` / `<= 90 s` upload budget

- **Partition & sizing (`24` shards × `4` pairs on L4 `W4`):** Shard `s in 0..23` owns
  the 4 fixed primary slots `{s, s + 24, s + 48, s + 72}` (1 wave on `W4`, `min_vcpu = 12`).
  From `costs.py` (`7557.9914 s/pair`, `1.5x` margin, `20` min setup/export):
  `base_minutes_per_shard = 209` min (`3h 29m`), 2× runaway wall `timeout_minutes = 418`
  min (`6h 58m`). One short same-pin L4 canary (`15` min base / `30` min effective)
  qualifies all 24 initial shard packets via `finalize.py`.
- **Upload budget (`research/watch6_search_efficacy_v3/uploads.py`):** All small inputs
  (`runtime-contract.json`, `verified-ledger.json`, `judge-approval.json`, `mac-mirror.json`,
  `cost-evidence.json`, `release-evidence.tar.gz`) and all `.sha256`/`.complete` sidecars
  (including checkpoint-part sidecars) are bundled into a single manifest-bound
  `input-metadata.tar.gz`, leaving only the `8` × `2 MiB` checkpoint `.part` files
  separate (`9` input uploads + `bootstrap.sh` + `source.py` = **`11` normal transfers**,
  or **`12` transfers** including `source.tar.gz` fallback). `check_uploads` enforces
  offline `<= 4 MiB` per file, **`<= 12` total transfers**, and **`<= 90.0 s` estimated
  transfer time** under the `5.0 s/RPC + 1.0 MB/s` model (`~68.6 s` normal, `~74.8 s`
  worst-case fallback, well below the `150 s` canary transfer cap and `300 s` staging cap).

### 3.2 Fail-CLOSED account-wide serialized staging (`dispatch.py`)

- `research/watch6_search_efficacy_v3/dispatch.sh` holds an exclusive host `flock` on
  `<ops_dir>/watch6-staging.lock` around live `<operator_root>/tools/ops/dispatch.sh`:
  1. **No stale-manifest bypass:** Old or corrupt `<ops_dir>/manifests/*.json` entries are
     **never** ignored after 600 seconds. A manifest is considered finished with staging
     only upon positive on-disk proof: `upload-measurement.json`, a stop-verified (or
     unassigned `endpoint is None`) `receipt.json`, a `LIVE` `colab-allocation-unavailable/v1`
     proof, or `EXIT=` in `ops/logs/<name>.log` when zero output directory was created.
     An exited log whose output directory exists without staging/stop evidence fails
     **closed** immediately.
  2. **Fail-CLOSED wait expiry with durable reconciliation fence:** If
     `wait_other_jobs_staged` expires, `dispatch.py` writes `<ops_dir>/watch6-staging-reconcile.json`
     and exits nonzero (`exit 2`) **without** invoking `tools/ops/dispatch.sh`. If
     `wait_own_job_staged` expires (or own job exits mid-staging without staging/stop
     proof), `dispatch.py` writes `<ops_dir>/watch6-staging-reconcile.json` and exits
     nonzero (`exit 2`). While `watch6-staging-reconcile.json` exists, every subsequent
     `dispatch.py` invocation refuses closed before dispatch until the operator reconciles
     active sessions and runs
     `python -m research.watch6_search_efficacy_v3.dispatch --reconcile-staging --ops-dir <ops_dir>`,
     which verifies under `watch6-staging.lock` that zero manifests remain staging or
     unstopped before clearing the fence.

### 3.3 Positively proven pre-start failures (`is_prestart_transport_failure`)

- Grounded directly in `research/colab_cli/supervisor.py` (where the remote child blocks
  on `ready.json` until the line-1904 `recoverable_rpc(code)` call that immediately writes
  `hardware.json` at line 1905, followed by `poll-000000.json` at line 1982):
- An attempt is classified as **pre-start** iff `status == "INCOMPLETE"`, `exports == []`,
  zero files under `exports/`, `terminal is None`, no `terminal.json`, **no `hardware.json`**,
  **no `poll-*.json`** (and no `telemetry-*.json`, `usage-0*.json`, `spend-*.json`), zero
  scientific/integrity/child/timeout fault text, **AND one of six positive pre-1904
  supervisor proofs** exists in `output_dir`:
  1. Pre-allocation refusal (`endpoint is None`, `stop_verified is False`, `cleanup_error is None`,
     zero transfers/operations, zero measurement files, pre-allocation `ACCOUNTING`/`UPLOAD_CAP`/`SUPERVISOR` refusal);
  2. Warmup incomplete (`stop_verified is True`, `warmup-measurement.json` `status == "INCOMPLETE"`
     before `runtime.py`, no calibration/upload files);
  3. Post-warmup / pre-calibration startup transport failure (`warmup-measurement.json`
     `status == "PASS"`, no `rate-calibration.json` / `upload-measurement.json`, no
     `mkdir`/`source-fetch` control ops, failed transport record + verified transport traceback);
  4. Rate calibration incomplete (`rate-calibration.json` `status == "INCOMPLETE"`, no
     `upload-measurement.json`);
  5. Staging incomplete (`upload-measurement.json` with `staging_completed is False`,
     neither `source-delivery.json` nor `input-hashes.json` written, `UPLOAD_CAP` or
     failed transport record + verified transport traceback);
  6. Post-staging upload-cap gate before line 1904 (`upload-measurement.json`,
     `source-delivery.json`, and `input-hashes.json` present with `failure_class == "UPLOAD_CAP"`
     at lines 1865–1871 before `ready.json` RPC).
- **Missing evidence is NOT pre-start:** Any bare directory lacking `hardware.json` without
  one of Proofs 1–6 (including an ambiguous post-staging non-`UPLOAD_CAP` failure at line
  1904 where `ready.json` may have been written before `hardware.json`) returns `False`.
- Positively proven pre-start failures never executed a single world step and therefore
  allow **unlimited** outcome-blind retries (`prestart_retries_used >= 0` tracked in
  lineage) without consuming the shard's compute-completion cap.

### 3.4 Traceback-verified external-transport compute failures & partial-shard resume

- Once compute has started (`hardware.json` present), an incomplete attempt qualifies for
  outcome-blind partial-shard resume (`is_external_transport_compute_failure`) **only**
  when all of the following hold on disk:
  1. `output_dir` exists on disk with `hardware.json`, `status == "INCOMPLETE"`,
     `stop_verified is True`, `cleanup_error is None`, `terminal is None`, no `terminal.json`,
     and `failure_class in {"RPC_EXEC_POLL_TIMEOUT", "SUPERVISOR", "TRANSPORT_TIMEOUT", "RUNTIME_LOST"}`;
  2. Every `poll-*.json` snapshot in `output_dir` parses cleanly with `terminal is None`,
     `state.status in {"starting", "running"}`, null `returncode`/`reason`/`error`/`unassign_error`,
     and zero `Traceback`, `*Error:`, or scientific/integrity/timeout markers in
     `job_log_tail` and `supervisor_log_tail`;
  3. `receipt` contains at least one failed CLI/RPC/transfer operation record **and**
     explicit host/CLI **transport traceback evidence** (`Traceback (most recent call last):`
     with frames in `colab` / `httpx` / `httpcore` / `urllib` / `http.client` / `ssl` /
     `socket` / `subprocess` / `research/colab_cli`, zero frames in `src/`, `web/`,
     `research/watch6_search_efficacy*`, or `job.sh`, and a terminal transport exception);
  4. **Zero scientific/integrity/timeout fault text** anywhere in `error`, `traceback`,
     `last_traceback_line`, or `poll-*.json` (any `NaN`, non-finite, mirror/hash/digest
     mismatch, `AssertionError`, `ValueError`, child exit, 418-min wall cap, or 900s
     `faulthandler` watchdog immediately vetoes classification).
- **Compute completion cap:** Each shard allows `1` initial compute-started attempt plus
  at most **`MAX_COMPUTE_COMPLETIONS_PER_SHARD = 2`** outcome-blind partial-shard resume
  completions (`3` compute-started attempts total; verified pairs skipped via
  `verified-ledger.json` with `skip_verified: true`). Exceeding 2 failed compute-started
  attempts on an unfinished shard closes V3 outcome-blind as `INVALID_OR_INCOMPLETE`.

## 4. Explicit ruling parser (`ratify.py`) and pre-outcome final-look audit (`final_look.py`)

1. **Explicit verdict parsing (`ratify.py`):** `parse_ruling_verdict` parses non-fenced
   `Verdict:` and `STATUS:` lines in Astra's ruling Markdown. It accepts **only** explicit
   `Verdict: APPROVE` or `Verdict: APPROVE-WITH-CONDITIONS` (with non-contradictory
   `STATUS:`), and refuses `REJECT`, `NOT_RATIFIED`, missing `Verdict:`, or conflicting
   verdicts both when creating `astra-ruling.json` and on re-read inside
   `validate_ruling_receipt` prior to live dispatch and final analysis.
2. **Pre-outcome full-slot integrity & lineage audit (`final_look.py`):** Before opening
   any `pair-XX.json` outcome payload, `run_final_look` inspects every V3 attempt across
   `<ops_dir>/manifests/*.json` (auditing both `<output_dir>` and `<output_dir>-retry-1`,
   `auto-retry` `RETRY.json` links, and `recover.py` `prior_receipts` links) and verifies
   for every shard `0..23` (every slot `0..95`):
   - `stop_verified` (`cleanup_error is None`) and `manifest_sha256` binding on every runtime;
   - At least one passing same-pin L4 canary receipt and zero forbidden failures on any
     canary or main attempt;
   - A single, unbroken linear retry-cap lineage `A_1 -> ... -> A_M` per shard rooted at
     the initial packet (`total_dispatches_for_shard == 1`), with `A_1 .. A_{M-1}` all
     verified `"prestart"` or `"external_transport_compute"`, `A_M` `"pass"`, and failed
     compute-started attempts `<= 2`;
   - Immutable `.sha256`/`.complete` sidecars, `.meta.json` (`seed_kind == "primary"`),
     receipt export-inventory hashes, zero unrecorded/duplicate pair files, and exact
     `96/96` primary slot coverage via `resume.build`.
3. Only after every pre-outcome check passes across all 96 primary slots does
   `final_look.py` open the 96 `pair-XX.json` payloads and run `analyze.decide` once.
