# Pre-Data Judge Request for Astra — Watch6 V3 Confirmation (`REGISTRATION_V3.md`)

- **Study:** T34 Watch6 exact-stop efficacy ceiling V3 (`research/watch6_search_efficacy_v3`)
- **Lane:** Confirmation (`OPUS_META.md` §2 item 1 & §3 row 1; owner authorized 2026-10-31)
- **Implementer:** Argon (`snake-dqn-argon-v3-confirm`)
- **Independent Judge:** Astra (author never reviews own build; `AGENTS.md` standing rule)
- **Pre-registration under review:** `research/watch6_search_efficacy_v3/REGISTRATION_V3.md`
- **Data status:** **Zero V3 scientific data exist.** Live Colab dispatch is mechanically blocked by `research/watch6_search_efficacy_v3/dispatch.py` until your ratified ruling is bound on the Mac via `python -m research.watch6_search_efficacy_v3.ratify --ruling <PATH_TO_RULING_V3.md>`. Watch6 V2's 88 exported pairs (`exports/` and `pair-*.json`) remain completely unopened and excluded from all V3 code and seed scans.

---

## Summary of What Is Unchanged vs. What Changed

### 1. Science, Controller, Product, and Analysis — 100% Unchanged from V2
V3 imports `research.watch6_search_efficacy.{engine, product, run, analyze, uploads, performance}` directly with **zero edits to any V2 file**:
- `N = 96` complete paired sessions (`df = 95`), `H = 14400` frames (10 min at 24 fps), hero slot 0, alternating arm execution order.
- Bare B8 s3 (`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`) vs served exact-stop (`DenseTrigger(min_length=600, interval=5)`, `H = 300`, `exact_stop.run` + `ceiling._choose`), 5× `frp3-s12 + v8(lambda=8)` (`eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723`), default `GameSession` Watch6 mechanics v2 (`configs/mechanics_v2.yaml`, `SNAKE_WATCH_SOURCE=live`), continuous respawn, live + `GridBatchSim` mirror checked every frame.
- Co-primary intersection-union pass bars:
  1. Candidate minus bare `deaths / 1k` two-sided paired Student-t 95% CI (`df = 95`) wholly `< 0`, AND
  2. Fieller paired two-sided 95% lower limit on pooled full-horizon mass ratio `(candidate - bare) / bare` `>= -0.05`.
- Same verdicts: `GO_EFFICACY_ONLY`, `FAIL_EFFICACY`, `INCONCLUSIVE`, and `INVALID_OR_INCOMPLETE` prior to science.

### 2. Failure-Handling Changes Only (`REGISTRATION_V3.md` §§2–4)
1. **Fresh disjoint deterministic seed banks (§2):**
   - `NAMESPACE = "t34-watch6-exact-stop-efficacy-20261031-v3"`
   - Primary (`i in 0..95`): `uint32_be(SHA256(f"{NAMESPACE}/primary/{i}")[:4])`
   - 1-to-1 reserve (`i in 0..95`): `uint32_be(SHA256(f"{NAMESPACE}/reserve/{i}")[:4])`
   - Audited pre-data (`freshness.py`) for zero internal collisions across all 195 seeds (`96` primary + `96` reserve + `3` fixtures) and zero collisions against Watch6 V1, Watch6 V2, Watch replay banks 01–04 (and 01–32), exploration namespaces (`bl-*`, `t34-*`, `t15-*`, `s1-*`, `q0/q1-*`, `t34-watch6-alllen-screen-20261031*`), committed JSON banks, and historical FRP training exclusions, without opening any `exports/` or `pair-*.json`.
2. **Smaller blast radius (`24` shards × `4` pairs on L4 `W4`, §3.1):**
   - Sized from the same verified L4 W4 bare-loop + 24 H6000 W4 replay sessions (`7557.9914 s/pair`, 1 wave of 4 pairs on `W4`): `base_minutes_per_shard = 209` min, 2× wall `timeout_minutes = 418` min (`6h 58m`).
   - One short same-pin L4 canary (`15` min base / `30` min effective, 3 tiny H12 `fixture-pair-00..02.json` pairs on `W4` + 1 long-body H300 `branch_fixture`, identical uploaded inputs) qualifies all 24 initial shard manifests.
3. **Serialized staging per `RULING_SHARD6` §2 (§3.2):**
   - `research/watch6_search_efficacy_v3/dispatch.sh` holds `<ops_dir>/watch6-staging.lock` silently around live `<operator_root>/tools/ops/dispatch.sh`, waiting for any active staging job in `<ops_dir>/manifests/*.json` to finish staging first and releasing the lock only after the newly allocated runtime writes `upload-measurement.json` (or `receipt.json` / `EXIT=`). At most one runtime stages at a time; already-staged computing shards run concurrently up to `ACCOUNT_ASSIGNMENT_CAP = 6`.
4. **Pre-start zero-export transport retries (§3.3):**
   - Verified pre-start transport failures (`exports == []`, `terminal is None`, `stop_verified is True`, `cleanup_error is None`, or proven unallocated pre-dispatch refusal, with zero child/scientific error) do not consume the 2 compute-started completion attempts and do not trigger reserve seeds.
   - Bounded by sane caps: `MAX_PRESTART_RETRIES_PER_SHARD = 6`, `MAX_PRESTART_RETRIES_STUDY = 72`, and `MAX_TOTAL_DISPATCHES_PER_SHARD = 10`.
5. **Partial-shard resume from verified pairs (§3.4):**
   - Mid-compute external transport/VM loss (`stop_verified is True`, `cleanup_error is None`, zero child/scientific fault) seals already-exported `pair-XX.json` + `.meta.json` units into a Mac-verified outcome-blind ledger (`skip_verified: true`) and resumes only unverified slots of that shard for up to `MAX_COMPUTE_COMPLETIONS_PER_SHARD = 2` (3 compute-started primary attempts total).
6. **Strict MCAR 1-to-1 reserve seeds (§3.5):**
   - Slot `i` switches to `reserve_seed(i)` **only** if slot `i` has **zero** exported pair payload after **3 compute-started primary attempts** on its shard, where **all 3** failed due to external transport/VM infrastructure loss (`RPC_EXEC_POLL_TIMEOUT`, `TRANSPORT_TIMEOUT`, `RUNTIME_LOST`, or external `SUPERVISOR` transport disconnect with `terminal is None`, `stop_verified is True`, `cleanup_error is None`).
   - **Explicitly excluded (fail closed to `INVALID_OR_INCOMPLETE`):** any `CHILD_NONZERO_EXIT`, mirror/Q/action fault, **shard 418-min wall-clock timeout**, or **900s `faulthandler` watchdog timeout** (preventing survival/length-correlated censoring).
   - **Capped at:** `<= 1` reserve seed per slot (1 compute-started reserve attempt) and `<= 16` reserve substitutions study-wide across `N = 96`.

---

## Binding Pre-Data Questions for Astra

Please review `research/watch6_search_efficacy_v3/REGISTRATION_V3.md` and the V3 package implementation on `main` and issue a binding pre-data ruling (`RULING_V3.md`):

1. **Ratification of V3 Confirmation Pre-Registration:** Do you ratify `REGISTRATION_V3.md` pre-data (`STATUS: RATIFIED`) for live execution of the short L4 canary followed by the 24 L4 W4 confirmation shards?
2. **MCAR Reserve-Seed Boundary (§3.5):** Confirm that the 1-to-1 slot-paired reserve rule (strictly after 3 external-transport failed compute attempts with zero export for slot `i`, excluding child faults and excluding wall-clock / 900s-watchdog timeouts, capped at `<= 1` per slot and `<= 16/96` study-wide) preserves unbiased `df = 95` paired inference without multiplicity or censoring adjustment.
3. **Pre-Start Retry & Serialized Staging Envelope (§§3.2–3.4):** Confirm that separating pre-start zero-export transport retries (`<= 6/shard`, `<= 72` study-wide, `<= 10` total dispatches/shard) from mid-compute resume completions (`<= 2/shard`) under `watch6-staging.lock` serialization satisfies confirmation-lane operational integrity.

*(Note per `OPUS_META.md` §4: V2's 88 sealed pairs remain untouched on disk; any separate ruling on unsealing V2 for descriptive/exploration diagnostics only is deferred until after V3 is ratified and frozen/dispatched.)*
