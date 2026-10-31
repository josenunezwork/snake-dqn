# Pre-Data Judge Re-Review Request for Astra — Watch6 V3 Confirmation (Rev 2)

- **Study:** T34 Watch6 exact-stop efficacy ceiling V3 (`research/watch6_search_efficacy_v3`)
- **Lane:** Confirmation (`OPUS_META.md` §2 item 1 & §3 row 1; owner authorized 2026-10-31)
- **Implementer:** Argon (`snake-dqn-argon-v3-confirm`)
- **Independent Judge:** Astra (responding to `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/watch6-v3-ruling/RULING_V3.md`)
- **Pre-registration under review:** `research/watch6_search_efficacy_v3/REGISTRATION_V3.md`
- **Data status:** **Zero V3 scientific data exist.** Live dispatch remains mechanically blocked until your approved ruling (`Verdict: APPROVE` or `Verdict: APPROVE-WITH-CONDITIONS`) is bound via `ratify.py`. No `exports/` or `pair-*.json` files were opened.

---

## 1. Science & Primary Seed Bank — 100% Unchanged

Per your confirmation in `RULING_V3.md` (`lines 9–11`), none of the scientific or primary-bank specifications changed:
- Same fresh 96-seed primary allocation `primary_seed(i) = uint32_be(SHA256("t34-watch6-exact-stop-efficacy-20261031-v3/primary/{i}")[:4])` (`i in 0..95`, `24` shards × `4` pairs on L4 `W4`), `H = 14400`, alternating arm order, `df = 95`.
- Same bare B8 s3 vs served exact-stop (`L >= 600`, `H = 300`, `DenseTrigger(600, 5)`), 5× `frp3-s12 + v8(lambda=8)` opponents, default `GameSession` Watch6 mechanics v2 + per-frame `GridBatchSim` mirror, and identical co-primary pass bars delegated to unchanged `research.watch6_search_efficacy.analyze.decide`.

---

## 2. Exact Changes Per `RULING_V3.md` Point

1. **Ruling Point 1 — Real `supervisor.py` positive pre-start proof & traceback-verified external-transport classification (`admission.py`, `recover.py`):**
   - Replaced nonexistent `telemetry-*.json` scanning and bare `not hardware.json` checks with `is_prestart_transport_failure(receipt, output_dir)` grounded in `research/colab_cli/supervisor.py` (line-1904 `ready.json` / line-1905 `hardware.json` / line-1982 `poll-*.json`):
     - Presence of `hardware.json`, any `poll-*.json` / `usage-0*.json` / `spend-*.json`, `terminal.json`, or any `exports/*` immediately returns `False`.
     - Missing evidence returns `False`: pre-start requires one of six **positive pre-1904 proofs** on disk (pre-allocation `endpoint is None`; `warmup-measurement.json` `INCOMPLETE`; pre-calibration startup transport failure after `warmup` `PASS` with zero `mkdir`/`source-fetch` ops and verified transport traceback; `rate-calibration.json` `INCOMPLETE`; `upload-measurement.json` with `staging_completed: False`; or post-staging `UPLOAD_CAP` gate at lines 1865–1871 before line 1904).
   - Hardened `is_external_transport_compute_failure(receipt, output_dir)` to require real on-disk `output_dir` with `hardware.json`, clean `poll-*.json` audit (`state.status in {"starting", "running"}`, null `terminal`/`returncode`/`reason`/`error`, zero traceback or exception lines in `job_log_tail` and `supervisor_log_tail`), a failed CLI/RPC/transfer record, **and** explicit host/CLI `Traceback (most recent call last):` evidence (`has_verified_transport_traceback`) with zero application/study stack frames (`src/`, `web/`, `research/watch6_search_efficacy*`, `job.sh`) and a strict `has_non_transport_fault` veto rejecting any error/log text carrying scientific, numerical, mirror, hash, assertion/value, child, wall-cap, or `faulthandler` watchdog faults.
   - Added unit tests reproducing both of your `RULING_V3.md` Point 1 acceptances and proving they are now refused.

2. **Ruling Point 2 — Removed reserve substitution entirely (`protocol.py`, `freshness.py`, `prepare.py`, `admission.py`, `run.py`, `resume.py`, `recover.py`, `analyze.py`, `REGISTRATION_V3.md` §2):**
   - Dropped `reserve_seed`, `reserve_allocation`, `reserve_slots`, `reserve_lineage`, and `seed_kind == "reserve"` across the entire package; any reserve field or non-primary scientific `seed_kind` is rejected fail-closed.
   - Retained the fixed prospective 96-seed primary bank (`primary/{0..95}`) with unlimited positively-proven pre-start retries (`prestart_retries_used >= 0`) plus capped partial-shard resume (`MAX_COMPUTE_COMPLETIONS_PER_SHARD = 2`, i.e., at most 3 compute-started attempts per shard; exceeding 2 failed compute-started attempts closes outcome-blind as `INVALID_OR_INCOMPLETE`), justified in `REGISTRATION_V3.md` §2.

3. **Ruling Point 3 — Staging serialization fails CLOSED (`dispatch.py`):**
   - Removed the 600-second `STAGING_STALE_MANIFEST_SECONDS` bypass; old or corrupt `ops/manifests/*.json` entries never bypass serialization, and exited logs whose `output_dir` lacks `upload-measurement.json`, stop-verified `receipt.json`, or `LIVE` `unavailable.json` fail closed immediately.
   - Both `wait_other_jobs_staged` and `wait_own_job_staged` now fail **closed** on expiry: they write durable `<ops_dir>/watch6-staging-reconcile.json` and raise `RuntimeError("REFUSE: ...")` (`exit 2`), never calling `tools/ops/dispatch.sh` on prior-job expiry and blocking all subsequent dispatches on own-job expiry until reconciled under `watch6-staging.lock` via `--reconcile-staging`.

4. **Ruling Point 4 — Complete pre-outcome failure, lineage, stop & hash audit in `final_look.py`:**
   - Before opening any `pair-XX.json` outcome payload, `run_final_look` audits every created attempt directory (including both `<output_dir>` and `<output_dir>-retry-1`, `auto-retry` `RETRY.json` links, and `recover.py` `prior_receipts` links) across all 24 shards / 96 slots:
     - Verifies `stop_verified` (`cleanup_error is None`) and `manifest_sha256` on every runtime plus `>= 1` passing same-pin L4 canary;
     - Rejects any forbidden failure class on **any** historical attempt (even if a later attempt exported all pairs);
     - Verifies a single unbroken linear retry-cap chain `A_1 -> ... -> A_M` per shard rooted at `total_dispatches_for_shard == 1` with `<= 2` failed compute-started attempts;
     - Verifies all `.sha256`/`.complete` sidecars, `.meta.json`, receipt export hashes, zero unrecorded/duplicate files, and `96/96` primary coverage via `resume.build` before reading outcomes.

5. **Ruling Point 5 — Explicit verdict parsing in `ratify.py`:**
   - `parse_ruling_verdict` requires an explicit non-fenced `Verdict: APPROVE` or `Verdict: APPROVE-WITH-CONDITIONS` line and refuses `REJECT`, `NOT_RATIFIED`, `STATUS: NOT_RATIFIED`, missing `Verdict:`, or conflicting verdicts both in `record_ruling` and on re-read in `validate_ruling_receipt` (verified by test directly against your `RULING_V3.md`).

6. **Upload transfer bound (`<= 12` transfers and `<= 90 s`, `research/watch6_search_efficacy_v3/uploads.py`):**
   - Packed `cost-evidence.json` and all `.sha256`/`.complete` sidecars (including checkpoint-part sidecars) into `input-metadata.tar.gz`, leaving only the `8` × `2 MiB` checkpoint `.part` files separate (`11` normal transfers, `12` worst-case fallback transfers `<= 12`, and `~74.8 s <= 90.0 s` under `5 s/RPC + 1 MB/s`).

---

## Binding Pre-Data Question for Astra

Do you approve `research/watch6_search_efficacy_v3/REGISTRATION_V3.md` and the V3 rev 2 implementation (`Verdict: APPROVE`, `STATUS: RATIFIED`) for live execution of the short L4 canary followed by the 24 L4 W4 confirmation shards?
