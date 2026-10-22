RULING: (1) YES — DENSITY-2 may be COMPLETED by one outcome-blind completion job that runs only the 85 missing pre-registered episodes (C-P1 all 72 + B70p/mixed indices 24–35, same namespace/seeds/commit f1904777/B8 bytes/rules) plus a 12-episode A identity-replay, merged on the Mac with the 408 verified exports into the original single look; the JUDGMENT_DENSITY2_CAP "terminal" clause is lifted because its own condition (a diagnosed transport path) is now met by TELEMETRY_STALL_20261022; at most two completion attempts, each gated on the requalified supervisor; the look is taken on whatever is verified after that. (2) YES — "idempotent units exported as completed + verified-ledger resume-skip on retry" becomes a standing requirement for every Colab package, development and registered, with outcome-blind, pre-sized, pre-registered retries and a Mac-side aggregator as the only look; it adds no look to any one-look or sequential design. (3) P1: periodic full snapshots already exist (checkpoint_every_seconds = 1800, exported with sidecars); set 900 s for the ≤ 75-min Y3 source, keep 1800 s for tranches, label them OPERATIONAL/non-scoring, resume only from the latest verified one; Y3 as registered covers the capture code but NOT the periodic kind — add one periodic-snapshot replay to the Y3 replacement job (30/60 min, 1.93 u; Y3 envelope v2 12.00 → 12.75 u) before any tranche may resume from one.

# Resume / recoverability policy — DENSITY-2 completion, standing requirement, P1 snapshots

Judge: independent Fable (resume-policy judge), 2026-10-22, read-only. Inputs read in full: `JUDGMENT_SPARE_SLOT.md`,
`exp/bl-density/JUDGMENT_DENSITY2_CAP.md`, `exp/bl-p1/JUDGMENT_Y3_ENVELOPE.md`, the DENSITY-2 pack
`exp/bl-density/density2-20261022-gatefix-f1904777/` (`production-output-02/{receipt,manifest}.json`, polls 0–15, the
export tree listing and ONE episode export's key names — no outcome field of any episode was read), the telemetry note
`snake-dqn-telemetry/research/colab_cli/TELEMETRY_STALL_20261022.md`, `research/bl_density/{runner,analysis,verify}.py`,
`research/bl_ppo/{run,p1_runtime,REGISTRATION_P1}.*`. `t34-round0-study-01/` not opened. No jobs, no Colab calls, no commits.

## 0. Facts (verified)

- Production-02 remote terminal: `status succeeded, returncode 0`, 493 exports in the saved terminal inventory
  (`poll-000015.json`), progress `492/492` at producer-elapsed 4,031 s. The Mac broker then spent ~90 min re-exporting
  already-present files and force-renewing, hit the 163-min session deadline, receipt `INCOMPLETE /
  RPC_EXEC_POLL_TIMEOUT / Refused('deadline reached during telemetry recovery')`, `stop_verified true`. Balance delta
  1124.37 → 1120.33 = 4.04 u (the whole effective cap was consumed by transport, not by science).
- Local verified exports: **408 / 492 episodes** (+ `.sha256` + `.complete`, sizes and hashes match the terminal
  inventory). Missing, exactly: **C-P1 frozen/mixed/scripted 0/24 each (72)** and **B70p mixed 24/36 (12)**, plus
  `density/summary.json`. The missing set is the alphabetical tail of the inventory — a transport-order artifact,
  independent of any outcome.
- The episode export carries deterministic gameplay fields (`seed, plan, construction, decisions, lengths, sampled,
  death_cause, death_decision, h300_*`) and timing fields (`*_seconds`, `cpu_seconds_by_stage`). `verify.semantic`
  already defines the timing-free canonical form used for Mac↔x86 identity. Byte hashes of the 85 lost files are known
  (inventory) but can never be reproduced (timings) — they are not an identity check for re-computed units.
- Seeds are `sha256(namespace, arm, mix, index)` (`runner._seed`); re-running index k in the same namespace yields
  the same seed and plan. The runner writes each episode export immediately on completion (`export_json` in the
  pool loop) — units are already idempotent and completion-marked; what is missing is a skip-ledger input
  (`--out` must be a fresh directory; there is no "skip verified" argument) and a Mac-side merge/aggregate entrypoint
  (`summarize`/`finalize_cpu` run only inside the remote runner).
- The CAP judgment's terminal clause: "no further attempt or envelope change may be made without a new judgment that
  first diagnoses the transport path." TELEMETRY_STALL_20261022 diagnoses it (broker amplification: re-export of
  present files + forced full-state renewals + 1,224 small GETs) and fixes it (verified-export ledger, compact polls,
  batched ZIP transfer, terminal evidence persisted before download). Requalification on the new transport tree is
  pending.
- P1: `run.py` already exports a full `capture_full_snapshot` every `checkpoint_every` seconds (`snapshot-NNNNN.pt`,
  ≤ 160 MiB, observed ≈ 43 MB) into the export tree, via `export_torch` with sidecars, so the supervisor downloads them
  while the job runs. Default 1800 s (`p1_manifests.py:320`). Source-03 died at < 30 min, before the first one. Y3 as
  registered restores the **pause-boundary** snapshot (`boundary.kind = "exact"`), not a periodic one.

## 1. DENSITY-2 completion — ruling and conditions

**Why it is lawful.** N (492) was fixed pre-data and is unchanged; no outcome has been read (asserted by the operator,
consistent with the evidence trail: no `summary.json` exists locally, and this ruling is written before any merge
script exists). Completing pre-registered units is not an enlargement, not a second look and not a pooling of two
designs: it is the same design finishing its delivery. The spare-slot clause "INCOMPLETE ... is not extended" governs
adding units/episodes beyond the registered allocation after a look; it does not forbid delivering the registered
allocation. The CAP ruling's terminal clause was conditional on a transport diagnosis; the diagnosis exists.
The decision to complete is made **here, now, unconditionally and irrevocably** — it cannot depend on outcomes
because none are known. This ruling therefore also binds: if any person reads an outcome field of the 408 before the
merged look, DENSITY-2 reverts to INCOMPLETE_PEEKED and the completion is void.

**Conditions (all mandatory).**

1. **Transport first.** The telemetry fix merges to main and the Colab CLI lane is re-QUALIFIED on the new transport
   tree (same standard as 2026-10-19 PASS on dab00823); then the Round-12 §4.1 tiny canary of the *skip-ledger path*
   through the real supervisor (≤ 12 episodes, a synthetic ledger that skips some of them, ≤ 0.5 u / ≤ 20 min;
   PASS = skipped units not recomputed, non-skipped exports recovered + hash-verified). No completion dispatch before
   both.
2. **Package (Sol, ≤ 0.5 day, Codex-V review, pre-data `METRICS_AMENDMENT_05.md`).**
   - `runner.py --skip-ledger <json>`: ledger = list of `{episode_id, sha256, size}` of verified exports; the runner
     refuses a ledger whose namespace ≠ run namespace, refuses any ledger entry not in the frozen allocation, and
     plans exactly the allocation minus the ledger (seeds derive unchanged; test: planned seeds for index k equal
     the original run's seeds for index k; test: 85 jobs planned from the real 408-entry ledger).
   - `--identity-replay`: additionally runs the 12 A episodes with index 0–3 in each mix (chosen by index, written
     here, not by outcome), exported under `episodes-replay/`, never into `episodes/`.
   - `analysis.py merge` (Mac): reads verified exports from N output trees, refuses duplicates of an `episode_id`,
     refuses any record outside the namespace, checks the replay set against the originals with `verify.semantic`
     equality (12/12 required, else `IDENTITY_FAIL` and the completion is void), computes the CPU calibration ratio
     r = Σcpu(replay)/Σcpu(original) over the 12, then runs `summarize(492 records)` + `finalize_cpu` and writes the
     one `summary.json`. Overhead: the original run's parent overhead is lost with its summary; charge the completion
     run's measured per-attempted-episode overhead uniformly to all 492 (equal per-episode charge; it cannot favour an
     arm). State this in the amendment.
   - Session-confound rule (pre-data, because all C-P1 and 12 B70p/mixed episodes come from a different L4 session
     than every A episode): the registered per-X×mix rule is applied on raw CPU exactly as registered. If r ∉
     [0.80, 1.25], the rule is additionally evaluated with the completion run's CPU rescaled by 1/r; a cell whose
     two evaluations disagree is `INSUFFICIENT_SESSION` (may be REJECTED by the raw rule; can never be ADMITTED).
     If r ∈ [0.80, 1.25] the raw rule stands alone. The ratio and both evaluations are reported in every case.
3. **Manifest.** New immutable preparation (`completion-01`), commit f1904777 for the science (+ the requalified
   supervisor), same `NAMESPACE`, same allocation table, `--workers 10`, `skip_ledger_sha256` and the 408-entry
   ledger file bound in `input-hashes.json`; `density.phase = "production-completion"`, `one_terminal_look = true`,
   `prior_spend_units` = 4.50 (series to date, conservative). Budget: est 35 min, timeout 90 min, cap 2.5 u, reserve
   0.5 u (85 episodes ≈ 4,400 CPU-s + replay ≈ 700 CPU-s ≈ 22 min at the pilot's effective parallelism; the 2×+ is
   the owner's generosity rule, not a forecast). Nothing else in the manifest may differ from production-02.
4. **No peeking.** Until `merge` has written `summary.json`: no tool, script or person opens an outcome field of any
   episode export; the ledger builder reads only `path/size/sha256` (and must be that narrow in code). The remote
   `summary.json` of the completion job (covering its 97 episodes) is **not** a look and is never read; it is kept
   sealed next to the receipt.
5. **If more units are lost.** One further completion attempt (`completion-02`, same manifest shape, ledger grown by
   whatever `completion-01` verified) is pre-authorized without a new judge round, mechanically, on a verified
   mechanical class (`stop_verified true`, no outcome read). After that DENSITY-2 is terminal: `merge` runs on
   whatever is verified; a cell short of its N is reported with `attempted`/`complete` counts and judged by the
   registered rule (which may yield INSUFFICIENT); no third attempt, no re-sizing, no pooling with the pilot.
   Series worst case: 4.50 + 2 × 2.5 = 9.5 u.
6. **Unchanged.** ≤ 180-min wall cap per job, no third enlargement, pilot never pooled, served hero B8 s3 untouched,
   release OK stays with the owner. The 408 files and `production-output-02/receipt.json` stay as they are (INCOMPLETE
   receipt, immutable); the completed study is a new artifact (`completion/summary.json`) that cites both receipts.

The L4 native ruler main-02 (282/562) is in the same situation; §2 applies to it, but its completion needs its own
short pre-data note by the preparer (its endpoints are timing measurements, so the session-confound rule needs its own
form) — not ruled here.

## 2. Standing requirement — idempotent completed units + verified-ledger resume-skip

**Applies to every Colab package, development and registered, from now on; no package is admitted without it.**

Required properties (verified by offline tests before the Mac preflight):
- Work decomposes into units keyed by a deterministic id derived from (namespace, pre-registered allocation index);
  each unit's export is immutable, written atomically on completion with `.sha256` and `.complete` published last;
  partial units are never exported as complete. (Density already does this per episode; P1's unit is the periodic
  snapshot — see §3.)
- The job takes a **verified ledger** (ids + hashes of units already verified on the Mac) and skips exactly those;
  it refuses a ledger from another namespace/allocation and never recomputes a verified unit into the result set.
- Aggregation (the "look") runs on the Mac over the union of verified units from all attempts, from a registered
  entrypoint that refuses duplicate ids and foreign namespaces. The remote per-attempt summary is diagnostic only
  and sealed, never the look, for any job that could be retried.
- The retry decision is **mechanical and outcome-blind**: pre-registered trigger (verified mechanical failure class,
  `stop_verified`, units missing), pre-registered maximum attempts (owner's 2× rule ⇒ one original + two completion
  attempts), pre-registered per-attempt caps, and the ledger builder reads only path/size/hash. Any retry that
  depends on a human having looked at a unit's content is void.
- Each attempt: fresh manifest, fresh output dir, own receipt; prior receipts immutable; INCOMPLETE receipts retain
  their verified-export records (the telemetry fix already does this).

**Interaction with one-look and sequential rules.** A ledger-skip completion adds no look: the sample size is fixed
pre-data, the decision to continue is outcome-independent, and the aggregate is computed once, at the end, on the
registered N. It is not an extension, interim analysis or optional stopping. The only things that would make it one
are (a) choosing *whether* to complete after seeing outcomes, (b) choosing *which* units to complete other than
"all missing", or (c) changing N — all forbidden above. For sequential designs (P1 tranches), a resume from a verified
unit continues the same trajectory to the same registered pause point; the pause-point snapshot remains the only
scoring artifact, so no additional decision point is created. Time-cap INCOMPLETE of a running job: completing to
the registered N under the pre-registered attempt cap is allowed; raising N or the cap mid-series is not.

## 3. P1 snapshot cadence and Y3 coverage

- Cadence: the knob exists and the snapshots are already exported and downloaded during the run. Set
  `checkpoint_every_seconds = 900` for the Y3 source (≤ 75 min; ensures ≥ 4 operational snapshots) and keep **1800**
  for the 1M qualification and 20M tranches (≈ 43 MB × 24 per 12-h job ≈ 1 GB of transfer; 900 s would double the
  transport load that just failed — not worth it; the owner's "2× generous" is about caps, not about adding risk).
  Label them `OPERATIONAL`: non-scoring, never evaluated, never nominated, never selected among — resume is always
  from the **latest** verified one (rule fixed here; no checkpoint choice). After the registered pause snapshot of
  that job is verified, operational snapshots are retained as records (ledger entries) and their local bytes may be
  pruned except the last two.
- Does Y3 cover it? **Mechanism yes, evidence no.** Periodic and pause snapshots are produced by the same
  `capture_full_snapshot` + `export_torch` path, so a Y3 PASS certifies the transitive-set capture and the exactness of
  restore for that code. But Y3 restores a whole-world pause-boundary snapshot (`boundary.kind = "exact"`); a periodic
  snapshot is taken after an actor update at an arbitrary point in the 128-world collection cycle and carries no
  boundary record. Nothing registered proves that resuming from it reproduces the uninterrupted trajectory. Therefore:
  until proven, a tranche interrupted mid-way restarts from its last **registered pause** snapshot (current rule) —
  operational snapshots give only diagnostics.
- To make Y3 cover it (registration amendment, pre-data, Sol ≤ 0.25 day): the Y3 replacement job performs **two**
  replays — the registered pause-boundary replay (unchanged) and a replay from the **latest operational snapshot**
  taken during the source's continuation, compared against the source's traced action tape over the same window with
  the same exactness criteria (wall-time counters excluded). Both must PASS for Y3 PASS; an operational-replay failure
  alone is `Y3_PASS_BOUNDARY_ONLY` (tranches may run, but resume only from pause snapshots). Price: replacement
  20/40 min → **30/60 min, 1.93 u, reserve 0.39 u**; Y3 envelope v2 **12.00 → 12.75 u** (worst case 12.41 u); the
  mechanical-retry reserve of the Y3 ruling is unchanged. Operational snapshots are then resume-eligible under §2.

## 4. Must not change

Pre-registered thresholds, features, stride, horizon, B8 bytes and amendment SHAs of DENSITY-2; the no-third-
enlargement and no-pilot-pooling rules; P1's 100M target, pause points, no-checkpoint-selection rule and scientific
protocol; L4 1.54 u/h ceiling with zero tolerance; canary identity rule (proof == current == manifest pin);
Claude dispatches, Codex never; owner keeps release authority; the sealed study stays closed.

Not committed by the judge.
