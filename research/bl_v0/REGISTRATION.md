# V0 Mac latency ladder — frozen before V0 data

Authorized by the owner's Sol-K implementation request (2026-10-21) and
round-13 JUDGMENT §5. This is a research feasibility probe, not release authority.
Commit/publish this file before the target-platform smoke or ladder. Preserve
these bytes; corrections require a prospective sibling amendment.

## Design and caps

This Mac, CPU FP32, W1, one thread, ExactT3Operator with native_mode="kernel".
Nine unchanged Y1 fixtures (RULER_FIXTURES_08.md), each 30 batches per cell.
Alternate C1, C2, C3, C4 within every fixture/repeat; C1 supplies that repeat's
cached exact H300 branch outcomes. First sample cold; remaining 29 warm, median
and nearest-rank p99. frozen-800 = doom, frozen-1500 = cap fixture.

- C1: unchanged kernel H300, all legal actions.
- C2: H300 bare first, alternatives descending Q then lowest index; stop at
  first survivor; all-die evaluates all and uses unchanged T3 score/tie rule.
- C3: exact H64 prefixes, all legal actions; oracle outcomes from C1 plus
  exactly one CPU FP32 dummy leaf call, 128→64→64→6 (12,806 parameters), batch
  size = legal count, real leaf featurisation charged.
- C4: C3 with the same stop order applied to oracle H300 outcomes; unevaluated
  branches explicitly NOT_EVALUATED. Dummy call still has legal-count shape.

No learned model, H32 cell, Colab call, efficacy bank, controller build, serving
change or real timing by Sol-K. Claude alone operates the real ladder, after
K-1, Fri 2026-10-23 22:00–23:00 PT or Sat 2026-10-24 08:00–10:00 PT. First cap
wins: end of chosen window or 3,600 leased seconds. Never K-1 Thu 20:00–22:00
or X5 Fri 09:00–11:00. No process/daemon killing or automatic retry.

Smoke only: nice -n 10, one thread, one fixture × one repeat × four cells,
≤180 seconds, actual entrypoint, Mac lease, guards/load logged; PRECHECK_ONLY,
never a latency/speed admission. Smoke remains excluded from protected windows.

## Frozen kill rules and precedence

1. Missing/corrupt identity, attestation, nonfinite timers, C2/C3/C4 chosen
   action different from C1, or changed source/fixture bytes ⇒ INVALID, no
   scientific claim. C2 mismatch specifically invalidates C2; preserve attempt.
   Required per batch: branch kernel_rows > 0 and stock_rows == 0, with fallback
   and noncandidate rows zero. Keep parent/root stock rows separate.
2. Missing any of nine ×30×4 cells or cap/window expiry ⇒ INCOMPLETE, never a
   smaller PASS. All failed/stopped artifacts remain create-only.
3. C3 warm p99 >1.458 s on frozen-1500 OR frozen-800 ⇒
   LEARNED_CUTOFF_CLOSED, no V1.
4. Literal §5(c): C3 warm median <2× C2 warm median on frozen-800 ⇒ NO_CONSUMER,
   no V1. Equality does not trigger this strict inequality.
5. Otherwise PASS admits only V1 registration; NOT_READY(L2′) persists. Report
   C3/C4 doom-root ≤5 s burst reading descriptively, never as a substituted bar.

Two upstream contradictions are disclosed before data. The summary describes
the consumer bar as C2/C3 <2, whereas the §5 table literally writes C3 <2*C2;
we freeze the latter rather than silently reversing it. Scripted fixtures have
zero eligible Vector61/v8 rows, so truthful kernel_rows=0 cannot satisfy the
blanket positive-row requirement. Retain those rows and mark their attestation
INVALID; without a prospective ruling the full ladder cannot PASS. Never add
unrelated dummy v8 work or invent execution counters to clear this requirement.

## Premise register (round-12 §4.2)

| Claim | Receipt | Platform / exact path | Falsifier |
|---|---|---|---|
| H64 + leaf fits dense cap | exp/bl-v0/timing.json, NOT_READY(V0) | Mac / kernel prefix64_oracle | V0 C3 p99 |
| Exact stop saves about 4× on bare-survives roots | same, NOT_READY(V0) | Mac / kernel stop | V0 C1/C2 |
| Kernel hook ≥5× | exp/bl-k1/timing.json, NOT_READY(K-1) | Mac / kernel | K-1 frozen-1500 opponent_v8 |
| Exact search frozen/mixed value ≥+40 | NOT_READY(L2′) | paired fresh exact search | L2′; inherited by V1 |
| Real pipeline executes | exp/bl-v0/smoke-*/smoke.json, NOT_READY(canary) | Mac / same entrypoint and four cells | bounded precheck ≤180 s, 0 u |

Forecast is judgment-derived, unmeasured: 35–50 minutes for 1,080 batches,
hard stop 60 minutes. Builder 0.5–1 day; independent semantics review 0.25 day;
0 Colab units, 0 upload bytes. A smoke duration is not a throughput forecast.

## Evidence and exclusion contract

Atomic exclusive exp/mac-lease.json carries owner, purpose, until and lease_id;
refuse any existing lease (no stale stealing). Record lease id, idle.json before
work, power/lid/thermal guard, process/load snapshots between batches, source
and fixture hashes, argv/platform/thread settings, per-branch frames/evaluated/
skipped, chosen action, per-batch native counters, stage_seconds and leaf wall.
No native speed claim without executed-path counters. Timers are exclusive;
whole wall includes residual Python work. No smaller-N or smoke PASS.

Do not open the sealed t34-round0-study-01 artifact directory. Do not edit
research/t34_round0*, research/t34_pathb, research/t34_exit_b8*, or
research/t34_round0d. K-1 and Sol-S's ruler implementation remain untouched.
