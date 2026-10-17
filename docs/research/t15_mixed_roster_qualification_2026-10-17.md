# T15 mixed Watch roster qualification — 2026-10-17

**Outcome: SERVING_PASS for the corrected evidence assembly — all six checks pass.**
The opt-in implementation is complete. No serving default has changed: the
released checkpoint remains **b8-s3** and the Watch roster remains **all_b8**.
`SNAKE_SERVE_WATCH_ROSTER=mixed` selects an unwrapped B8 hero at slot 0 and
FRP3-s12 + v8 opponents at slots 1 through N−1. Play retains its existing policy
selection and opt-in veto behavior. This qualifies the opt-in serving path;
it is not a release or authorization to change the default.

## Evidence and registration

Authority is [the T15 spec](specs/t15_mixed_roster_qual_spec_2026-10-17.md).
The sibling [protocol](../../research/t15_mixed_roster_20261017/protocol.md)
was committed and pushed at `0853cae6` before any non-smoke episode. Its SHA256 is
`0b882fc2a11e47ce67a0ec541f560d157bd32c6e62c95ce9554683633603428d`.
Product commit: `55860cb9`; frozen producer/audit commit: `ac59916`.
The integrated qualified source is
`9cc59403741e7810c142f92f5bd068a0b63133d6`, including the concurrent T14 CI merge.
The run intent binds 1,099 source/configuration/frontend files, including the
actual built UI assets. Its SHA256 is
`959142e96c53a709c580b5eab6ff362885e439564a57a369f0dd952262d3e3d7`.

Both checkpoint bytes and registry pins were verified:

| Role | Checkpoint | SHA256 |
| --- | --- | --- |
| Bare hero | b8-s3, ego2s-bv8 | `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8` |
| Every opponent | frp3-s12, vector61, exact v8 | `eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723` |

The namespace `t15-mixed-roster-20261017/v1` allocated 33 seeds before gameplay:
25 Watch, 3 parity, 3 latency, 1 controls and 1 UI. Freshness auditing found no
collisions against 1,184,677 registered historical/exclusion seeds. The retained
audit documents its exact scan scope and exclusions; opaque external work is
outside the freshness claim. Smoke allocations are separate and never qualify.

Raw evidence is retained under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t15-mixed-roster-20261017/run-v1`.
The final receipt belongs to sibling `run-v1-assembly`, which preserves all raw
observations and the original invalid UI envelope. Publication bindings and the
evidence archive accompany the final audit in
[`research/t15_mixed_roster_20261017/results/`](../../research/t15_mixed_roster_20261017/results/).
The [receipt](../../research/t15_mixed_roster_20261017/results/receipt.json)
binds **116 files**, SHA256
`8f6711affa4d23ee65fb6c727ef8e82da30f120b1c711460ef4338d362e4ba3f`.
The [independent audit](../../research/t15_mixed_roster_20261017/results/audit.json)
returns `SERVING_PASS`, SHA256
`e20530cb27b6dd5d0e692c37e100a89a508ff72413eec5b676efa184dd49c23b`.
The [summary](../../research/t15_mixed_roster_20261017/results/summary.json),
[assembly provenance](../../research/t15_mixed_roster_20261017/results/assembly-provenance.json),
[correction memo](../../research/t15_mixed_roster_20261017/results/UI_ENVELOPE_CORRECTION.md),
and [publication manifest](../../research/t15_mixed_roster_20261017/results/publication.json)
provide the detailed bindings. The archive includes the served frontend bundle.
Replaying source/registry validation also requires the qualified checkout and
checkpoint files, since the audit intentionally checks those live bytes.
The [archive](../../research/t15_mixed_roster_20261017/results/evidence.tar.gz)
contains 118 files and is 32,054,428 bytes; SHA256
`0e5063d1394e0f4517f846bc2cd60ae7d1427f20d8f80b1339713e7c72baa4a1`.
The [independent review](../../research/t15_mixed_roster_20261017/results/REVIEW.md)
verified all receipt/source hashes, all 114 original-to-assembly file mappings,
the one-field UI correction and raw timing percentiles. Final report review
confirmed the figures, limits and conditional release/rollback wording.

## Qualification checks

| Check | Status | Evidence |
| --- | --- | --- |
| 1. Observation/execution identity | PASS | N6/H5000, N8/H5000 and N12/H10000: all 20,000 world frames and both role streams agree exactly with the eval adapter. |
| 2. Opponent-only veto scope | PASS | All 125,000 continuous Watch frames retain a bare hero and exact v8 wrappers/counters on opponents 1–5, including respawns. |
| 3. Selection, fallback and Play regression | PASS | 32 real-session controls plus the unit matrix; invalid FRP pins/bytes/source restore all-B8 with a visible reason. |
| 4. Latency and server speed ceilings | PASS | Update p95 7.792/9.519/13.141 ms; zero overruns at 12 fps and enforced ceilings 24/12/12 fps. |
| 5. Continuous Watch stability | PASS | 25 fresh N6/H5000 episodes; 125,000 advancing frames, no exceptions or session errors. |
| 6. Live UI | PASS | Observed opponent counter badge, no hero counter, 12 fps default and accepted/enforced 24 fps control cap. Corrected assembly envelope passed the frozen audit; invalid original is retained. |

Parity uses the common `promotion-v2-watch-rect` profile, rowwise forwards and a
harness-local evaluator adapter that applies v8 to opponent slots. The hero is
terminal only in these identity probes to match evaluation. Actual hero packets
are 5,000, 1,823 and 1,169 respectively: **7,992 active hero decisions**, with
explicit inactive entries after terminal death. All 20,000 world frames, active
opponent observations/Q/actions and hero activity records are checked. This is
separate from the actual all-respawn Watch lifecycle used for stability/timing.
No evaluator or feature-source bytes were changed for parity.

The control matrix covers default/all-B8/mixed/unknown roster selection; explicit
B8/FRP3/champion/unknown checkpoints; legacy veto overrides; missing, unfilled and
wrong pins; unavailable/corrupt bytes; wrapper-source refusal; no partial wrapper
survival after refusal; B8 fallback; mode/rebuild/checkpoint replacement; the
three roster ceilings; and existing Play veto eligibility. B8 Train remains
refused. Inspecting a mixed opponent uses that opponent's policy without changing
the fixed B8 roster hero.

## Latency and cadence

All three cases passed the protocol's quiet-machine condition before and after
measurement, with AC power and open lid. Timing waited until a foreign Python
test workload finished by itself; the task did not signal or stop it. Each roster
received 200 warm-up frames and 1,500 measured updates/full ticks, plus separate
fresh sessions with 200 warm-up frames and 120 paced ticks at requested 12 fps
and requested 120 fps. The server capped the latter at 24/12/12 fps for
N6/N8/N12. Every update p95 is below 66.667 ms and all six paced probes have zero
processing overruns. **No ceiling adjustment, retry or timing amendment was needed.**

| Case | Update p50 / p95 / max (ms) | Full tick p50 / p95 / max (ms) |
| --- | ---: | ---: |
| Watch6 | 6.027 / 7.792 / 34.076 | 6.819 / 8.635 / 38.940 |
| Watch8 | 7.504 / 9.519 / 32.845 | 8.253 / 10.356 / 33.809 |
| Watch12 | 11.329 / 13.141 / 36.511 | 12.201 / 14.073 / 37.514 |

| Case | Request → accepted speed (fps) | Processing budget (ms) | Paced tick p95 / max (ms) | Overruns | Achieved cadence (fps) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Watch6 | 12 → 12 | 83.333 | 14.399 / 21.007 | 0/120 | 9.436 |
| Watch6 | 120 → 24 | 41.667 | 14.882 / 16.439 | 0/120 | 16.290 |
| Watch8 | 12 → 12 | 83.333 | 18.633 / 21.296 | 0/120 | 9.425 |
| Watch8 | 120 → 12 | 83.333 | 19.110 / 20.906 | 0/120 | 9.371 |
| Watch12 | 12 → 12 | 83.333 | 24.542 / 65.725 | 0/120 | 8.933 |
| Watch12 | 120 → 12 | 83.333 | 23.774 / 26.121 | 0/120 | 8.967 |

Full ticks include `GameSession.step()` and snapshot processing. Network sending
and client rendering are outside this measurement. Achieved cadence is reported
separately because the server sleeps a full frame interval after processing.
The 65.725 ms maximum paced Watch12 tick remains within its 83.333 ms processing
budget. Zero observed overruns in these finite probes is not a guarantee against
future scheduling spikes.

Prospective update-p95 review thresholds at 1.5× the measured per-roster values
would be **11.688 / 14.279 / 19.712 ms** for Watch6/8/12, in addition to the
66.667 ms qualification bound and the enforced processing budgets. These are
proposed monitoring inputs only; neither they nor a new monitor were installed.

## Continuous mixed Watch6 baseline

The 25 registered episodes produced **125,000 hero decisions, 34 deaths and
124,966 alive frames**. Descriptive death rate: **0.272 per 1,000 decisions**.
The pooled per-decision logical-length histogram gives p50 **248**, p95 **841**,
minimum **1**, maximum **1,153**. The evidence retains the full histogram and
per-episode counts, seeds, quantiles, observation/dispatch rows and source hashes.
The five opponents together made **625,000 decisions** and applied **3,288 vetoes**.

These values describe this mixed N6, all-respawn Watch configuration. They are
neither a new skill/promotion gate nor comparative safety evidence; they do not
establish N8/N12 death baselines. The existing all-B8 release monitor does not
consume mixed-roster episodes. A prospective mixed-only monitoring design could
use twice this death rate (**0.544 per 1,000 decisions**) as a descriptive review
trigger, with an explicitly defined sample window. No monitor or automatic action
is enabled here.

## Live UI evidence and envelope correction

An owned loopback server used the registered UI seed and an isolated scores DB.
The live header showed `opponents v8`, aggregate veto/decision counts and no hero
veto counter. The DOM/accessibility evidence retains IDs #1–#5 and their individual
counts. The normal-width screenshot ellipsizes the long badge; it does not make
every per-opponent count legible. The slider's maximum is 24, the initial speed
is 12 fps, and selecting 2× was observed at 24 fps.

The startup backend record identifies the actual pinned mixed session and has
zero startup counters. The later `/api/state` control payload supports the same
identity with advancing counters; it is not the screenshot's exact frame. Both
screenshots, both DOM snapshots, browser capture metadata and backend records are
retained. The owned tab was closed and server stopped after capture.

The original `records/ui-000.json` has `wall_seconds: null` and remains **INVALID**
and byte-identical. A separate assembly changes only that metadata field to
**38.903733 seconds**, calculated from retained startup and browser-capture UTC
timestamps. It preserves the invalid original, both timestamp sources and the
correction memo. It does not rerun observations, replace source bytes, select new
seeds or relax the audit. This UTC-derived duration is not a latency measurement.
The final verdict applies only to the corrected, independently audited assembly.
The [correction memo](../../research/t15_mixed_roster_20261017/results/UI_ENVELOPE_CORRECTION.md)
pins the original envelope and timestamp sources. The corrected envelope SHA256 is
`eff4bde54ea4e4b7904bd095eb9a4864fde4f236ac934cbaa5eb6f9a3276ad19`.

The independent audit can be replayed without gameplay:

```bash
./venv/bin/python -m research.t15_mixed_roster_20261017.audit \
  --root /Users/josenunez/Projects/ml/snake-dqn-artifacts/t15-mixed-roster-20261017/run-v1-assembly \
  --out /tmp/t15-audit-replay.json
```

Use a new output path: audit outputs are create-only. Expected exit status is 0
with `SERVING_PASS`. This command validates retained evidence; it does not rerun
episodes or authorize a default change.

## Validation and operational boundaries

Preflight passed **413 backend/observation tests, 26 harness tests and 18 frontend
tests**, the TypeScript/Vite production build, Black, isort, Flake8 and whitespace
checks. Independent product/harness review preceded the frozen intent; independent
evidence review confirmed the baseline and reporting distinctions. The earlier
lint failure and smoke-control fixture/expectation failures are preserved alongside
their corrected preflight results and excluded from qualification. All runtime,
producer and audit changes preceded real gameplay.

Everything ran on this Mac using CPU and one torch/BLAS thread per worker, with
at most four owned episode/parity workers. No RunPod work occurred. No tmux
session was opened or controlled and no foreign process was signaled or stopped.
The original `saved_snakes` symlink is preserved. Checkpoint bytes, pins, checked-in
configurations and the serving defaults are unchanged.

## Proposed future change and expected stderr

The one-line future default change is **proposed only; NOT APPLIED**:

```python
WATCH_ROSTER_RELEASED_DEFAULT = "mixed"
```

The implementation currently keeps `WATCH_ROSTER_RELEASED_DEFAULT =
WATCH_ROSTER_ALL_B8`. Explicit opt-in is `SNAKE_SERVE_WATCH_ROSTER=mixed`.
Rollback is to unset that variable or set `all_b8`, then restart the owned serving
instance. The checkpoint default remains b8-s3 in either case.

Expected substantive stderr fields for an opt-in six-snake Watch session are:

```text
served-checkpoint: name=b8-s3 requested=None sha256=3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8 ... reason=None
safety-veto-serving: active=True scope=watch_opponents mode=watch wrapped_ids=[1, 2, 3, 4, 5] ... checkpoint_sha256=eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723 strict_checkpoint_match=True ... variant=v8 variant_requested=v8 wrapper_sources_match=True wrapper_method=free-space-veto/v8-space-and-head(lambda=8.0) reason=None
served-checkpoint: watch_roster=mixed requested=mixed hero=b8-s3 opponent=frp3-s12 opponent_pins=[] reason=None
```

The retained `ui-server.log` contains the complete lines. Scope and wrapped IDs
are authoritative: slot 0 has no wrapper or veto counter. A refused opponent pin
instead logs `watch_roster=all_b8` and its refusal reason, with no partial mixed
roster left active.

## DRAFT release erratum — not a release or default change

> If separately authorized for release, Watch's exhibition roster will consist
> of the released **b8-s3 hero without a serving veto**, facing **frp3-s12
> opponents with the v8 veto**. Opponent veto counters are labeled as opponent
> counters. B8's v8-derived training labels and live input features do not mean
> the B8 hero is wrapped by a serving veto.
>
> T7's claim remains **more mass at equal survival against the served agent,
> with no hero veto**; it is not a claim of greater safety. T15 qualifies this
> serving configuration. It does not isolate the effect of the opponent
> checkpoint from the effect of its wrapper, and provides no new promotion
> evidence. Play, the checkpoint default, training and checkpoint bytes remain
> unchanged. T15 does not introduce a Play or leaderboard-comparability change.
>
> Mixed Watch monitoring must use a separate roster-specific baseline and
> predefined windows. The N6 baseline is 0.272 hero deaths per 1,000 decisions,
> with logical-length p50 248 and p95 841. Timing thresholds must likewise be
> roster-specific and based on the retained quiet-machine measurements, while
> retaining the 66.667 ms qualification bound and enforced speed ceilings.
> These are prospective monitoring inputs; no new monitoring behavior or default
> change is authorized by this qualification. If the future default becomes
> mixed, roll back by explicitly setting `SNAKE_SERVE_WATCH_ROSTER=all_b8` and
> restarting the server; unsetting the variable would then select mixed again.
