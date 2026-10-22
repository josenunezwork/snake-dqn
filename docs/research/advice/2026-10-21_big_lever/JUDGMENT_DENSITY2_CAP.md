RULING: Yes — raise the DENSITY-2 production envelope once, for the one mechanical retry that JUDGMENT_SPARE_SLOT §1 kill-rule (i) already grants: total envelope `PRODUCTION_CAP` 5.0 u → **5.5 u**, reserve unchanged at **1.0 u**, `prior_spend_units` = **0.306 u** (canary 0.127 u + mechanical attempt 0.173 u, both measured as elapsed × priced 1.54 u/h; the balance delta 1124.65 → 1124.50 = 0.15 u is the lower bound and is superseded by the elapsed pricing); hence effective production cap **4.194 u** and priced timeout **163 min** (≥ 2 × 75 min, ≤ 180 min). Worst-case real spend of the whole DENSITY-2 series stays 0.306 + 4.194 = **4.50 u < 5 u** — the original ≤ 5 u ruling is honored in substance; only the nominal envelope (which includes the never-spent 1.0 u reserve) moves. No outcome has been read: the failed attempt has `exports = []`, no episode started, no seed consumed. Nothing else must happen first.

# DENSITY-2 cap judgment — may the envelope be raised for the mechanical retry?

Judge: independent Fable (cap judge), 2026-10-22 ~09:45 PT. Read-only. Inputs read in full:
`big-lever/JUDGMENT_SPARE_SLOT.md`; pack `exp/bl-density/density2-20261022-gatefix-f1904777/`
(`canary-admitted.json`, `canary-output/receipt.json` + `upload-measurement.json`, `production-admitted.json`,
`production-output/receipt.json`, `production-pending-02.json`); `research/bl_density/manifests.py`
(`finalize_production`, constants, identical on main `04ae4cbf` and gatefix `f1904777`). Sealed study not opened.
No jobs, no Colab calls, no commits.

## 1. Facts

- Canary: `PASS`, elapsed 297.9 s at 1.54 u/h ⇒ **0.127 u** (cap 0.5 / reserve 0.1 / 15 min). Its upload of the same
  3,244-byte `bootstrap.sh` took 1.1 s; 16/16 transfers `verified`; upload-measurement 12.5 s total.
- Production-01: `INCOMPLETE`, `failure_class = UPLOAD_CAP`, `exports = []`, `terminal = null`, `stop_verified = true`.
  A lease-heartbeat upload failed at 1.1 s, then `bootstrap.sh` stalled 125.2 s and the supervisor's shared 300-s
  total-upload cap fired (`supervisor.py:1367`). Elapsed 403.8 s at 1.54 u/h ⇒ **0.173 u**. No episode ran, so the
  namespace `bl-density-round14b-*` seeds are untouched and nothing could have been peeked at.
- Prior charge for the series: 0.127 + 0.173 = **0.300 u** by elapsed pricing (the operator's ~0.306 u includes a
  small rounding/poll margin; use 0.306 — overstating prior spend is the conservative direction).
- Admission arithmetic (`finalize_production`): effective = 5.0 − 1.0 − prior; priced timeout = min(180, 60·effective/1.54);
  refused if 2 × 75 > timeout. At prior 0.128 ⇒ 3.872 u / 150 min (what production-01 was admitted with). At prior
  0.306 ⇒ 3.694 u / **143 min < 150** ⇒ the refusal is purely this 2× safety rule, not a scientific cap.
- The spare-slot ruling, kill-rule (i): "Mechanical (transport, CUDA, export, hash) failure ⇒ one new attempt from a
  new prepared manifest, never a resumed one." It also says INCOMPLETE is "not extended" — that clause governs an
  episode-bearing INCOMPLETE (per-episode units retained), not a pre-episode transport failure, which is exactly the
  case clause (i) was written for. The ruling therefore already intends the retry; it simply did not foresee that a
  dead attempt's charge would be debited from the same 5 u and starve the 2× timeout rule.
- Owner budget is effectively unlimited (standing Colab-credits rule; never-idle rule). Scarcity never motivated the
  5 u figure — it was per-job sizing (1.54 u/h × 3 h + 20 % reserve).

## 2. Why raise rather than something else

- **Re-stating prior as 0.128 (canary only) is refused**: it would make `prior_spend_units` false; accounting must
  carry every charge of the series.
- **Lowering the 2× timeout rule or est_minutes is refused**: those are safety/forecast invariants, and the 75-min
  estimate is already at the optimistic end (105 min at 6-worker efficiency per the sizing).
- **Shrinking the reserve is refused**: the 20 % reserve is accounting-lag headroom, not spendable.
- Raising the nominal envelope by 0.5 u costs nothing real: the retry's charge is bounded by timeout × rate =
  163/60 × 1.54 = 4.18 u, so the series' worst case is ≤ 4.50 u, still inside the original ≤ 5 u ruling.

## 3. Exact parameter change (one retry only)

| parameter | production-01 | retry (this ruling) |
|---|---|---|
| total envelope (`PRODUCTION_CAP`) | 5.0 u | **5.5 u** |
| reserve (`PRODUCTION_RESERVE`) | 1.0 u | 1.0 u (unchanged) |
| `prior_spend_units` | 0.128 (canary) | **0.306** (canary 0.127 + mechanical 0.173) |
| effective `budget.unit_cap` | 3.872 u | **4.194 u** |
| `timeout_minutes` | 150 | **163** (= int(60 × 4.194 / 1.54); 2 × 75 = 150 ≤ 163 ≤ 180) |
| `requirements.est_minutes` | 75 | 75 (unchanged) |
| episodes / NAMESPACE / workers / SHAs | frozen | frozen, byte-identical |

Implementation note for Sol (mechanical, no re-review of science): `finalize_production` reads the module constant;
give it an explicit `envelope_units` argument (default 5.0) and pass 5.5 for this retry, recording
`density.total_envelope_unit_cap = 5.5` and `density.envelope_raise_reason = "JUDGMENT_DENSITY2_CAP mechanical retry"`.
The `prior_spend_units <= CANARY_CAP` guard (0.5) still passes at 0.306 but its message is now misleading; relax it to
`CANARY_CAP + mechanical charge` or rename — non-blocking.

## 4. Conditions

1. This is **the** single mechanical attempt under kill-rule (i). A second mechanical failure makes DENSITY-2
   terminal (`INCOMPLETE_MECHANICAL`); the slot then idles lawfully and no further attempt or envelope change may be
   made without a new judgment that first diagnoses the transport path.
2. New prepared manifest (`production-pending-02.json`, new `output_dir` production-output-02), never a resume; same
   commit `f1904777`; `source_qualification` must bind the requalified final-only PASS receipt
   (acceptance-1792685115), not remain `PENDING_REQUALIFICATION`.
3. The retry's transport is the same supervisor path the canary passed; the operator may (not must) run the
   zero-cost upload-readiness probe before dispatch. No change to the 300-s upload cap is required or requested here.
4. One look at terminal state, as registered; the failed attempt's receipt is accounting evidence only.
5. Nothing here touches the ≤ 180-min wall cap, the no-third-enlargement rule, the pooling ban, or the served hero.
