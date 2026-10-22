RULING: Y3 envelope raised from 8.00 u to 12.00 u (plus one bounded mechanical-retry reserve); WAIT for the supervisor telemetry-stall fix + transport requalification before the single new canary; forecasts, prices, protocol and authority unchanged.

Judge: independent Fable judge, 2026-10-22, read-only. Inputs: `P1_Y3_RETRY_BUDGET_04.md`, `y3/source-latebody-04-budget-stop-01/reconciliation.json`, `P1_Y3_ADMISSION_ERRATUM_03.md`, `E8_LATE_BODY_AMENDMENT_02.md` caps section. The forbidden study directory was not opened. No actor outcome has been read, so this is a funding/transport ruling only, not a scientific one.

## 1. Findings accepted as fact

- Four real supervisor receipts, all `stop_verified=true`. Own-spend total 3.399723 u (remainder 4.600277 u); legacy conservative charge 1.62 + 0.28 + 2.52 = 4.42 u, plus 0.25 u if the family canary is charged in-envelope (4.67 u).
- Unchanged invoice for one complete retry: source 4.81 u + replacement 1.29 u = 6.10 u. Every accounting view fails it. The stop was correct.
- Source-03 failure class SUPERVISOR (worker EOFError, cli_guard kernel 404, EXIT=1): mechanical, no Y3 result. Source-02's refusal was pre-amendment (E8); latebody-02 refused itself. None of the 3.40 u bought a scientific observation.
- Canary evidence is not reusable: pinned transport tree `8747f306…` vs current `40985ca4…`; the identity rule (proof == current == manifest pin) is correct and stays.
- A supervisor telemetry-stall fix is in progress and will change the transport tree again, forcing a second requalification if a canary is run now.

## 2. New Y3 envelope — exact numbers

Charging policy: keep the frozen legacy conservative debit (max(own, gross), rounded up to cents). Fold the family canary's 0.25 u into Y3 from now on so there is one ledger (its separate <=1 u authorization is consumed and closed).

| Line | Units |
|---|---:|
| Charged to date (1.62 + 0.28 + 2.52 + 0.25) | 4.67 |
| New canary on the requalified transport (ceiling; charge actual) | 1.00 |
| Source retry, unchanged price (75/150 min, 1.54 u/h max, zero tolerance, 0.96 u reserve) | 4.81 |
| Replacement reservation, unchanged (20/40 min, 0.26 u reserve) | 1.29 |
| Worst-case total | 11.77 |
| **Y3 envelope v2 (hard cap)** | **12.00** |

Slack 0.23 u is for cent rounding only; it does not fund any extra attempt. Nothing in this envelope may be spent before the new canary PASSes on the same transport identity as the dispatch manifest.

Mechanical-retry reserve (separate line, not part of the 12.00 u): if the source retry fails again with a verified mechanical class (SUPERVISOR/transport, `stop_verified=true`, no actor outcome read), one further source attempt is pre-authorized at the same unchanged price, 4.81 u, plus <=1.00 u for one canary if and only if the transport identity changed in between. Worst-case total exposure of this ruling: 12.00 + 4.81 + 1.00 = 17.81 u. Against ~1,120 u of owner credits and the ~39 u/day spend goal this is immaterial; the binding constraint is calendar, not units.

Replacement is still not admitted by this ruling: after a source PASS, reconcile the actual source debit against the 12.00 u cap and price the replacement from the actual residual, as the erratum requires. If the residual is below 1.29 u the replacement returns to a judge; it is not squeezed.

## 3. Transport: wait for the telemetry fix, then one canary

Ruling: WAIT. The source-03 death (EOFError + kernel 404) is exactly the failure mode the telemetry-stall fix addresses. Running a canary and a 4.81 u source on the current transport risks a repeat of the same mechanical failure and then a second requalification anyway: two canaries and up to two burned sources instead of one. One transport change, one canary, one source.

Conditions on the wait:
1. The fix merges to main, the qualification registry is updated to the new transport tree, and the Colab CLI lane is re-QUALIFIED on that pin (same standard as the 2026-10-19 PASS on dab00823) before the new canary manifest is created.
2. Bound: if the fix is not merged + requalified by **Fri 2026-10-23 18:00 PT**, the preparer proceeds on the then-current main transport without further judge approval, accepting the stall risk under the existing `stop_verified` protections. The never-idle rule is satisfied either way: the L4 lane is not reserved while waiting, and other lanes keep running. This wait does not threaten KEEP_B8 (Tue 23:00); a 75-min source plus a 20-min replacement fits with days to spare.
3. The new manifest must declare the job device explicitly (`cuda:0`) to clear the `ambiguous job device` refusal. That is a manifest transport field, not a scientific change, and is allowed. No device override, old-broker bypass or source repin.
4. New immutable preparation (`source-latebody-05-…`) with its own independent review and debit receipt, citing this judgment and the 12.00 u cap. The 04 budget-stop root stays as is.

## 4. Retry policy for further mechanical failures

- Mechanical = failure class SUPERVISOR or transport, `stop_verified=true`, no cleanup error, and no actor outcome read. Only this class triggers the reserve in section 2, and only once, automatically (Claude dispatches; no new judge round).
- A second consecutive mechanical failure after the fix is a root-cause problem, not a budget problem: STOP, no further retries, back to a judge with the two logs. Do not raise the cap again to outrun a transport bug.
- Not mechanical and not retried under this ruling: time-cap INCOMPLETE on a job that was running (amendment 02: never permission for another support check), any refusal by the admission/calibration gate (rate > 1.54 u/h, other assignments present, HOLD), or any run where an actor outcome was read. Those fall back to the original Y3 rules and the owner's/judge's scientific authority.
- Each retry is a fresh attempt directory and fresh receipt; prior receipts are never refunded or amended. The failed source-03 gross 2.52 u stays charged.

## 5. Must not change

- Source forecast 75 min / timeout 150 min / cap 4.81 u / reserve 0.96 u; replacement 20 / 40 min / 1.29 u / 0.26 u. No shortening to fit any invoice.
- L4 rate ceiling 1.54 u/h with `rate_tolerance_units_per_hour=0.0`; calibration refuses higher.
- Legacy conservative debit policy; immutability of every prior registration, amendment, erratum, receipt and the 04 stop record; old source-02 remains NOT_READY and is never rescored.
- Scientific protocol: ruling (c) requirements, float64 SVD `(1,h,L)` comparator, registered EV computation, thresholds, 4M fit-world transition cap, bank exclusions, 1M <=4 u gate (40/80 min, 0.8 u reserve) requiring an actual Y3 PASS, 100M endpoint, timing windows.
- Canary identity rule (proof == current == manifest pin); Claude performs usage + sessions immediately before dispatch with zero other assignments; Codex never dispatches; owner keeps release authority; the t34-round0-study-01 directory stays closed.

Not committed by the judge.
