# A100 advice round — synthesis (Claude, 2026-10-20 ~17:50 PT)

Round: 5 Fable prompters × 2 Astra calls (fast mode) = 10 Astra answers (`astra_p1..p5_*.md`), then 3 independent
Fable reviewers who wrote no advice (`REVIEW_A_evidence.md`, `REVIEW_B_decision.md`, `REVIEW_C_adversarial.md`).

## Verdict (unanimous: 10/10 Astra answers after cross-examination, 3/3 reviewers)
**A100s would not get us a better snake sooner.** No A100 runs this week, and no A100 placement test either.
- An A100 runtime's host has the same 12 vCPUs (6 physical cores) as an L4's in every record we hold. The teacher
  and Round 0 are bound by that host CPU, not the GPU: B8's forward pass is ~8–13% of worker time, so even a
  perfect GPU would give at most ~1.15× (Amdahl). A100 costs ~3.2× the units of an L4.
- The student fit is the only place a GPU could matter. Its arithmetic takes minutes; production fit time is
  unmeasured. Reopen A100 only if the first phase-timed L4 cached fit (already funded) takes ≥ 60 min with the GPU
  leg ≥ 40%. In that case run one matched L4 vs A100-40 fit packet (cap 12 u): adopt at ≥ 1.25× and ≥ 1 h saved;
  reject at ≤ 1.10×.
- Rejected: an A100 teacher test (P1), a GPU-resident search port (P2; pays only at multi-million searched
  decisions per round). G4 host probe (48 vCPU, ~13 u) only after a Round 0 GO, with a Fable portfolio ruling.

## What actually gets a better agent sooner (verified in code)
1. **The Claim A label → replay → fit pipeline is unbuilt.** This, not compute, binds Claim A:
   `research/t34_exit_b8/label.py:108–111` refuses fan-out with a student checkpoint, the manifest builder refuses
   nondefault checkpoints, there is no production ReplaySource/probe boundary, and the CUDA fit is unqualified.
   Spec it for Sol now and qualify it on idle L4 lanes with the 39 A5 export worlds already on disk. That also
   yields the fit-timing receipt.
2. **Path B pacing:** the pre-check and efficacy also pace the *bare* arm at 24 fps (`retention_v2.py:1289`,
   `1407–1416`, `1593–1601`). About 11 of ~27 Mac critical-path hours are sleep. A pre-data amendment (bare arm
   unpaced) could save ~10–12 h. It needs the review loop + Fable approval before Phase 0 ends (registered design).
3. **Prospective ruling:** SUCCESS_CONTRACT §5 (Path B before R1/R2 labels) vs DESIGN_V2 :103 / Amendment 2 §4
   ("or Round 0"). Overlap could save 1–2 days. It needs one Fable judge before any Round 0 analysis.
4. **Write the NO-GO branch now** (Round 0 / Phase 0 negative → T3-dense fallback). Phase 0 is the hedge.
5. Freeze the Claim B strict-gate design skeleton during the Mac wait.

Earliest credible dates (reviewers): Claim B ~Oct 25–28, Claim A ~Oct 26–30 PT. A100s move neither.

## Corrections to the facts brief
129 h is the *post*-speedup six-worker elapsed figure. "~95% CPython" is pre-fan-out. The Path B pre-check is
~6–7 h, not ~1.5 h. Chunk 000 is on its 166-min forecast (the "3.5 h" estimate was a stale snapshot).
