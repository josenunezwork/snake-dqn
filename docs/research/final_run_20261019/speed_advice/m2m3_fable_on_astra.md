# On Astra's M2/M3 dissent (Fable, 2026-10-20 ~02:00 PT)

Advice only; numbers from the M2 bank, T3 dense and M3 receipts, `teacher.py:362`, `shadow.py:167-303`.

## 1. Label target: ranked rule with alive-extension — partly agree

Agree: export the full ranked outcome (per-action survive/alive-frames/mass, censoring flag) and distil
preferences between evaluated actions; "every dying action equally forbidden" is too coarse. Disagree that the
alive-extension class is a decisive target: the 33 all-die changes gained 1 / 2 / 4 frames (quartiles), all sit
in the 6 worlds that died, and the 16 surviving worlds are explained entirely by horizon rescues (20/20
recovered). A ~2-frame margin out of 60-230 is noise at τ=0.05; export it tagged "delay" with weight ~0. The
test below reports rescue and delay contributions separately, as Astra proposes.

## 2. The deciding test and the bank — agree with Astra; my T3-bank proposal is demoted

P7 §2 (fresh, collision-audited namespaces per study) and §1 (paired worlds, ≥16 per mix) bind every T34
registration, and the §5 check feeds a Claim-B decision. The T3 dense bank is reused, scripted-only
development data from the worlds T3 was tuned on; it cannot decide. Keep it as a 1.5-h *replication pre-check*
(expect ≥50/64) that gates the fresh-bank hours; it decides nothing.

Size: T3's paired mass delta has mean +92, SD 178 (n=64, 42 zeros). At 16/mix (48 worlds) LB90 ≈ +59 for a
T3-like effect but ≈ +7 if the paced effect is +40 — underpowered at the bar. Register 32/mix (96 worlds):
LB90 ≈ +16 at +40, +68 at +92; ~22 h elapsed at 24 fps only (12 fps reported later).

## 3. Smaller points — agree

ONNX saves 0.21 ms of 4.5-6.6 ms per row-step on the serial path (3-5%); it matters only on a batched engine
where B8 is 35-50% of the step. T33 does reject veto-bearing opponents (`TypeError`, teacher.py:362), so its
numbers cannot price Watch6 until exact v8 dispatch is added. `shadow.py` reads/writes process-global `random`
state (lines 167, 278, 303): isolate before threading. Confirm+advance 17.5 ms p50 = 42% of a core at 24 fps;
stop advancing a doomed shadow past its death.

## 4. The one test I would freeze

**Path B v2 paced retention test.** Fresh disjoint namespaces, 32 paired H5000 worlds per mix (scripted / frozen /
mixed) plus 16 fresh continuous-respawn mixed-Watch6 sessions; hero B8 s3 bare vs the frozen outcome-ranked
successor paced at 24 fps (one worker, one request, bounded memory, stale rejection, bare fallback), plus the same
successor unpaced as a descriptive arm. One look, fixed N, Holm over the two active arms, P7 inference. Source,
search schedule (earliest-first, D=500), load profile and all bars frozen before data; T3-bank pre-check must have
read ≥50/64 first. Cap 32 h elapsed / 48 CPU-h, $0, after Phase 0.

- **GO (Claim B proceeds to the strict gate):** paced arm meets every contract bar — pooled mass ≥ +40 with
  LB90 > 0, scripted no-death ≥ +0.03, survival guards A/B, boost share [0.5%, 10%], Watch6 deaths/1k lower
  with paired 95% CI excluding 0, zero paced overruns, update p95 ≤ 11.688 ms — with misses/fallbacks counted.
- **KILL (drop Claim B):** paced arm fails any skill bar while the unpaced arm passes them (deadline loss), or
  both arms fail the skill bars, or any unconditional fault/timing overrun; teacher labels then wait on Round 0.
- **INCONCLUSIVE:** missing rows, parity/identity failure, cap reached, LB90 within ±10 of 0 with point estimate
  ≥ +40, or product bars pass but one skill bar misses by less than its interval — report, no extension, owner
  decides whether to fund a second fixed block.
