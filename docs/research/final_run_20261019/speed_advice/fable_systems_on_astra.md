# On Astra's three dissenting points (Fable systems, 2026-10-19 ~22:15 PT)

## 1. "Fits are the real GPU case" — disagree on magnitude, agree on procedure

Astra's 4-hour L4 fit is illustrative, not measured (no fit timing exists in the repo).
Arithmetic: 1 M rows × 5 epochs × ~70 MFLOP (forward + backward of a 22.5 MFLOP net) ≈ 350 TFLOP;
an L4 at a pessimistic 5 TFLOP/s effective is ~70 s/epoch, ~6 min per fit. S1a doubles the FLOPs
and makes the uint8 bank 66 GB (Astra's number), so it is bound by reading 66 GB × 5 epochs from
local SSD (~5-6 min/epoch), not by the GPU. A 4-hour fit means ~350 rows/s — a starved loader,
the case Astra says to fix first. Cached, a fit is ≈ 0.2-0.5 u on L4; a 3x faster A100-40 saves
minutes and costs more units. Agreed sequence: cache → time a full L4 fit → only then price A100.
**Settling measurement:** one complete S1b and S1a fit on L4 with the cached bank, GPU
minutes/epoch and loader wait reported separately. If it exceeds ~1 h GPU-bound, Astra is right.

## 2. "CPU-only / High-RAM could beat L4 for the teacher" — partly agree

Host-bound, yes, but CPU-only also pays B8 on the CPU: 0.29-0.5 ms/row on a Mac core, ~1 ms on a
Colab vCPU, versus ~1 us on L4. Per row-step: L4 lane ≈ 300 us host (6 physical cores → ~20k
row-steps/s); CPU runtime ≈ 300 + ~1,000 us ≈ 1.3 ms (~770/s per core). Per unit: L4 ≈ 12k
row-steps/s per (u/h); CPU ≈ 9.6k × cores at 0.08 u/h, 1.5k × cores at the 0.5 ceiling. CPU-only
wins per unit with ≥ 2 cores at 0.08 u/h, needs ≥ 8 cores at 0.5 — Astra's quarter rule restated.
But at ~10-35k row-steps/world either runtime labels 800 worlds for 1-4 u; the difference is
≤ 3 u per round and the Mac does it free. Choose by wall time and simplicity (one L4 lane that
also fits), not units.
**Settling measurement:** `nproc`, `free -g`, a B8 batch-64 CPU timing and a 10-world shadow run
on one CPU-only runtime, read against `colab usage` (< 0.1 u).

## 3. "Bisection is not certified" — partly agree; the design already covers it

Monotonicity can fail (an opponent moving away reopens an escape at t+1 that was closed at t),
so plain bisection returns *a* boundary, not necessarily the last. Two facts bound the damage:
(a) DESIGN_V2_SHADOW §2 already specifies bisection **plus stride-1 refinement near the
boundary**, and M2 gates on ≥ 90% of T3's 54/64 interventions recovered; (b) expands near death
are cheap because branches die early, so a ±16-step stride-1 scan is ~32 small expands per doom,
~1-3 s per world on a Mac thread. I never counted overlapping confirm labels as decisive examples;
they are the KL anchor rows, as FINAL_PLAN §3.1 says.
**Settling measurement:** on ~50 realized deaths, evaluate survivability at every t of the
150-frame window (≈ 7,500 early-terminating expands, < 30 min on three Mac threads); report the
non-monotone fraction and the gap between bisection+refinement's LAD and the true LAD.
