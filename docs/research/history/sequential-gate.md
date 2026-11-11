# Sequential Strict Gate, Phase R Protocol, and Two-Tier Gate Calibration

**Dates:** 2026-09-26 to 2026-10-16
**Verdict:** Adopted and active: the `4`-look Lan–DeMets O'Brien–Fleming sequential strict gate (`strict-sequential-obf-bonferroni-v1`) and paired survival band `v2` (`pooled_ni_continue`, ratified `2026-10-06`) cut expected evaluation episodes by `27–75%` while controlling familywise error `<= 0.0492` (`sequential_phase_r` remains opt-in simulation/code).

## What was tried
- Calibrated `H5000` mass decomposition, cross-policy pairing correlation, and Lan–DeMets OBF futility spending over `144` saved episodes (`48` dev Apex + `48` candidate `6102` + `48` paired Apex; `85/96` Apex self-deaths: `40/48` dev, `45/48` pilot) in `gate_calibration_20260926`.
- Validated the `4`-look (`61, 122, 183, 243` worlds/mix) sequential strict gate (`src/evaluation/sequential_gate.py`) over `200k` null / `50k` alternative Monte Carlo reps, `300k` skewness reps, and `1,250` exact cross-check reps (`sequential_gate_validation_20261002`).
- Validated paired survival noninferiority (`paired_ni_at_stop`, `M = 0.05`) vs the old `16`-world reference band (`paired_band_validation_20261003`), then calibrated pooled continuation band `v2` (`pooled_ni_continue`, `rci_obf`, `survival_band_v2_20261006`) and opt-in `sequential_phase_r` (`10,000` reps).

## What happened
- **Pairing & sample-size calibration:** Divergent trajectories drove per-world mass correlation near zero (**`r = -0.22` to `+0.11`**: `-0.01` frozen, `+0.11` mixed, `-0.22` scripted; death timing `R^2 = 0.85–0.90`), and a `16`-world draw inflated `10%`-MDE scripted `N` from `3,047` to `8,676` (`212/141/762` at `20%`). OBF futility held false stops at `0.096 <= 0.10` (vs `~27%` naive).
- **`4`-look OBF strict gate (`ADOPTED`):** Matched `sequential_gate.py` on `1,250/1,250` reps (`0` mismatches), held global-null `P(PASS) = 0.00058` and FWER `0.0492`, and cut expected worlds (vs `729` fixed) by **`27–75%`** (`532` = `-27.0%` at `delta = 0`, `369` = `-49.4%` at `+60`, `183` = `-74.8%` at `+130`; `597` at `+30`, `0.9434` power), adding a skew guard for `-2.83` skewness.
- **Paired survival bands (`v2` `RATIFIED`):** Un-paired `16`-world bands falsely rejected non-regressing models **`77–91%`** (`9.2–23.1%` pass) while letting `-0.05` drops pass `0.07–0.32` (`6.6–32.2%`); paired NI cut false rejects to `~7–9%` (`91.4%–94.5%` pass) and `-0.05` error to `0.020–0.053`, with `v2` (`pooled_ni_continue`, `0.075` floor) adding `CONTINUE_BANDS`.

## Why
Multi-snake chaos destroys per-world mass correlation once two policies take a single different move (`r = -0.22` to `+0.11`), requiring near-unpaired `N`, whereas pairing shares exact world seeds for deterministic incumbent caching and eliminates `16`-world reference-draw noise (`SD ~0.06` vs `0.02` margin).

## What to reuse / what not to repeat
- Reuse `src/evaluation/sequential_gate.py` and `research/sequential_strict_template/` (`/v2-pooled` with `rci_obf` and pre-final skew check; `sequential_phase_r` is opt-in) for confirmation gates.
- Never size gates assuming positive paired correlation between divergent policies, and never repeat unadjusted interim futility checks (`~27%` false stops) or un-paired `16`-world survival bands (`77–91%` false rejects).

## Full record
Original: `git show 90b3db94:docs/research/history/sequential-gate.md` | Primary sources: [gate_calibration_20260926/README.md](../../../research/gate_calibration_20260926/README.md), [paired_band_validation_20261003/README.md](../../../research/paired_band_validation_20261003/README.md), [sequential_strict_template/README.md](../../../research/sequential_strict_template/README.md)
