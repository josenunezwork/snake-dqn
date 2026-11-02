# Research Governance, Statistical Amendments, and CI/Rules Audits (`governance`)

## Question and Goal

How did the `snake-dqn` experimental governance framework evolve from the September 2026 three-tier gate system (`governance_tiers_2026-09-26.md`) through the October 2026 statistical, survival-band, compute-policy, long-horizon (`LH-1`), and sequential Phase R amendments, the T14 macOS ARM64 CI reproducibility lock (`t14_ci_validation_2026-10-17.md`), and the October 19, 2026 owner-ratified campaign rules audit (`2026-10-19_rules_audit_*`)?

## Timeline of Versions and Steps

| Date | Document / Milestone | What Was Established / Audited | Verdict / Status |
|---|---|---|---|
| 2026-09-26 | [`governance_tiers_2026-09-26.md`](../governance_tiers_2026-09-26.md) | Defined the three-tier evaluation hierarchy: Tier 0 smoke, Tier 1 development screen (`H3000`, `N >= 60`), and Tier 2 strict promotion gate (`H5000`, `N = 200`, paired bootstrap CI). | **RATIFIED BASELINE** |
| 2026-10-02 | [`compute_policy_2026-10-02.md`](../compute_policy_2026-10-02.md), [`governance_amendment_sequential_gates_2026-10-02.md`](../governance_amendment_sequential_gates_2026-10-02.md) | Set the RunPod speed bar (`>= 5x` Mac wall-clock, later lowered to `>= 4x` on 2026-10-05) and added two-stage O'Brien-Fleming group-sequential stopping for Tier 2 gates. | **RATIFIED** (RunPod superseded for T34) |
| 2026-10-03 | [`governance_amendment_paired_bands_2026-10-03.md`](../governance_amendment_paired_bands_2026-10-03.md) | Replaced raw absolute Tier 2 survival floors (`0.95` / `0.90` / `0.85`) with paired difference bands (`Δ >= -0.02` pooled, `Δ >= -0.03` per mix) plus a `0.50` catastrophe floor. | **RATIFIED** |
| 2026-10-04 – 2026-10-08 | `LH-1` long-horizon screen amendment (`ecddfab`) & erratum | Ratified the `H15000` (`N = 64`) post-Tier-2 long-horizon screen (`C2` descriptive, `C3` binding at `mass >= 300`); calibrated as `LH-1 ADOPTABLE` and passed `CLEAR` on `frp3-s12` ([`/Users/josenunez/Projects/ml/snake-dqn-artifacts/lh1-screen-frp3-s12/run-v1/receipt.json`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/lh1-screen-frp3-s12/run-v1/receipt.json)). | **RATIFIED / CLOSED** (SHA-bound file preserved via sibling erratum) |
| 2026-10-05 – 2026-10-08 | [`governance_amendment_strict_on_runpod_2026-10-05.md`](../governance_amendment_strict_on_runpod_2026-10-05.md), [`governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md`](../governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md) | Proposed running strict gates on RunPod x86; **never ratified** because Mac ARM64 vs Linux x86 float32 tie-breaking diverges on exact Q ties (`0` ULP difference). Strict gates remain **Mac-only**. | **NOT RATIFIED / SUPERSEDED** |
| 2026-10-06 | [`governance_amendment_survival_band_v2_2026-10-06.md`](../governance_amendment_survival_band_v2_2026-10-06.md) | Replaced binary endpoint survival in the paired band with continuous alive-frame fraction (`min(death_frame, H) / H`, `Δ >= -0.03` per mix, `-0.15` per-seed scripted floor) + `0.50` episode floor. | **RATIFIED & ACTIVE** |
| 2026-10-07 | [`governance_amendment_sequential_phase_r_2026-10-07.md`](../governance_amendment_sequential_phase_r_2026-10-07.md), [`sequential_phase_r_design_2026-10-07.md`](../sequential_phase_r_design_2026-10-07.md) | Adopted two-look (`n_1 = 96`, `n_2 = 192` worlds/seed) sequential Phase R with Hwang-Shih-DeCani spending (`γ = -4`, `α_1 = 0.01798`) and hierarchical Hartung-Knapp (`df = 4`) inference. | **RATIFIED** |
| 2026-10-17 | T14 CI validation (`fe69d208` on `3cbbbdc0`) | Pinned `macos-15` ARM64, `pytest==9.0.3`, `isort==8.0.1`, and `torch==2.9.1` in [`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml); verified `6,945` Python tests and `178` frontend tests pass locally. | **PASS** |
| 2026-10-19 | Rules audit & Fable synthesis ([`2026-10-19_rules_audit_fable_judgment.md`](../advice/2026-10-19_rules_audit_fable_judgment.md)) | 3-domain Astra audit (engineering, ops, research) synthesized by Fable and ratified by owner ("yes all") into [`AGENTS.md`](../../../AGENTS.md) and [`P7_STATISTICS.md`](../final_run_20261019/P7_STATISTICS.md). | **RATIFIED BY OWNER** |
| 2026-10-30 | Fast-process rule ([`AGENTS.md`](../../../AGENTS.md), [`docs/notes/governance-and-owner-rules.md`](../../notes/governance-and-owner-rules.md)) | Established two lanes (*Exploration* vs *Confirmation*), one reviewer per change, fewer canaries, and automatic mechanical retries while preserving held-out separation and Mac confirmation gates. | **STANDING RULE** |

## Final Verdict and Key Numbers

All durable governance rules, statistical gate definitions, and operational invariants are consolidated in [`docs/notes/governance-and-owner-rules.md`](../../notes/governance-and-owner-rules.md) and [`AGENTS.md`](../../../AGENTS.md). Key facts preserved from the consolidated erratum, CI validation, and rules-audit packets:

1. **Long-Horizon Screen (`LH-1`) Ratification Erratum (`2026-10-08`):**
   - The `LH-1` amendment (`lh1-screen:docs/research/governance_amendment_long_horizon_screen_2026-10-04.md`, SHA-256 `85c3b851df16ee6e97d651ad9f607d67847a139398744cf950b4280e1d803e27`) was ratified on **2026-10-05** in commit `ecddfab` (`C2` descriptive, `C3` binding only at `mass >= 300`; calibration verdict `LH-1 ADOPTABLE`).
   - Because `saved_snakes/frp3_s12.intent.json` binds that file's exact SHA-256 bytes, its `DRAFT — NOT YET RATIFIED` header was never edited in place. The `frp3-s12` `LH-1` screen executed under that hash and recorded verdict **`CLEAR`** at [`/Users/josenunez/Projects/ml/snake-dqn-artifacts/lh1-screen-frp3-s12/run-v1/receipt.json`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/lh1-screen-frp3-s12/run-v1/receipt.json).
2. **T14 CI Validation (`2026-10-17`, commit `fe69d208` on baseline `3cbbbdc0`):**
   - Pinned `pytest==9.0.3`, `isort==8.0.1`, `torch==2.9.1`, `scipy==1.15.2`, `psutil==7.0.0`, and `pyyaml==6.0.2` across [`requirements.txt`](../../../requirements.txt) and [`requirements-dev.txt`](../../../requirements-dev.txt), and switched the Python CI job in [`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml) from `ubuntu-latest` to **`macos-15` (ARM64)** (avoiding the `macos-26` Metal shader validation regression in `actions/runner-images#14380`).
   - Why Darwin ARM64 was required for CI: `tests/test_phase_r_A2_foundations.py` asserts exact SHA-256 digest `23c2704155e7f5a07be63814db2dc7076bc6a59e0de15709397ac08876d556fa` on `PREREGISTRATION.json`, which depends on `NormalDist().cdf(-4.617501629587915)`. Darwin ARM64 libm produces `1.941938163774193e-06` whereas Linux glibc x86_64 produces `1.941938163829704e-06` (`5.55e-17` last-bit `erf`/`erfc` drift).
   - Local verification passed **`6,945` Python tests** (`6` skipped, `77` slow deselected in `678.79 s`), **`178` Vitest frontend tests** across `19` test files plus `tsc -b` + `vite build`, and clean `black` / `isort` / `flake8` checks ([`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml)).
3. **2026-10-19 Campaign Rules Audit (`2026-10-19_rules_audit_*`, baseline `2482d3bc`):**
   - Three domain audits (`astra_eng`, `astra_ops`, `astra_research`) and Fable's synthesis ([`docs/research/advice/2026-10-19_rules_audit_fable_judgment.md`](../advice/2026-10-19_rules_audit_fable_judgment.md)) identified four self-imposed bottlenecks that were ratified for removal by the owner ("yes all", 2026-10-19 ~23:20 PT):
     1. **Fixed-controller evaluation unit:** Evaluating deterministic fixed controllers (`b8-s3`, search/shield wrappers) across `5` pseudo-"training-seed" blocks with Hartung-Knapp `t_{df=4}` inflated CI width by **`1.42x`** (`t_{0.90, 4} = 1.533` vs `z_{0.90} = 1.282`) and required **`2.01x`** the worlds; replaced with paired-world inference (`df = N - 1`, `N = 320` paired worlds -> `MDE ≈ 19.3` mass) in [`docs/research/final_run_20261019/P7_STATISTICS.md`](../final_run_20261019/P7_STATISTICS.md).
     2. **Single comparator (`b8-s3`):** Retired mandatory FRP3/champion multi-hero gate evaluations (`4` heroes x `1,920` worlds = `7,680` episodes) in favor of testing against the incumbent `b8-s3` (`2` heroes = **2x speedup**).
     3. **Claim A (bare) and Claim B (visibly labelled assisted):** Superseded the "bare network only" restriction so a visibly labelled assisted controller could also qualify under [`docs/research/final_run_20261019/SUCCESS_CONTRACT.md`](../final_run_20261019/SUCCESS_CONTRACT.md).
     4. **Batch pre-registration:** Replaced per-micro-step pre-registration ceremony with campaign-level registrations (`DESIGN_V2_SHADOW.md`) and later the 2026-10-30 Fast-Process Rule ([`AGENTS.md`](../../../AGENTS.md)).

## What Was Learned

- **Cross-architecture floating-point ties (`0` ULP) and libm `erf` differences break bitwise parity:** Mac ARM64 vs Linux x86 cannot be treated as bit-identical for greedy argmax on exact float32 ties or SHA-pinned `NormalDist.cdf` tables; strict gates and SHA-pinned statistical fixtures must run on Mac ARM64 ([`governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md`](../governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md)).
- **Match statistical degrees of freedom to the actual source of randomness:** `df = 4` across 5 training seeds is appropriate when comparing stochastic training recipes, but applying `df = 4` to two fixed deterministic controllers wastes half of evaluation capacity ([`2026-10-19_rules_audit_fable_judgment.md`](../advice/2026-10-19_rules_audit_fable_judgment.md)).
- **Separate exploration speed from confirmation rigor:** The 2026-10-30 Fast-Process Rule eliminated pre-registration/review ceremony on exploratory screens while preserving pre-data pass bars, fresh paired worlds, and Mac-only strict gates for release confirmation ([`docs/notes/governance-and-owner-rules.md`](../../notes/governance-and-owner-rules.md)).

## Current Status

**Active governance record.** All current standing rules live in [`AGENTS.md`](../../../AGENTS.md) and [`docs/notes/governance-and-owner-rules.md`](../../notes/governance-and-owner-rules.md).

## Keep Files in This Lineage

- [`docs/notes/governance-and-owner-rules.md`](../../notes/governance-and-owner-rules.md) — Comprehensive compendium of project governance, statistical amendments, and standing owner rules.
- [`docs/research/GLOSSARY.md`](../GLOSSARY.md) — Definitions of project terminology, metrics, rosters, and gate tiers.
- [`docs/research/INDEX.md`](../INDEX.md) — Chronological index of research studies, gates, and governance documents.
- [`docs/research/compute_policy_2026-10-02.md`](../compute_policy_2026-10-02.md) — October 2, 2026 cloud compute policy.
- [`docs/research/governance_tiers_2026-09-26.md`](../governance_tiers_2026-09-26.md) — Baseline three-tier evaluation and promotion governance.
- [`docs/research/governance_amendment_sequential_gates_2026-10-02.md`](../governance_amendment_sequential_gates_2026-10-02.md) — Group-sequential O'Brien-Fleming stopping amendment for Tier 2 gates.
- [`docs/research/governance_amendment_paired_bands_2026-10-03.md`](../governance_amendment_paired_bands_2026-10-03.md) — Paired survival-difference band amendment (`Δ >= -0.02` / `-0.03`).
- [`docs/research/governance_amendment_strict_on_runpod_2026-10-05.md`](../governance_amendment_strict_on_runpod_2026-10-05.md) — Unratified proposal to run strict gates on RunPod.
- [`docs/research/governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md`](../governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md) — Status note recording why strict-on-RunPod was superseded (Mac vs x86 float32 tie non-identity).
- [`docs/research/governance_amendment_survival_band_v2_2026-10-06.md`](../governance_amendment_survival_band_v2_2026-10-06.md) — Survival band v2 amendment (continuous alive-frame fraction + `0.50` episode floor).
- [`docs/research/governance_amendment_sequential_phase_r_2026-10-07.md`](../governance_amendment_sequential_phase_r_2026-10-07.md) — Sequential Phase R (`n_1 = 96`, `n_2 = 192`, Hartung-Knapp `df = 4`) amendment.
- [`docs/research/sequential_phase_r_design_2026-10-07.md`](../sequential_phase_r_design_2026-10-07.md) — Statistical design and simulation calibration for sequential Phase R.
- [`docs/research/advice/2026-10-19_rules_audit_fable_judgment.md`](../advice/2026-10-19_rules_audit_fable_judgment.md) — Owner-ratified Fable synthesis of the 2026-10-19 rules audit.
