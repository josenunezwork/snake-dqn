# Research Governance, Statistical Amendments, and CI/Rules Audits (`governance`)

**Dates:** 2026-09-26 to 2026-10-30
**Verdict:** Active governance record: established 3-tier evidence gates, ratified sequential OBF strict gates, survival band `v2` (`pooled_ni_continue`), `LH-1`, and sequential Phase R, pinned strict gates and Python CI (`macos-15` ARM64, `T14`) to Darwin ARM64 after cross-platform float/`erf` drift, and ratified the `2026-10-19` Campaign Authority + `2026-10-30` Fast-process rule.

## What was tried
- Defined Tier `0` (saved-data), Tier `1` (dev screen), and Tier `2` (strict promotion) gates (`governance_tiers_2026-09-26.md`) and CPU-slot thermal rules (`compute_policy_2026-10-02.md`).
- Ratified statistical amendments (`2026-10-02` to `10-08`): `4`-look OBF strict gates, `paired_ni_at_stop` -> band `v2` (`pooled_ni_continue`, `0.05` pooled NI, `0.075` per-mix floor, `0.30` floor), `LH-1` (`48` H10000 worlds/mix, sibling erratum preserving hash `85c3b851...`), and opt-in `sequential-phase-r-obf-hk-v1` (`strict_on_runpod` superseded).
- Repaired CI on `macos-15` ARM64 (`T14`, `fe69d208`: `torch==2.9.1`, pytest `9`, isort `8`, `76`-test numerical preflight).
- Conducted the `2026-10-19` `3`-lane Astra + Fable rules audit (`P7_STATISTICS.md`) and adopted the `2026-10-30` Fast-process rule (*Exploration* vs *Confirmation* lanes).

## What happened
- **Mac-only strict gates (`strict_on_runpod` SUPERSEDED):** Mac Accelerate vs x86 MKL AVX2 broke exact float32 Q ties differently in **`~1` in `50` H10000 episodes (`1/54` in `FRP-v5-S2`)**, keeping strict and serving gates on Mac.
- **`T14` CI repair (`PASS`):** Linux vs Darwin `erf` differed in the **last bit** (`1.941938163774193e-06` vs `1.941938163829704e-06` at `4.617501629587915`) and macOS `26` failed `6` Metal tests (issue `14380`); `macos-15` ARM64 passed **`6,945` Python + `178` frontend tests**.
- **Owner-ratified campaign & fast-process rules (`2026-10-19`, `2026-10-30`):** Ratified `15` audit fixes (paired world = fixed-controller unit, released bare `B8 s3` comparator, survival guard A/B, fail-closed provider lag, `CODEX_MODEL` pin) and split *Exploration* (no pre-registration ceremony, `1` reviewer) from *Confirmation* (pre-data bars, fresh paired worlds, verified stops).

## Why
Hardware-specific BLAS reductions (`arm64` Accelerate vs `x86` AVX2) and OS math libraries (`erf` last bit, macOS `26` Metal) break bit-exact trajectory and SHA-table parity across platforms, while fixed controllers over frozen weights have zero training-seed variance (making paired worlds the true unit of inference).

## What to reuse / what not to repeat
- Keep release confirmation gates on Mac, Python CI on `macos-15` ARM64 (`torch==2.9.1`), and SHA-bound governance docs immutable (use sibling errata, as on `LH-1`).
- Use the *Exploration* lane for screens/pilots/speed work and reserve *Confirmation* ceremony for releases (`AGENTS.md`, `P7_STATISTICS.md`).

## Full record
Original: `git show 90b3db94:docs/research/history/governance.md` | Primary sources: [governance-and-owner-rules.md](../../notes/governance-and-owner-rules.md), [P7_STATISTICS.md](../final_run_20261019/P7_STATISTICS.md), [t14_ci_validation_2026-10-17.md](../t14_ci_validation_2026-10-17.md)
