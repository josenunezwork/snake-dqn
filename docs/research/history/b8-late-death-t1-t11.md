# B8 Late-Game Death Census, Two-Expert Panel and T1–T11 Probe Series (2026-10-14 to 2026-10-17)

**Dates:** 2026-10-14 to 2026-10-17
**Verdict:** Exact search proved `46/49` (`93.9%`) of B8's late (`L >= 800`) self-enclosure deaths were escapable within `300` decisions (`T3` dense shadow `54/64` vs `38/64` survival), every bare-network fine-tune and static feature probe (`T2–T6`, `T8–T11`) failed (`0.73–0.77` D300 AUC), and unmodified `b8-s3` passed restated `T7` `STRICT_PASS` for release.

## What was tried
- Ran a scripted death census (`b8_death_census_20261014`, `5` seeds × `32` worlds) after M4 Phase R look 2 showed `B8` beating `frp3-s12 + v8` by `+128.7` mass (`LB +101.5`) while failing single-seed guard `c6` vs `champion_a5 + v8` (`-0.156` vs `>= -0.05`, single-seed `SE` `~0.075`).
- Replayed all `75` census deaths (`49` at `L >= 800`, `26` at `L < 800`) backward `300` steps across all legal actions (`T1a`), probed B8's `256`-D post-ReLU fuse vector (`T1c`, `T5b`), and ran the `T3` `D=300` dense rollout ceiling (`L >= 600`, stride `5`).
- Screened `5`-seed × `8M` fine-tunes from B8 across Huber/anchor regimes (`T2`), `T3` rollout labels, `T4a` two-sided counters, `T4e/T4f` `H300` return regression, `T6` snapshot restarts, and `T11` `LOCAL30`.
- Probed `24` per-action geometry features (`T5`), arrival-race + raw-raster D300 CNN (`T8`), `46`-pair post-LAD divergence (`T9`), and tail-certificate / horizon AUC decay (`T10`).

## What happened
- **Death census:** B8's death hazard per `1,000` decisions was `0.068` at `L < 800` vs `0.177` at `L >= 800` (`49/75` deaths, `85%` enclosures, median trap `31` steps; champion spent only `5.6%` of decisions at `L >= 800` vs B8's `42%` of `659,595` decisions).
- **Search headroom (`T1a`, `T3` dense `GO`):** `T1a` found `46/49` (`93.9%`) late deaths escapable within `300` steps (median LAD Q-gap `0.45`; `17/46` boost-only), and `T3` dense search lifted scripted survival `38/64` -> **`54/64`** and mass `608.1` -> **`700.0`** with just `75` changed actions in `301,589` decisions.
- **Static features & post-LAD lag (`T1c`, `T5`, `T8–T10`):** H100 fuse AUC was `0.837`, but D300 AUC stayed `0.73–0.77` across `60` scalars (`0.912 → 0.726` D30->D300), `24` geometry features (`0.748`), fuse GBMs (`0.7630–0.7693`), arrival-race (`0.749`), and raw-raster CNN (`0.742`); `75%` pocket separation appeared at offset **`49`** (`65–83` sustained, `T9`).
- **Fine-tunes (`KILL` / `No GO`) & `T7` (`STRICT_PASS`):** `T2` (`HUBER_ANCHOR0` `KILL`, `ANCHOR0` `+0.003`), `T3` labels (`-88.0 / -87.9 / -81.2` mass), `T4a` (`-0.027` survival, `-24.1` mass), `T4e` (`34%` boost at `0.8M`), `T6` (`0.0000` survival), and `T11` (`1.073% > 1%` boost at `300,160` steps) failed; unmodified `B8` passed `T7` (`+138.67`, Tier 2 `-0.0156`).

## Why
B8 dies more often at `L >= 800` primarily because it grows big enough to spend `42%` of its decisions there (vs `5.6%` for champion), where fatal enclosures look nearly identical to safe play at the LAD (median Q-gap `0.45`) and separate geometrically only `49–65` steps later, causing static single-frame student updates to over-perturb normal play or collapse into boost.

## What to reuse / what not to repeat
- Stratify survival hazard by body length (`L < 800` vs `L >= 800`) and avoid single-seed `32`-world survival gates (`SE ~0.075`, `MDE ~0.19`).
- Use online multi-step rollout search (`75` interventions in `301,589` decisions lifted survival `38/64 -> 54/64`) rather than static D300 feature engineering (`0.73–0.77` AUC) to rescue sealed-neck enclosures.

## Full record
Original: `git show 90b3db94:docs/research/history/b8-late-death-t1-t11.md` | Primary sources: [PANEL_DECISION_MEMO.md](../panel_2026-10-15/PANEL_DECISION_MEMO.md), [t1_commitment_study_2026-10-15.md](../t1_commitment_study_2026-10-15.md), [t3_ceiling_test_2026-10-15.md](../t3_ceiling_test_2026-10-15.md)
