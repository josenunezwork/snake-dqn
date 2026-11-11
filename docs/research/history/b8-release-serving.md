# B8 Seed 3 Strict Gate (T7), Serving Qualification & Release (T12/T12b/T12c), Product Roster Audits (T13/T13b), and Mixed Watch Qualification & Default Switch (T15/T15b)

**Dates:** 2026-10-16 to 2026-10-17
**Verdict:** Bare `b8-s3` (`3e649783`, `veto_variant: none`) passed T7 `STRICT_PASS`, cleared Mac serving qualification (`T12b` `SERVING_PASS`) to ship as default live policy (`T12c`), and switched Watch's default roster to `mixed` (`b8-s3` hero vs `frp3-s12 + v8` opponents) after `T13`/`T13b` roster audits and `T15` `SERVING_PASS` (`T15b`).

## What was tried
- Ran one-look Mac CPU strict gate `T7` (`5,760` H10000 episodes + `45` H5000 prefix controls across `960` fresh worlds) comparing the `5`-seed bare `B8` family against `frp3-s12 + v8` (Tier 1) and `champion_a5 + v8` (Tier 2).
- Ran 8-check Mac serving qualification `T12` (`73` fresh seeds) and Check-4 GC/speed-ceiling repair `T12b` (`24 fps` for `<= 6` driven AI, `12 fps` for `7..12`) before flipping `CHECKPOINT_RELEASED_DEFAULT = "b8-s3"` (`T12c`).
- Audited continuous-respawn H5000 Watch rosters (`25` paired fresh worlds/cell) at `12` snakes (`T13`) and at the `6`-snake product default + `8`-snake `all_b8` (`T13b`) across `all_b8`, `all_frp3` (hero-only `+v8`), `all_champion` (hero-only `+v8`), and `mixed` (`b8-s3` hero slot `0` vs `5` or `11` `frp3-s12 + v8` opponents).
- Qualified `SNAKE_SERVE_WATCH_ROSTER=mixed` across 6 Mac serving checks (`T15`, `33` fresh seeds) and switched Watch's default roster to `mixed` (`T15b`).

## What happened
- **`T7` (`STRICT_PASS`) & `T12b`/`T12c` (`RELEASED`):** Pooled `B8` beat `frp3-s12 + v8` by `+138.666306` H5000 mass (`LB90 +129.477718`, `5/5` seeds `+`; Tier 2 survival `-0.015625 >= -0.05`), selecting `b8-s3` (`+149.650051`). Clamping `7..12`-AI speed to `12 fps` fixed `T12`'s `1/120` `Watch12` GC overrun (`0 / 120` overruns; `Watch6` `5.967 / 8.048` ms, `Play6` `4.589 / 5.790` ms).
- **Red-Queen arena effect (`T13` & `T13b`):** Legacy `all_frp3` wrapped only hero slot `0` in `v8`, so `5` or `11` unwrapped foes churned at `2.109` arena deaths/1k (`12` snakes; `1.159`/1k at `6` snakes; `300` hero kills in `T13`).
- **Homogeneous `all_b8` vs `all_frp3`:** `all_b8` cut arena deaths `65–66%` (`0.712` and `0.404`/1k) and raised arena mass `+53%` (`1646.70` vs `1072.01`) to `+61%` (`1324.33` vs `824.57`), while making hero scores harder (`6`-snake hero p95 length `596` vs `723`, `-127` outside `100` -> `PLAN_MIXED_PENDING_ROSTER_QUALIFICATION`).
- **Mixed Watch (`T13b` -> `T15` `SERVING_PASS` -> `T15b` `RELEASED`):** At `6` snakes (`T13b`), `mixed` cut bare `b8-s3` hero deaths/1k from **`0.392` to `0.296`** (`+0.096`/1k gap `<= 0.10`; **`0.272`/1k** in `T15`, p50/p95 length `248`/`841`) and lifted hero H5000 mass from **`236.91` to `316.76`** (`Watch6` update p50/p95 `6.027 / 7.792` ms < `11.688 ms` trigger, `0 / 120` overruns).

## Why
Legacy Watch inflated hero stats because `5` or `11` opponents ran unwrapped without `v8`, whereas `6` `b8-s3` clones created much larger, longer-lived opponents (Red-Queen effect); `mixed` restored balanced exhibition opponents (`frp3-s12 + v8`) while keeping hero slot `0` completely bare, and `Watch12` `24 fps` overruns were `27.8–42.9 ms` Python GC pauses rather than `5.14 ms` neural forwards.

## What to reuse / what not to repeat
- Never compare single-hero Watch scores across changing opponent populations (Red-Queen effect); evaluate against fixed opponent mixes (`T7` / `mixed` roster).
- Enforce server-side fps ceilings (`24 fps` for `<= 6` AI, `12 fps` for `7..12` AI) so `27–43 ms` Python GC tails during `12`-snake snapshot serialization never cause paced tick overruns.

## Full record
Original: `git show 90b3db94:docs/research/history/b8-release-serving.md` | Primary sources: [t12_b8_serving_qualification_2026-10-17.md](../t12_b8_serving_qualification_2026-10-17.md), [b8_s3_release_2026-10-17.md](../b8_s3_release_2026-10-17.md), [t15_mixed_roster_qualification_2026-10-17.md](../t15_mixed_roster_qualification_2026-10-17.md)
