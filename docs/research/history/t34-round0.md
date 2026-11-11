# T34 Round 0 Sparse-Teacher (`teacher_v2`) & Round 0-D Dense-Teacher Premise Tests (`t34-round0`)

**Dates:** 2026-10-20 to 2026-10-21
**Verdict:** Closed `KILL / INTEGRITY_FAILURE` pre-look — all `240` paired H5000 worlds completed and passed `10%` (`24`-world) Mac replay, but `analyze` was invoked before publishing the replay ledger append, sealing all `240` pair outcomes unread and leaving fallback `t34_round0d` (`DRAFT`) unactivated.

## What was tried
- Pre-registered `t34_round0`: a `240`-pair H5000 test (`96` scripted, `72` frozen, `72` mixed in `100/100/40` L4 chunks, `32.64 u` initial / `65.28 u` max liability) of deterministic `teacher_v2` (`H = 300`, shadow `D = 300`, bisection + radius `3`) vs bare `b8-s3`.
- Set GO bars requiring scripted survival `>= 0.66` (`>= +0.10` gain), equal-mix mass gain `>= +40` (`LB90 > 0`), survival guard A (`LB95 > -0.05` pooled, `> -0.075` per mix, `>= 0.30` absolute), and a pre-look amended relative boost floor (`0.5 * bare_share <= teacher_share <= 0.10`).
- Ran `20,000`-rep operating-characteristic (`OC`) simulations across `5` historical/normal models, verified 3-way physical worker parity (`09d00b7a304d861f880d52e14c8ad9d22567770667ae6991574586110fd12917`), and attested mechanics v2 (`ATTESTED_MECHANICS_V2`).
- Prepared contingent `t34_round0d` (`DenseTrigger(600, 5)`, `272` disjoint seeds, `120 u` ceiling, `89/89` tests passing) in case Round 0 returned a scientific `STOP`/`INCONCLUSIVE`.

## What happened
- **Operating characteristics & sizing:** Protected-regression joint Type-I error stayed `0.00750–0.03670 <= 0.06` across all `5` models; joint mass+survival pass at true `+40` was `0.23945` / `0.26175` (mass-only `0.93195` / `0.92905` at `+60`), and `A5` L4 ran at `57.173` and `54.372` worlds/h/lane (`>= 50` bar).
- **Pre-look boost amendment (`30b2739a`):** Replaced the absolute `[0.005, 0.10]` (`0.5%`) boost floor after measuring bare `b8-s3`'s natural boost share at `0.12%` over `190,779` decisions.
- **Terminal settlement (`KILL / INTEGRITY_FAILURE`):** All `240` pairs sealed `COMPLETE` and `24` Mac replay worlds passed, but `analyze` ran before the replay's ledger append was published to remote history; `require_published_history` raised `HistoryError` before reading any pair file, sealing all `240` outcomes unread and leaving `t34_round0d` unactivated.

## Why
`analyze` enforced a strict remote-git publication check (`require_published_history`) as its first step and mapped `HistoryError` directly to terminal `KILL` when the operator ran `analyze` before pushing the replay ledger commit.

## What to reuse / what not to repeat
- Calibrate behavioral guard floors against the parent checkpoint's measured rate (`b8-s3` boosts on `0.12%` of decisions, below a naive `0.5%` absolute floor).
- Automate ledger publication inside the phase script rather than relying on manual operator `git push` ordering between replay and `analyze`, and never map a pre-look git check to an unrecoverable study `KILL`.

## Full record
Original: `git show 90b3db94:docs/research/history/t34-round0.md` | Primary sources: [PREREGISTRATION_ROUND0.md](../../../research/t34_round0/PREREGISTRATION_ROUND0.md), [t34_round0/README.md](../../../research/t34_round0/README.md), [VERIFICATION_ROUND0D.md](../../../research/t34_round0d/VERIFICATION_ROUND0D.md)
