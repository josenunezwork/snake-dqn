# T34 P7 statistics rules (OWNER-RATIFIED 2026-10-19 ~23:20 PT, pre-data)

Owner approved ("yes all") the rules-audit statistics changes
([Fable judgment](../advice/2026-10-19_rules_audit_fable_judgment.md) findings 3, 4, 11, 15 and "missed" items;
Astra audits `../advice/2026-10-19_rules_audit_astra_{research,ops,eng}.md`). This file fixes the statistical
frame for every T34 registration (Round 0, Path B v2, R1–R3 screens, strict gate). Each study's own
pre-registration fills in sizes, banks and seeds **before its data** and may not contradict this file.
Binding alongside [SUCCESS_CONTRACT.md](SUCCESS_CONTRACT.md) and [DESIGN_V2_SHADOW.md](DESIGN_V2_SHADOW.md).

## 1. Unit of inference follows the claim

| Claim | Unit | Notes |
|---|---|---|
| **Claim B** (fixed assisted controller over frozen B8/student) and any fixed bare checkpoint vs B8 s3 | **paired world** (candidate and B8 s3 play the same world, same hero slot, same opponents) | mix-stratified, equal weight per mix, ≥ 16 worlds per mix (governance_tiers fixed-controller rule); no training-seed population exists |
| **Claim A as a recipe** ("this distillation recipe improves B8") | independent training replicate | conditional on the B8 initializer; worlds within a replicate do not raise n |
| Continuous-respawn Watch6 product bars (contract §3) | **complete paired session** | cluster by session; never treat lives or ticks as independent |

Every bar's comparator is **released bare B8 s3** (`3e649783…`).

## 2. Seeds and banks

- **Disjoint seed sets per mix** (scripted / frozen / mixed). If a design must reuse seeds across mixes, the
  pooled variance must be covariance-aware over shared-seed clusters (T7's independent pooling is not reused).
- Fresh, collision-audited namespaces per study; strict-gate banks are never touched by development work.
- Phase 0 (running, its own approved bar +80 / LB90 > +40) speaks only to **its own premise**; it is a
  cross-check for Round 0 (bar +40 / LB90 > 0), not a substitute, and neither re-labels the other.

## 3. Bars, guards, ties

- Skill bars and product bars exactly as SUCCESS_CONTRACT §2–§3, comparator B8 s3.
- Contract "Tier 1 + Tier 2" is renamed **survival guard A / survival guard B** and both are computed **vs
  B8 s3**: guard A = survival band v2 (pooled alive-frame margin −0.05, single-mix −0.075), kept exactly as
  ratified 2026-10-06 with its per-study operating-characteristic check; guard B = scripted no-death gain ≥
  +0.03. Historical FRP3/champion comparisons are reported, not gating. "Tier 0/1/2" names are reserved for
  evidence authority (governance_tiers).
- **Tie rule (Claim A vs Claim B):** if both qualify and the paired A−B pooled-mass difference has a two-sided
  90% interval containing 0, the bare candidate (Claim A) wins.
- Multiplicity: each study pre-declares its family (arms × looks) and controls it (Holm across arms; the look
  schedule's own alpha spending across looks).

## 4. Tripwire precedence (fits and controllers)

| Class | Rule | Denominator / window | Effect |
|---|---|---|---|
| Unconditional fault | non-finite loss/Q/grad; illegal action; > 90 % one-action share | any 100k-decision window | kill that fit/arm immediately |
| Fit-drift tripwire | disagreement vs previous student > 5 % at any length decile; boost share > 1.5× previous student; held-out decisive agreement < 70 % | greedy H5000 trajectories on the 32-world probe bank | kill that fit (plan §8) |
| Delivered-controller bar | boost share within [0.5 %, 10 %] | delivered H5000 decisions, all mixes | the candidate fails the bar (not a kill) |

The older prereg-skill boost tripwire (> 2 % overall / > 1 % below L700, training window) does **not** apply to
T34 fits; the fit-drift row replaces it.

## 5. Looks

Each study chooses fixed-N or sequential by **expected cost at the +40 bar** using development variance,
recorded before data. Sequential designs must set futility compatible with +40 (not the cap-600 +50 target).
Running fixed-N studies are never converted.

## 6. Wording and isolation

- A failed bar is reported as "this teacher/controller failed its registered bar", never "B8 is the bare
  ceiling".
- Path B development sessions on the Mac must not write into the B8 s3 release-monitoring log or the
  leaderboard (STATUS.md monitoring until the 2026-10-24 check-in).
