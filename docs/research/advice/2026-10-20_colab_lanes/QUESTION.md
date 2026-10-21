# Colab lanes question — 2026-10-20 ~20:10 PT

## Goal
A strictly-gated better served snake than bare B8 s3 (served since 2026-10-17), as soon as possible — AND, per the
owner's standing rule, spend the prepaid Colab Pro+ units (~1,154 u left) by ~2026-11-20 (~38 u/day) on lanes that
produce DECISIONS. Today's pace is ~10 u/day and the planned lanes run out ~Oct 25; ~80–85% of the balance has no lane
(`docs/research/ops/COLAB_UNITS_LEDGER.md`).

## Open question
Given everything below, what lanes should the remaining ~900–1,050 Colab units fund between now and ~Nov 20 so that each
one produces a decision that moves us toward a better served snake — and in what order, starting when?

## Facts (verify in the repo; cite file:line)
- Campaign state & rules: `docs/STATUS.md`, `docs/research/final_run_20261019/{HANDOFF_2026-10-20.md (START HERE + updates),
  CAMPAIGN_LOG.md, SUCCESS_CONTRACT.md (+§5 amendment), DESIGN_V2_SHADOW.md, P7_STATISTICS.md, FINAL_PLAN.md}`.
- Tonight's results: A100s don't help (host-bound; `docs/research/advice/2026-10-20_a100/SYNTHESIS.md`, gpu_expert_*.md);
  T35 outcome-head probe KILL (`research/t35_outcome_head/RESULT.md`); T35 two-deviation rescue probe pooled KILL
  (`research/t35_two_deviation/RESULT_EXTEND.md`); Claim A distillation blocked: frozen loss flips 0.4% of decisive rows and
  the 1:1 row rule gives ~1 update/epoch (`snake-dqn-artifacts/fable-panel-claima/design.md`); boost-share floor changed to a
  relative floor (owner) — Round 0 amendment `research/t34_round0_amend/`.
- In flight: Round 0 (teacher acting vs bare B8, 240 pairs, Colab L4, chunks 000 done/001 running/002 next; verdict ~Thu
  10-22). Path B paced retention (Mac-only, ~18–19 h after Phase 0 ends Wed ~10:00; re-ratification in progress) — the
  fastest route to a better served snake (serving the search-assisted hero). Phase 0 (Mac) ends Wed.
  NO-GO plan: `docs/research/specs/2026-10-20_t34_nogo_branch_plan.md`; Round 0-D draft (dense T3 teacher, ~27–68 u,
  reviews in `snake-dqn-artifacts/fable-panel-round0d/`).
- Compute facts: work is host-bound CPython (B8 forward ~8–13%); L4 runtime = 12 vCPU, ~1.5–2 u/h; A100 same host, ~3×
  price; G4 = 48 vCPU, ~8.7 u/h (unmeasured for us); max 3 concurrent runtimes; ~0.036 u per teacher world (A5);
  qualified CLI supervisor `research/colab_cli`. Strict gates and serving qualification stay on the Mac.
- Compute portfolio adopted 10-20 (lanes L0–L5): `docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md`;
  ledger `docs/research/ops/COLAB_UNITS_LEDGER.md`; research index `research/INDEX.md`.

## Rules for your answer
Read-only (no edits/commits/jobs). Put the full answer in your final message: concrete lanes, each with the decision it
produces, unit cap, runtime type, earliest start, dependencies (Round 0 GO/NO-GO, Path B result), and kill rule; label
verified vs estimated claims with file:line. Do not list options for someone else to pick — make your recommendation.
