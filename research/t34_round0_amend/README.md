# `t34_round0_amend`

**Purpose:** Pre-look T34 Round 0 analysis amendment replacing the absolute 0.5% delivered-boost floor with the relative `0.5 x bare_share <= teacher_share <= 0.10` clause.

**Status / Outcome:** Ratified pre-look (`30b2739a`, 2026-10-20), then **closed unexecuted on pair outcomes** when Round 0 halted pre-look with `KILL / INTEGRITY_FAILURE` on unpublished replay-ledger ordering (2026-10-21).

## Key Files & Results

- **Package entrypoints / artifacts:** [`AMENDMENT_BOOST_FLOOR.json`](AMENDMENT_BOOST_FLOOR.json), [`rule.py`](rule.py), [`decide.py`](decide.py), [`ledger/t34-round0-study-01.jsonl`](ledger/t34-round0-study-01.jsonl)
- **Related docs / results:** [`docs/notes/t34-campaign.md`](../../docs/notes/t34-campaign.md), [`docs/research/final_run_20261019/CAMPAIGN_LOG.md`](../../docs/research/final_run_20261019/CAMPAIGN_LOG.md)
