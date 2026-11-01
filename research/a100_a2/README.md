# `a100_a2`

**Purpose:** A2 historical-label model-scale and target falsifier comparing frozen-B8 linear (`c0`), `1.64M` (`S`), and `6.68M` (`M`) encoders with single-horizon D300 vs discrete-hazard heads across 25%/50%/100% parent-world doses on T5/T8 and excluded A5 tables.

**Status / Outcome:** Completed single look (2026-10-22): **`scale = NOT_CLOSED_EXPLORATORY`**, **`target = KEEP_SINGLE_HORIZON`** (T5 D300 AUC @100%: frozen-B8 linear `0.767` > `s_b8 0.719` > `m_random 0.711` ~ `s_random 0.706`; larger model gave no gain while data volume added `+0.02–0.05` AUC; exploratory only, no serving claim).

## Key Files & Results

- **Package entrypoints / artifacts:** [`PREREGISTRATION.md`](PREREGISTRATION.md), [`SEMANTICS_ERRATUM.md`](SEMANTICS_ERRATUM.md), [`HANDOFF.md`](HANDOFF.md), [`HANDOFF_ACCOUNTING_20261022.md`](HANDOFF_ACCOUNTING_20261022.md), [`model.py`](model.py), [`job.py`](job.py), [`aggregate.py`](aggregate.py)
- **Related docs / results:** [`docs/research/final_run_20261019/CAMPAIGN_LOG.md`](../../docs/research/final_run_20261019/CAMPAIGN_LOG.md), [`docs/notes/big-lever-program.md`](../../docs/notes/big-lever-program.md)
