# `speed_s0_baseline`

**Purpose:** Speed program S0 matched baseline benchmark for the unmodified T34 ExIt-B8 v2 teacher collector (`teacher_v2.run_worlds`) across three opponent mixes and short/long body regimes.

**Status / Outcome:** Completed diagnostic baseline (2026-10-20): contended M5 Pro Mac measured `103.1 s` per 6-world round (`56.8%` opponent v8 veto, `13.5%` inference, `11.9%` digest), motivating S1 and S2.

## Key Files & Results

- **Package entrypoints / artifacts:** [`RESULT_MAC_CONTENDED_01.md`](RESULT_MAC_CONTENDED_01.md), [`REFERENCE_MAC.json`](REFERENCE_MAC.json), [`harness.py`](harness.py), [`s0_colab.sh`](s0_colab.sh)
- **Related docs / results:** [`docs/notes/gpu-engine-and-speed.md`](../../docs/notes/gpu-engine-and-speed.md), [`docs/notes/t34-campaign.md`](../../docs/notes/t34-campaign.md)
