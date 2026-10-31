"""R7 macro-option fallback package on 1-step-all-die C/T TRAIN roots only.

Implements OPUS_META §3 row 5 / review-7 §3 / research-t7-1 §1-2:
- Six pre-data frozen deterministic spatial options ``{O_0..O_5}`` (``K=15`` ticks)
  with exact pending-growth (``seg_count < length``) and boost-burn action legality
  and NO ``!spacious`` interrupt.
- Strict ``exact_stop`` gating: emits 1-step ``exact_stop`` whenever any 1-step H300
  continuation survives on selection suffix 0 (0 ordinary flips); invokes pruned
  depth-2 ``K=15`` macro-search (6 Ply-1 -> top-3 Ply-1 x 6 Ply-2 + 270-tick B8
  continuation on CRN suffix 0, scored on independent suffixes 4..7) ONLY on
  1-step-all-die C/T ``train`` roots (strictly ``assessment_roots_used == 0``).
"""

from __future__ import annotations

NAMESPACE = "t34-r7-macro-fallback-ct-train-20261031"
ALLOCATION_NAME = "ct-train-all-die-batches"

B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
FRP3_SHA256 = "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"
CHAMPION_SHA256 = "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93"

CT_BANK_SHA256 = "112a2aa4c369ecd0e0f735db410e9ac5a7d330aca09846a50d1a734e6b3175f0"
CT_LABEL_SHA256S: tuple[str, ...] = (
    "3b86db00fdbe11e7c20b82992599599a4265bebf47a68aa94be9b84fccb6162c",
    "44ef1bdf31274585eab2f6130e1913faadbf72f07b16d360a54c23c5e8bc1277",
    "6684e30e752ebbee00c302dd308750edacae648bf96ffc7e96b5704655076186",
    "1b2387507412ced28db804bb35c460d7a8e3a9dd62215691c34943768a1d26e3",
    "6fc27dbd38f1da9d03411e1ca1c61d067663c6d32daf2af20b9f1fdf2807a689",
    "d5f49020006e7b3f07d78ea6cb8b7263dba396714615dbcaf08270913dbd4da1",
    "e28d8637b9cd39501523984d94f7b1588905fa6b50998f2298d766b6aaa707eb",
    "cce958ae9a90b19d4e77b5fa471d5b09f8f08e269baed592f63b0512e2937d0b",
)

OPTION_NAMES: tuple[str, ...] = (
    "O0_delay_coil",
    "O1_gate_breach",
    "O2_open_voronoi",
    "O3_corpse_forage",
    "O4_boost_evac",
    "O5_voronoi_cutoff",
)

MACRO_DURATION_TICKS = 15
BRANCH_HORIZON_TICKS = 300

R7_KILL_THRESHOLDS: dict[str, float | int] = {
    "min_independent_suffix_H300_rescue_L800_all_die": 0.25,
    "min_total_death_root_rescue_lift_over_exact_stop_pp": 10.0,
    "max_newly_caused_ordinary_lethal_flips": 0,
    "max_mac_1thread_tracker_p95_ms_per_tick": 1.0,
    "min_realized_nonfatal_macro_duration_ticks": 10.0,
}
