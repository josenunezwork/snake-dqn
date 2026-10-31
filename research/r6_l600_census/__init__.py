"""R6 Watch6-v2 L<600 safe-harvest separability census package.

Implements OPUS_META §3 row 4 / review-6 §3 / research-t6-3 §3-4:
- 64 fresh Watch6-v2 worlds (6,000 frames, 5x frp3-s12+v8, mechanics v2, respawn).
- 6 deduplicated L<600 roots per world (384 roots), evaluating all <=6 advisory-legal
  1-step H=500 continuations with K-1 native v8 and ProvenanceObserver accounting.
- 5-fold world-disjoint OOF separability probes comparing ego2s-bv8 vs ego2s-bv8
  augmented with +2 ego-rotated 37x37 FIFO-age-weighted corpse/trail planes and
  4 FIFO/cluster/race scalars against R6's pre-declared kill criteria.
"""

from __future__ import annotations

NAMESPACE = "t34-r6-watch6-v2-l600-census-20261031"
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
FRP3_SHA256 = "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"

R6_KILL_THRESHOLDS: dict[str, float] = {
    "min_qualifying_delta_A500": 5.0,
    "min_oof_qualifying_root_recall": 0.15,
    "min_mean_selected_delta_A500": 8.0,
    "max_H500_lethal_flip_rate": 0.005,
}
