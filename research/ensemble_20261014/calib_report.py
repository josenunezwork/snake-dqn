#!/usr/bin/env python3
"""Print the Q-calibration diagnostics of ensemble probe rows (development information only).

./venv/bin/python research/ensemble_20261014/calib_report.py FILE.jsonl [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(xs, nd=2):
    return "[" + " ".join(f"{x:.{nd}f}" for x in xs) + "]"


def main() -> int:
    for f in sys.argv[1:]:
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            cal = (r.get("diag") or {}).get("calibration") or {}
            print(f"== {Path(f).name} {r['mix']} ({r.get('student')})")
            for b, c in cal.items():
                if not c.get("decisions"):
                    continue
                k = len(c["vmax_mean"])
                off = [
                    sum(c["pair_agree"][i][j] for j in range(k) if j != i) / max(k - 1, 1)
                    for i in range(k)
                ]
                vc = [c["vmax_corr"][i][j] for i in range(k) for j in range(i + 1, k)]
                ac = [c["adv_corr"][i][j] for i in range(k) for j in range(i + 1, k)]
                print(
                    f" {b}: n={c['decisions']} unanimous={c['unanimous_frac']:.3f} "
                    f"distinct={c['mean_distinct_actions']:.3f} "
                    f"mean!=vote={c['mean_ne_vote_frac']:.4f} "
                    f"mean!=z={c['mean_ne_zmean_frac']:.4f} vote!=z={c['vote_ne_zmean_frac']:.4f}"
                )
                print(f"   vmax_mean {fmt(c['vmax_mean'], 1)} vmax_sd {fmt(c['vmax_sd'], 1)}")
                print(f"   adv_sd {fmt(c['adv_sd'], 2)} gap_mean {fmt(c['gap_mean'], 2)}")
                print(
                    f"   agree_chosen {fmt(c['member_agree_chosen'], 3)} "
                    f"pair_agree(avg/other) {fmt(off, 3)}"
                )
                if vc:
                    print(
                        f"   vmax_corr min/mean {min(vc):.3f}/{sum(vc) / len(vc):.3f} "
                        f"adv_corr min/mean {min(ac):.3f}/{sum(ac) / len(ac):.3f} "
                        f"vmax member spread {c['vmax_member_spread']:.2f} "
                        f"q(chosen) spread {c['qchosen_member_spread']:.2f}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
