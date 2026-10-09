"""Turn ``bench_throughput.py`` JSON into the markdown tables used in the scope doc."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from statistics import median


def main(path: str) -> int:
    data = json.load(open(path))
    cells = defaultdict(list)
    feats = {}
    for c in data["cells"]:
        key = (c["scenario"], c["engine"], c["envs"])
        cells[key].append(c)
        for k, v in c.items():
            if k.startswith("feat_") and k.endswith("_us_per_agent"):
                feats[(c["scenario"], k[5:-13])] = v
    print("| scenario | engine | envs | env-frames/s | agent-steps/s | mean len (end) |")
    print("|---|---|---|---|---|---|")
    for (scen, eng, n), rows in sorted(
        cells.items(), key=lambda kv: (kv[0][0] != "fresh",) + kv[0][1:]
    ):
        fps = median(r["env_frames_per_s"] for r in rows)
        aps = median(r["agent_steps_per_s"] for r in rows)
        ln = median(r["mean_len_end"] for r in rows)
        print(f"| {scen} | {eng} | {n} | {fps:,.0f} | {aps:,.0f} | {ln:,.0f} |")
    print()
    print("| scenario | featurizer | us/agent | agents/s/core |")
    print("|---|---|---|---|")
    for (scen, name), us in sorted(feats.items()):
        print(f"| {scen} | {name} | {us:.1f} | {1e6 / us:,.0f} |")
    print()
    print("verify_big_world:", json.dumps(data["verify_big_world"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
