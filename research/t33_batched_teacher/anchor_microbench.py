"""Three-round random-safe unique-hash benchmark against frozen per-row hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy

BASE = "f76820776e39a6a53c13197f6f0948aa7c6a95f4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "show", f"{BASE}:src/simd_env/fast_anchors.py"], text=True
    )
    namespace = {"__name__": "t33_frozen_anchor", "__file__": __file__}
    exec(compile(source.replace("cache=True", "cache=False"), "<frozen anchor>", "exec"), namespace)
    rows = 12288
    seeds = [i // 48 for i in range(rows)]
    sim = SimpleNamespace(frame=np.full(rows, 701, dtype=np.int64))
    slots = np.column_stack((np.arange(rows), np.ones(rows, dtype=np.int64)))
    masks = np.random.default_rng(17).random((rows, 6)) < 0.5
    old = namespace["FastProfileAnchorSimdPolicy"]("random_safe", seeds)
    new = FastProfileAnchorSimdPolicy("random_safe", seeds)
    assert old.actions(masks, sim, slots).tobytes() == new.actions(masks, sim, slots).tobytes()
    result = {"before": [], "after": []}
    for _ in range(3):
        for label, policy in (("before", old), ("after", new)):
            start = time.perf_counter()
            for _repeat in range(10):
                policy.actions(masks, sim, slots)
            result[label].append(rows * 10 / (time.perf_counter() - start))
    receipt = {
        "baseline_commit": BASE,
        "baseline_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "candidate_sha256": hashlib.sha256(
            Path("src/simd_env/fast_anchors.py").read_bytes()
        ).hexdigest(),
        "rows": rows,
        "unique_seed_slot_frame_triples": rows // 48,
        "rows_per_s": result,
        "rounds": 3,
        "native_threads": 1,
        "note": "observed under concurrent local CPU profile; same-host interleaved arms",
    }
    with args.out.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
