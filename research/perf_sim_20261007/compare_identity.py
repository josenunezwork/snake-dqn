"""Compare two identity-harness outputs (reference vs candidate) and report byte identity.

Eval files (``episode_harness --mode identity``): per (mix, world seed) the per-frame world
digest chain (every snake's segments / direction / length / alive / boost state / chosen
action, the food list, the frame counter, Python and NumPy RNG states), the network digest
(every ApexNetwork forward: input and output float32 bytes, in call order), the canonical
rollout record and the veto diagnostics with wall-clock fields (``*seconds*``) removed.

Actor files (``actor_harness --mode identity``): per episode the digest of every row sent to
the buffer (states, next states, actions, rewards, dones, bootstrap steps, masks, mask modes,
priorities), the action sequence digest, plus the world / network digests as above.

Exit 0 only if every compared item is identical and both sides cover the same episodes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _strip_seconds(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_seconds(v) for k, v in value.items() if "seconds" not in k}
    if isinstance(value, list):
        return [_strip_seconds(v) for v in value]
    return value


def _load(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _key(row: Dict[str, Any]) -> Any:
    if "mix" in row and "world_seed" in row:
        return (row.get("episode"), row["mix"], row["world_seed"])
    return ("train", row.get("episodes"))


def first_divergence(a: List[str], b: List[str]) -> Any:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i + 1
    return None if len(a) == len(b) else min(len(a), len(b)) + 1


def compare(ref: Path, cand: Path) -> Dict[str, Any]:
    left = {_key(r): r for r in _load(ref)}
    right = {_key(r): r for r in _load(cand)}
    problems: List[str] = []
    if set(left) != set(right):
        problems.append(f"episode sets differ: {sorted(set(left) ^ set(right), key=str)[:5]}")
    checked = 0
    frames = 0
    q_rows = 0
    sent_rows = 0
    for key in sorted(set(left) & set(right), key=str):
        a, b = left[key], right[key]
        checked += 1
        ta, tb = a.get("trace", {}), b.get("trace", {})
        frames += int(ta.get("frames", 0))
        for side, t in (("reference", ta), ("candidate", tb)):
            if int(t.get("non_int_coords", 0)):
                problems.append(f"{key}: {side} saw {t['non_int_coords']} non-int coordinates")
        q_rows += int(ta.get("q_rows", 0))
        for field in ("world_chain", "q_digest", "q_rows", "frames"):
            if ta.get(field) != tb.get(field):
                where = first_divergence(ta.get("frame_digests", []), tb.get("frame_digests", []))
                problems.append(f"{key}: trace.{field} differs (first divergent frame {where})")
        if "record_canonical" in a or "record_canonical" in b:
            if a.get("record_canonical") != b.get("record_canonical"):
                problems.append(f"{key}: rollout record differs")
            da = _strip_seconds(json.loads(a.get("veto_diagnostics_canonical", "null")))
            db = _strip_seconds(json.loads(b.get("veto_diagnostics_canonical", "null")))
            if da != db:
                problems.append(f"{key}: veto diagnostics (timing removed) differ")
        if "transitions" in a or "transitions" in b:
            sent_rows += int(a.get("transitions", {}).get("rows", 0))
            if a.get("transitions") != b.get("transitions"):
                ta_, tb_ = a.get("transitions"), b.get("transitions")
                problems.append(f"{key}: sent transitions differ {ta_} vs {tb_}")
            for field in ("steps", "agent_transitions"):
                if a.get(field) != b.get(field):
                    problems.append(f"{key}: {field} differs")
    return {
        "reference": str(ref),
        "candidate": str(cand),
        "episodes_compared": checked,
        "frames_compared": frames,
        "network_rows_compared": q_rows,
        "buffer_rows_compared": sent_rows,
        "identical": not problems,
        "problems": problems[:50],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="+", help="REF=CAND pairs of jsonl files")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    results = []
    for pair in args.pairs:
        ref, cand = pair.split("=", 1)
        results.append(compare(Path(ref), Path(cand)))
    ok = all(r["identical"] and r["episodes_compared"] > 0 for r in results)
    out = {"all_identical": ok, "results": results}
    text = json.dumps(out, indent=2)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
