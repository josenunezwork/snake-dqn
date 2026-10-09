"""Study-side verification of the M3 Phase R shard start markers and records (stdlib only).

Complements ``research/sequential_phase_r/audit.py`` (which does not read start markers).
Run with ``python -I``::

  python -I research/redesign_m3_20261008/phase_r_verify.py --root R [--out verify.json]

For every ``shards/look-<k>``: ``start.json``'s binding equals the look gate's (look k, the
plan file / plan sha256, the previous receipt's file sha256, the world index range), its commit
is the plan's, it started after the previous look's analysis (``analysed_utc``); every record
carries that binding and commit, a unit id of the marker, its look, a world of its seed's bank
inside the look's index range (controls: index 0), its roster and hero checkpoint sha256; every
planned unit's records are present; no shard exists past the last receipt's look (and none past
a STOP). Exit 0 PASS, 1 FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> dict:
    problems = []
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text())
    sizes = plan["plan"]["look_sizes"]
    banks = {int(s): [int(w) for w in v] for s, v in plan["banks"].items()}
    commit = plan["study"]["commit"]
    receipts = sorted((root / "looks").glob("look-*.json"), key=lambda p: int(p.stem.split("-")[1]))
    last = int(receipts[-1].stem.split("-")[1]) if receipts else -1
    shards = sorted((root / "shards").glob("look-*"), key=lambda p: int(p.name.split("-")[1]))
    checked = 0
    for sd in shards:
        k = int(sd.name.split("-")[1])
        if k > last:
            problems.append(f"{sd.name}: shard past the last receipt (look {last})")
        start = json.loads((sd / "start.json").read_text())
        b = start["binding"]
        prev = root / "looks" / f"look-{k - 1}.json"
        want = {
            "look": k,
            "plan_file_sha256": sha256_file(plan_path),
            "plan_sha256": plan["plan_sha256"],
            "previous_receipt_sha256": sha256_file(prev) if k > 0 else None,
            "world_index_range": [0 if k == 0 else sizes[k - 1], sizes[k]],
        }
        if b != want:
            problems.append(f"{sd.name}: binding {b} != look gate {want}")
        if start["commit"] != commit:
            problems.append(f"{sd.name}: commit {start['commit']} != plan commit {commit}")
        if k > 0:
            prev_receipt = json.loads(prev.read_text())
            if prev_receipt.get("action") != "CONTINUE":
                problems.append(f"{sd.name}: previous look ended the study")
            analysed = (prev_receipt.get("extra") or {}).get("analysed_utc")
            if analysed is None or datetime.fromisoformat(
                start["started_utc"]
            ) <= datetime.fromisoformat(analysed):
                problems.append(f"{sd.name}: started before look {k - 1} was analysed")
        units = set(start["units"])
        seen_units = {}
        lo, hi = want["world_index_range"]
        for path in sorted((sd / "records").glob("*.json")):
            r = json.loads(path.read_text())
            checked += 1
            tag = f"{sd.name}/{path.name}"
            if r.get("binding") != b or r.get("commit") != commit or r.get("look") != k:
                problems.append(f"{tag}: binding / commit / look mismatch")
            if r.get("unit_id") not in units:
                problems.append(f"{tag}: unit {r.get('unit_id')} not in the start marker")
            seen_units.setdefault(r.get("unit_id"), set()).add(int(r["world_seed"]))
            bank = banks[int(r["seed"])]
            idx = bank.index(int(r["world_seed"])) if int(r["world_seed"]) in bank else -1
            if r.get("world_index") != idx:
                problems.append(f"{tag}: world_index {r.get('world_index')} != bank index {idx}")
            if r.get("control"):
                if idx != 0 or k != 0 or r.get("horizon") != 5000:
                    problems.append(f"{tag}: control not world 0 / look 0 / H5000")
            elif not (lo <= idx < hi) or r.get("horizon") != 10000 or not r.get("prefix_h5000"):
                problems.append(f"{tag}: decision record outside the look / not nested H10000")
            ck = plan["study"].get("checkpoints") or {}
            hero_key = (
                f"{r['hero']}|s{int(r['seed'])}"
                if f"{r['hero']}|s{int(r['seed'])}" in ck
                else r["hero"]
            )
            want_sha = (ck.get(hero_key) or {}).get("sha256")
            if not want_sha or (r.get("hero_checkpoint") or {}).get("sha256") != want_sha:
                problems.append(f"{tag}: hero checkpoint sha256 is not the plan's for {hero_key}")
            roster = r.get("roster_member_sha256s") or []
            identity = json.dumps((r.get("record") or {}).get("world_identity"), sort_keys=True)
            if not roster or any(sha not in identity for sha in roster):
                problems.append(f"{tag}: roster sha256s missing or not in the world identity")
        planned = start.get("unit_worlds")
        if planned is None:
            problems.append(f"{sd.name}: start marker lacks unit_worlds")
            planned = {}
        for uid in units:
            if set(planned.get(uid, [])) != seen_units.get(uid, set()) or uid not in seen_units:
                problems.append(f"{sd.name}: unit {uid} records do not cover its planned worlds")
    return {
        "schema": "redesign-m3-phase-r-verify/v1",
        "root": str(root),
        "shards": [p.name for p in shards],
        "records_checked": checked,
        "problems": problems[:50],
        "n_problems": len(problems),
        "verdict": "PASS" if not problems else "FAIL",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = verify(args.root)
    text = json.dumps(out, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
