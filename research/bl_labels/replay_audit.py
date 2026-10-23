"""Mac-only action-tape replay of platform-local F0 audit roots.

Inputs must be trusted factory exports whose payload and sidecars were verified
on the Mac. Pickled snapshots are internal full-state evidence, never arbitrary
external documents. Policy choices are inspected for tie classification only;
the simulator always consumes the recorded six-seat action row.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from research.bl_labels.exports import publish_json
from research.bl_served_search import ceiling
from research.bl_served_search.b2_engine import WatchDriver


def replay_root(
    record: dict[str, Any], b8: Path | None = None, frp3: Path | None = None
) -> dict[str, Any]:
    """Replay recorded actions; refuse state drift and classify local Q ties."""
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.simd_env.ego2s_policy import student_q

    if "snapshot_b64" in record:
        payload = base64.b64decode(record["snapshot_b64"], validate=True)
        if hashlib.sha256(payload).hexdigest() != record["snapshot_sha256"]:
            raise ValueError("INVALID cross-platform snapshot digest mismatch")
        driver = WatchDriver.restore(payload)
    else:
        from research.bl_labels.factory import make_driver

        if b8 is None or frp3 is None:
            raise ValueError("action-tape replay requires pinned local B8 and FRP3 inputs")
        driver = make_driver(record["initial_allocation"], b8, frp3)
    divergences: list[dict[str, Any]] = []
    checked = 0
    for row in record["tape"]:
        driver.prepare()
        if int(row["decision"]) != driver.decision:
            raise ValueError("INVALID cross-platform action tape decision gap")
        actions = np.asarray(row["actions"], dtype=np.int64)
        if actions.shape != (6,) or np.any(actions < 0) or np.any(actions > 5):
            raise ValueError("INVALID cross-platform six-seat action row")
        if bool(driver.sim.alive[0, 0]) and row.get("hero_q") is not None:
            q = student_q(driver.hero.net, driver.sim, np.array([[0, 0]]), "rowwise").numpy()[0]
            mask, local = ceiling._legal(q, driver.sim.get_advisory_action_mask()[0, 0])
            source = int(actions[0])
            if int(local) != source:
                # Measure the difference on the Mac side; a source choice outside
                # its Mac advisory mask is unexplained, even when raw Q is tied.
                scale = np.float32(max(abs(float(q[local])), abs(float(q[source]))))
                ulp = max(float(abs(np.spacing(scale))), float(np.finfo(np.float32).tiny))
                gap = abs(float(q[local]) - float(q[source]))
                explained = bool(mask[source]) and gap <= 4 * ulp
                divergences.append(
                    {
                        "decision": driver.decision,
                        "source": source,
                        "mac": int(local),
                        "mac_q_gap": gap,
                        "mac_ulp": ulp,
                        "within_four_ulp": explained,
                    }
                )
        if driver.frozen.runtime is not None:
            driver.frozen.runtime.prepare(driver.sim)
        driver.sim.step(actions.reshape(1, 6))
        driver.frozen.observe_step(driver.sim)
        driver.decision += 1
        if int(driver.sim.get_step_events()["death_cause"][0, 0]):
            driver.awaiting_respawn = True
        if gid.env_digest(driver.sim, 0) != row["env_digest"]:
            raise ValueError(f"INVALID cross-platform simulator drift at {row['decision']}")
        checked += 1
    if not checked:
        raise ValueError("cross-platform audit root has no action tape")
    return {
        "decisions": checked,
        "policy_divergences": divergences,
        "unexplained_policy_divergences": sum(not x["within_four_ulp"] for x in divergences),
    }


def audit(
    records: list[dict[str, Any]],
    *,
    require_mac: bool = True,
    b8: Path | None = None,
    frp3: Path | None = None,
) -> dict[str, Any]:
    """Select exactly 64 roots outcome-blind and require a Mac replay platform."""
    if require_mac and platform.system() != "Darwin":
        raise ValueError("F0 64-root cross-platform audit is Mac-only")
    keyed = [(str(x["root_id"]), x) for x in records]
    if len({key for key, _ in keyed}) != len(keyed):
        raise ValueError("duplicate cross-platform audit root")
    if len(keyed) < 64:
        return {"status": "INCOMPLETE", "available_roots": len(keyed), "required_roots": 64}
    selected = sorted(keyed, key=lambda pair: hashlib.sha256(pair[0].encode()).hexdigest())[:64]
    results = [
        {"root_id": key, **(replay_root(row, b8, frp3) if b8 else replay_root(row))}
        for key, row in selected
    ]
    unexplained = sum(row["unexplained_policy_divergences"] for row in results)
    return {
        "schema": "bl-labels-f0-mac-replay/v1",
        "status": "INVALID" if unexplained else "PASS",
        "platform": platform.platform(),
        "roots": results,
        "root_count": 64,
        "scope": "Recorded-action simulator replay; platform-local training labels only",
    }


def main() -> None:
    """Read only verified immutable evidence and write a create-only receipt."""
    from research.bl_labels.exports import _verify

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--frp3", type=Path, required=True)
    args = parser.parse_args()
    records: list[dict[str, Any]] = []
    for path in args.unit:
        _verify(path)
        value = json.loads(path.read_text())
        if value.get("status") != "COMPLETE":
            raise ValueError("cross-platform audit refuses incomplete or invalid evidence")
        for record in value.get("audit_roots", []):
            if "tape_end" in record:
                record = {**record, "tape": value["audit_tape"][: int(record["tape_end"])]}
            records.append(record)
    result = audit(records, b8=args.b8, frp3=args.frp3)
    publish_json(args.out, result)
    print(
        json.dumps({key: value for key, value in result.items() if key != "roots"}, sort_keys=True)
    )
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
