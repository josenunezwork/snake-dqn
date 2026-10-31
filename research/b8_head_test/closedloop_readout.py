"""Mac receipt-bound ledger and paired readout. Never use remote outcome summaries."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from research.b8_head_test.closedloop import (
    ARMS,
    ledger,
    statistics,
    validate_ledger,
    verified,
)
from research.doom_auc_smoke.evidence import atomic, digest, json_once


def build_ledger(outputs: list[Path], identity: str) -> tuple[dict, dict]:
    """Outcome-blind union uses marker/path/size/hash and verified stop only."""
    result, locations = ledger(identity), {}
    expected = {f"batch-{i:04d}" for i in range(3)}
    for output in outputs:
        receipt = json.loads((output / "receipt.json").read_text())
        manifest_path = output / "manifest.json"
        if (
            receipt.get("stop_verified") is not True
            or receipt.get("cleanup_error")
            or digest(manifest_path) != receipt.get("manifest_sha256")
        ):
            raise ValueError("verified stopped manifest-bound receipt required")
        manifest = json.loads(manifest_path.read_text())
        if (
            receipt.get("commit") != manifest["commit"]
            or manifest["env"].get("B8_HEAD_MODE") != "main"
        ):
            raise ValueError("foreign receipt or mode")
        exports = {r["path"]: r for r in receipt["exports"]}
        if len(exports) != len(receipt["exports"]):
            raise ValueError("duplicate export paths")
        root = output / "exports/closedloop"
        observed = set()
        for marker in sorted(root.glob("units/*.json.complete")):
            meta = json.loads(marker.read_text())
            row = {k: meta[k] for k in ("unit", "path", "size", "sha256")}
            validate_ledger(dict(ledger(identity), units=[row]), identity, expected)
            if row["unit"] in locations:
                raise ValueError("duplicate immutable unit across attempts")
            path = root / row["path"]
            if (
                path.is_symlink()
                or path.stat().st_size != row["size"]
                or digest(path) != row["sha256"]
                or Path(str(path) + ".sha256").read_text().strip() != row["sha256"]
                or meta != dict(ledger(identity), **row)
            ):
                raise ValueError("unit publication differs")
            bound = exports.get("closedloop/" + row["path"], {})
            if bound.get("size") != row["size"] or bound.get("sha256") != row["sha256"]:
                raise ValueError("unit not supervisor-bound")
            observed.add("closedloop/" + row["path"])
            result["units"].append(row)
            locations[row["unit"]] = root
        if observed != {
            p for p in exports if p.startswith("closedloop/units/") and p.endswith(".json")
        }:
            raise ValueError("unmarked receipt units")
    validate_ledger(result, identity, expected)
    return result, locations


def readout(outputs: list[Path], input_dir: Path, destination: Path) -> None:
    if platform.system() != "Darwin":
        raise ValueError("endpoint readout is Mac only")
    config = json.loads((input_dir / "config.json").read_text())
    identity = config["identity"] + "/main"
    value, locations = build_ledger(outputs, identity)
    if {r["unit"] for r in value["units"]} != {f"batch-{i:04d}" for i in range(3)}:
        raise ValueError("incomplete endpoint units; no result")
    rows = []
    for row in value["units"]:
        payload = verified(locations[row["unit"]], row, identity)
        index = int(row["unit"].split("-")[-1])
        expected = {(w, a) for w in range(index * 64, (index + 1) * 64) for a in ARMS}
        if {(r["world_id"], r["arm"]) for r in payload["rows"]} != expected:
            raise ValueError("foreign parent allocation")
        rows.extend(payload["rows"])
    result = statistics(rows)
    destination.mkdir(parents=True, exist_ok=False)
    json_once(destination / "verified-ledger.json", value)
    json_once(destination / "result.json", result)
    text = (
        "# B8 advantage-head closed-loop development screen\n\n"
        "Exploration only; no release evidence. "
    )
    text += (
        "One all-2,048-root head, fixed 3,000-update recipe. "
        "192 paired terminal-hero v1 H5000 worlds.\n\n"
        "| Arm | Delta mass | LB90 | CI95 | Deaths | Alive fraction | Override rate |\n"
        "|---|---:|---:|---|---:|---:|---:|\n"
    )
    for arm in ARMS:
        r = result[arm]["all"]
        text += (
            f"| {arm} | {r['mass_delta']:.3f} | {r['lb90']:.3f} | "
            f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | {r['deaths']} | "
            f"{r['alive_fraction']:.4f} | {r['override_rate']:.2%} |\n"
        )
    text += (
        "\n10,000 paired-parent bootstrap draws, stratified by mix; "
        "mixes weighted equally; seed 343014. "
        "Intervals condition on the single fitted head. "
        "All four fixed margins reported; no margin selected. "
        "Per-mix results are in result.json. Deaths are terminal hero deaths; "
        "override denominator is living decisions. "
        "Watch6-v2 was not included: the reused simulator implements v1 terminal-hero "
        "dynamics, not v2 continuous respawn.\n"
    )
    atomic(destination / "RESULT.md", text.encode())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, action="append", required=True)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    a = p.parse_args()
    readout(a.output, a.input_dir, a.destination)


if __name__ == "__main__":
    main()
