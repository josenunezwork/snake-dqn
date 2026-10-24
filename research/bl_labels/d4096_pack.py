"""Create the ruled OLD D4096 rebinding without regenerating scientific inputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from .exports import _verify, _write_once, publish_json

ART = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
ORIGINAL = ART / "bl-labels-f0-speed/d1024-d4096-pack-01"
PACK = ART / "bl-labels-f0-speed/d4096-old-equiv-02-pack-01"
OLD_SCIENCE = "bf8ac97dc66d1b4515e4c5510db3ad5886fa695d"
SCIENCE = "111c47818a097c57451386ef4f8a7009616f8f77"
JUDGMENT_SHA = "ccf1e0cd8d60a2019c983ab2714d58663ca42e2a228a0ddf78c16216c2ef83fb"
ALLOCATION_SHA = "74b1c5db8cd0c082311cf2c02666418847855298762174e0b9f721762cbff229"


def _info(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
    }


def create(original: Path, pack: Path, judgment: Path, binding: Path) -> dict:
    """Rebind exactly two control strings; preserve every other scientific byte."""
    if hashlib.sha256(judgment.read_bytes()).hexdigest() != JUDGMENT_SHA:
        raise ValueError("D4096 start judgment sha differs")
    if pack.exists() or binding.exists():
        raise FileExistsError("D4096 rebinding pack and binding are create-only")
    control_path = original / "d4096-control.json"
    _verify(control_path)
    raw = control_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != (
        "c3fe9861cd177de57644175d487746f028024d1596fbeda1fdc57250a0a28207"
    ):
        raise ValueError("original D4096 control bytes differ from pack-01")
    control = json.loads(raw)
    expected = copy.deepcopy(control)
    if (
        control.get("source_commit") != OLD_SCIENCE
        or control.get("mac_gate", {}).get("source_commit") != OLD_SCIENCE
        or control["mac_gate"].get("stage") != "d4096"
        or control["mac_gate"].get("status") != "HOLD"
        or control["mac_gate"].get("deadline_seconds") != 468000
        or len(control.get("allocation", [])) != 7020
    ):
        raise ValueError("original D4096 control differs from ruled pack")
    digest = hashlib.sha256(
        json.dumps(control["allocation"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != ALLOCATION_SHA:
        raise ValueError("original D4096 allocation differs")
    needle = f'"source_commit":"{OLD_SCIENCE}"'.encode()
    if raw.count(needle) != 2:
        raise ValueError("original control does not contain exactly two source bindings")
    rebound = raw.replace(needle, f'"source_commit":"{SCIENCE}"'.encode())
    expected["source_commit"] = expected["mac_gate"]["source_commit"] = SCIENCE
    if json.loads(rebound) != expected:
        raise ValueError("rebinding changed an unruled control field")
    names = [
        "d4096-control.json",
        "d4096-mac-hold.json",
        "d4096-seed-audit.json",
        "d4096-verified-ledger.json",
        "d1024-control.json",
    ]
    originals = {}
    payloads = {}
    for name in names:
        path = original / name
        _verify(path)
        originals[name] = _info(path)
        payloads[name] = path.read_bytes()
    payloads["d4096-control.json"] = rebound
    hold = payloads["d4096-mac-hold.json"]
    if hold.count(needle) != 1:
        raise ValueError("original HOLD source binding differs")
    payloads["d4096-mac-hold.json"] = hold.replace(needle, f'"source_commit":"{SCIENCE}"'.encode())
    ledger = json.loads(payloads["d4096-verified-ledger.json"])
    if ledger.get("units") != [] or ledger.get("allocation") != control["allocation"]:
        raise ValueError("original D4096 ledger is not the frozen empty allocation")
    pack.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        path = pack / name
        _write_once(path, payload)
        sha = hashlib.sha256(payload).hexdigest()
        _write_once(Path(str(path) + ".sha256"), f"{sha}  {name}\n".encode())
        _write_once(Path(str(path) + ".complete"), b"complete\n")
    value = {
        "schema": "bl-labels-f0-d4096-source-rebinding/v1",
        "status": "PREPARED_NOT_DISPATCHED",
        "judgment_sha256": JUDGMENT_SHA,
        "package": str(pack.resolve()),
        "original_package": str(original.resolve()),
        "original_files": originals,
        "files": {name: _info(pack / name) for name in names},
        "control_changes": {
            key: {"from": OLD_SCIENCE, "to": SCIENCE}
            for key in ("source_commit", "mac_gate.source_commit")
        },
        "allocation_sha256": digest,
        "allocation_parents": len(control["allocation"]),
        "allocation_index_span": 8792,
        "judgment_count_erratum": "8792 is the inclusive index span; frozen fold-1--4 rows = 7020",
        "deadline_seconds": 468000,
        "namespace": control["namespace"],
        "seed_audit_byte_identical": True,
        "empty_verified_ledger_byte_identical": True,
        "prior_d1024_control_byte_identical": True,
        "collection_started": False,
        "outcomes_opened": False,
    }
    publish_json(binding, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ORIGINAL)
    parser.add_argument("--pack", type=Path, default=PACK)
    parser.add_argument("--judgment", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    args = parser.parse_args()
    value = create(args.original, args.pack, args.judgment, args.binding)
    print(json.dumps({k: value[k] for k in ("status", "package", "allocation_sha256")}))


if __name__ == "__main__":
    main()
