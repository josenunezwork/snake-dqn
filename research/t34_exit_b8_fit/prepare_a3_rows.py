"""Extract a bounded actual-state A3 bank from an existing T3 observation shard.

This is an offline format conversion, not replay regeneration or a new world bank.
The input bytes and chosen prefix are recorded for the identity receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    """Hash without materializing an artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(source: Path, output: Path, states: int = 10_000, batch: int = 64) -> dict:
    """Convert raw or losslessly packed T3 rows into memory-mappable arrays."""
    if states < 1 or batch < 1:
        raise ValueError("states and batch must be positive")
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="t34-a3-extract-") as temporary:
        temp = Path(temporary)
        with zipfile.ZipFile(source) as archive:
            names = set(archive.namelist())
            keys = ("packed", "scalars") if "packed.npy" in names else (
                "local", "global", "scalars"
            )
            for key in keys:
                # Exact named members only; never extract archive paths.
                with archive.open(key + ".npy") as reader, (temp / (key + ".npy")).open(
                    "xb"
                ) as writer:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
        arrays = {key: np.load(temp / (key + ".npy"), mmap_mode="r") for key in keys}
        if any(len(array) < states for array in arrays.values()):
            raise ValueError(f"input shard has fewer than {states} actual states")
        destinations = {
            "local": np.lib.format.open_memmap(
                output / "local.npy", mode="w+", dtype=np.uint8, shape=(states, 7, 31, 31)
            ),
            "global": np.lib.format.open_memmap(
                output / "global.npy", mode="w+", dtype=np.uint8, shape=(states, 4, 37, 37)
            ),
            "scalars": np.lib.format.open_memmap(
                output / "scalars.npy", mode="w+", dtype=np.float32, shape=(states, 60)
            ),
        }
        for start in range(0, states, batch):
            stop = min(start + batch, states)
            if "packed" in arrays:
                from research.redesign_m3c_20261008.obs_pack import train_layout, unpack_ref

                local, glob = unpack_ref(arrays["packed"][start:stop], train_layout())
            else:
                local, glob = arrays["local"][start:stop], arrays["global"][start:stop]
            destinations["local"][start:stop] = local
            destinations["global"][start:stop] = glob
            destinations["scalars"][start:stop] = arrays["scalars"][start:stop]
        for array in destinations.values():
            array.flush()
    result = {
        "schema": "t34-a3-observation-prefix/v1",
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "states": states,
        "batch": batch,
        "claim": "Existing T3 actual observation prefix; not teacher replay acceptance",
        "arrays": {key: sha256(output / (key + ".npy")) for key in destinations},
    }
    (output / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--states", type=int, default=10_000)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.out, args.states, args.batch), sort_keys=True))


if __name__ == "__main__":
    main()
