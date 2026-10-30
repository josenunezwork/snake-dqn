"""Bounded checkpoint inputs and offline parity with broker upload size guards."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from research.colab_cli import source
from research.watch6_search_efficacy import protocol as p
from tools.ops.package_spec import Input, digest, verify_immutable

PART_BYTES = 2 * 1024**2
CHECKPOINTS = {
    "b8_s3_t7_20261016.pth": ("b8-s3", 6_562_005),
    "frp3_m3_s12_u60000_20261005.pth": ("frp3-s12", 6_906_247),
}


def checkpoint_inputs(saved: Path, directory: Path) -> list[Input]:
    """Split exact released bytes without reserializing or changing checkpoint pins."""
    pins = json.loads((p.ROOT / "web/backend/served_checkpoint_pins.json").read_text())
    directory.mkdir(parents=True, exist_ok=False)
    inputs = []
    for name, (pin, size) in CHECKPOINTS.items():
        path = saved / name
        if (
            path.stat().st_size != size
            or digest(path) != pins["checkpoints"][pin]["checkpoint_sha256"]
        ):
            raise ValueError(f"released checkpoint identity differs: {name}")
        with path.open("rb") as stream:
            for index in range((size + PART_BYTES - 1) // PART_BYTES):
                part = directory / f"{name}.{index:03d}.part"
                with part.open("xb") as output:
                    output.write(stream.read(PART_BYTES))
                inputs.append(Input(part, "checkpoint-parts/" + part.name))
    return inputs


def stage_checkpoints(input_root: Path) -> None:
    """Verify each immutable part, then publish only the full pinned checkpoint."""
    pins = json.loads((p.ROOT / "web/backend/served_checkpoint_pins.json").read_text())
    output = input_root / "checkpoints"
    output.mkdir(exist_ok=False)
    for name, (pin, size) in CHECKPOINTS.items():
        with tempfile.NamedTemporaryFile(dir=output, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                for index in range((size + PART_BYTES - 1) // PART_BYTES):
                    part = input_root / "checkpoint-parts" / f"{name}.{index:03d}.part"
                    verified = verify_immutable(part)
                    if verified["size"] != min(PART_BYTES, size - index * PART_BYTES):
                        raise ValueError("checkpoint part size differs")
                    stream.write(part.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
                if (
                    temporary.stat().st_size != size
                    or digest(temporary) != pins["checkpoints"][pin]["checkpoint_sha256"]
                ):
                    raise ValueError("reassembled checkpoint digest differs from released pin")
                temporary.chmod(0o444)
                os.link(temporary, output / name)  # atomic, create-only publication
            finally:
                temporary.unlink(missing_ok=True)


def check_uploads(manifest: dict) -> dict:
    """Check every possible upload using the broker's unchanged per-file bound.

    The broker uses max_export_bytes for private uploads and staging files too.
    Build its actual sparse source plan/archive, including fallback, offline;
    include all four commit-pinned runtime controls and every input sidecar.
    """
    limit = manifest["max_export_bytes"]
    rows = []

    def check(name: str, size: int) -> None:
        if size > limit:
            raise ValueError(f"upload exceeds bounded transfer size: {name}: {size} > {limit}")
        rows.append({"dest": name, "size_bytes": size})

    for item in manifest["inputs"]:
        if "path" in item:
            check("inputs/" + item["dest"], Path(item["path"]).stat().st_size)
    with tempfile.TemporaryDirectory(prefix="watch6-upload-preflight-") as temporary:
        staging = Path(temporary)
        source.prepare(p.ROOT, manifest, staging)
        for name in ("runtime.py", "bootstrap.sh", "release.py", "source.py"):
            data = subprocess.check_output(
                ["git", "show", f"{manifest['commit']}:research/colab_cli/{name}"],
                cwd=p.ROOT,
                timeout=10,
            )
            (staging / name).write_bytes(data)
        for path in sorted(staging.iterdir()):
            check(path.name, path.stat().st_size)
    return {"status": "PASS", "max_upload_bytes": limit, "files": rows}
