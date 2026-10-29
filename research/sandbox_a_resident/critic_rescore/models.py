"""Lossless inference-only projection of the five fixed rescore networks.

Mac admission rebuilds the projection from the original hash-verified sources.
Runtime receives no optimizer, actor-training or simulator state. This format
cannot resume training and is exclusive to the frozen critic diagnostic.
"""

from __future__ import annotations

import io
import json
import lzma
import tempfile
from pathlib import Path

import torch

from research.sandbox_a_resident.training_compact import decode_stream, describe_state
from research.sandbox_a_resident.units import digest, publish

SCHEMA = "critic-rescore-inference-models/v1"
MAX_DECODED_BYTES = 32 * 1024**2
B8_SHA = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
FRP_SHA = "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"
CHAMPION_SHA = "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93"
CRITIC_SHAS = (
    "2b5478e895e7b2b68c02e3bd118612257ea4e97c0cfab727013d7e2143338ab6",
    "33863b37377f781c26846961c5b53cb5e44f65f8e3f14bec2d3ecfb83f363737",
)
ORIGINALS = dict(
    b8=B8_SHA, frp=FRP_SHA, champion=CHAMPION_SHA, seed1=CRITIC_SHAS[0], seed2=CRITIC_SHAS[1]
)


def original_projection() -> dict:
    """Read fixed original files on Mac; refuse any original-byte mismatch."""
    root = Path(__file__).resolve().parents[3]
    artifacts = root.parent / "snake-dqn-artifacts/sandbox-a-resident"
    saved = root.parent / "snake-dqn/saved_snakes"
    paths = {
        "b8": saved / "b8_s3_t7_20261016.pth",
        "frp": saved / "frp3_m3_s12_u60000_20261005.pth",
        "champion": saved / "champion_a5_freespace_20260621.pth",
    }
    models = {}
    for name, path in paths.items():
        if digest(path) != ORIGINALS[name]:
            raise ValueError("original network identity mismatch")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if name == "b8":
            models[name] = {key: blob[key] for key in ("config", "state_dict", "obs_spec", "arch")}
        else:
            models[name] = {
                key: blob[key]
                for key in ("dqn_state_dict", "input_size", "hidden_size", "output_size")
            }
    for seed, directory in enumerate(
        (
            artifacts / "train01-attempt-11/output",
            artifacts / "train01-s2-attempt-04/output-a100-fallback",
        ),
        1,
    ):
        compressed = directory / "exports/training/iteration-0000.pt.xz"
        meta = json.loads(Path(str(compressed) + ".manifest").read_text())
        if (
            meta["original"]["sha256"] != ORIGINALS[f"seed{seed}"]
            or digest(compressed) != meta["derivative"]["sha256"]
        ):
            raise ValueError("original critic identity mismatch")
        with tempfile.TemporaryDirectory(prefix="rescore-project-") as tmp:
            original = Path(tmp) / "iteration-0000.pt"
            with original.open("xb") as stream:
                decode_stream(
                    compressed,
                    stream.write,
                    expected_size=meta["original"]["size"],
                    expected_sha256=ORIGINALS[f"seed{seed}"],
                )
            state = torch.load(original, map_location="cpu", weights_only=False, mmap=True)
            if state["schema"] != "sandbox-a-train-state/v1" or state["iteration"] != 0:
                raise ValueError("expected iteration-zero prepared critic")
            # Clone only the critic tensors: no references into the huge original mmap.
            models[f"seed{seed}"] = {k: v.detach().clone() for k, v in state["critic"].items()}
            del state
    return dict(schema=SCHEMA, originals=ORIGINALS, models=models)


def load_bundle(path: Path) -> dict:
    """Decode a bounded single XZ stream, then load tensor-only inference state."""
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=64 * 1024**2)
    raw = decoder.decompress(path.read_bytes(), max_length=MAX_DECODED_BYTES + 1)
    if len(raw) > MAX_DECODED_BYTES or not decoder.eof or decoder.unused_data:
        raise ValueError("oversize, truncated or trailing model bundle")
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (
        set(value) != {"schema", "originals", "models"}
        or value["schema"] != SCHEMA
        or value["originals"] != ORIGINALS
        or set(value["models"]) != set(ORIGINALS)
    ):
        raise ValueError("foreign inference model bundle")
    return value


def verify_bundle(path: Path) -> dict:
    """Mac-only equality proof against every original source, not bundle claims."""
    for member in (path, Path(str(path) + ".sha256"), Path(str(path) + ".complete")):
        if not member.is_file() or member.is_symlink():
            raise ValueError("incomplete inference model bundle")
    if Path(str(path) + ".sha256").read_text().strip() != digest(path):
        raise ValueError("corrupt inference model bundle")
    actual = describe_state(load_bundle(path))
    expected = describe_state(original_projection())
    if actual != expected:
        raise ValueError("inference projection differs from original tensors/metadata")
    return actual


def build_bundle(path: Path) -> dict:
    """Publish one minimal model input, verify the round trip before admission."""
    projection = original_projection()
    stream = io.BytesIO()
    torch.save(projection, stream)
    if stream.tell() > MAX_DECODED_BYTES:
        raise ValueError("inference model bundle exceeds decoded bound")
    raw = lzma.compress(stream.getvalue(), format=lzma.FORMAT_XZ, check=lzma.CHECK_SHA256)
    publish(path, lambda p: p.write_bytes(raw))
    proof = verify_bundle(path)
    return dict(
        path=str(path),
        sha256=digest(path),
        size=path.stat().st_size,
        decoded_bytes=stream.tell(),
        originals=ORIGINALS,
        projection=proof,
    )
