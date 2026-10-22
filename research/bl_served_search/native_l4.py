"""Bounded L4 transport wrapper for the CPU-only paired native ruler.

CUDA is used only for the supervisor's family-canary action observation.  The
paired ruler itself remains in :mod:`native_ruler` with its CPU and one-thread
attestations unchanged.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from research.bl_served_search import native_ruler
from research.bl_served_search.exports import export_json, verify_export

CANARY_CAP_SECONDS = 600.0
MAIN_CAP_SECONDS = 1800.0


def _remaining(started: float, cap: float) -> float:
    remaining = cap - (time.monotonic() - started)
    if remaining <= 0:
        raise RuntimeError("INCOMPLETE native L4 wrapper wall cap")
    return remaining


def _invoke_native(arguments: list[str]) -> None:
    """Call the existing source-bound native ruler without changing its API."""
    original = sys.argv
    try:
        sys.argv = ["native_ruler", *arguments]
        native_ruler.main()
    finally:
        sys.argv = original


def cuda_action_audit(fixtures: Path, out: Path) -> None:
    """Record one real masked B8 CUDA decision from a frozen ruler root.

    The copied network and one forward are deliberately outside the CPU ruler's
    timing cells.  It provides the family gate with an attributable Tensor.argmax
    call, while the receipt's measured CPU kernel cells retain their own rules.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("L4 family gate requires an available CUDA device")
    if os.environ.get("SNAKE_COLAB_CANARY") == "1":
        from research.colab_cli.canary_observer import install

        install()
    import json

    from research.bl_served_search.ruler import validate_fixtures
    from research.bl_served_search.ruler_operator import ExactT3Operator
    from src.model.ego2s_network import obs_tensors
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    value = json.loads(fixtures.read_text())
    native_ruler.validate_source(value["source_identity"])
    fixture = validate_fixtures(value)[0]
    operator = ExactT3Operator(native_mode="kernel", search_mode="all")
    world = operator._restore(fixture)
    observations = build_ego_raster(
        world.sim,
        EgoRasterConfig(version="bv8"),
        backend="numba",
        rows=np.asarray([[0, 0]], dtype=np.int64),
    )
    legal = np.asarray(world.sim.get_advisory_action_mask()[0, 0], dtype=bool)
    if legal.shape != (6,) or not legal.any():
        raise RuntimeError("frozen ruler root has no legal B8 action")
    device = torch.device("cuda")
    network = copy.deepcopy(world.hero.net).to(device).eval()
    with torch.inference_mode():
        q_cuda = network(*obs_tensors(observations, device)).float()
        if tuple(q_cuda.shape) != (1, 6) or not torch.isfinite(q_cuda).all():
            raise RuntimeError("malformed B8 CUDA Q values")
        mask = torch.as_tensor(legal, device=device, dtype=torch.bool).unsqueeze(0)
        cuda_action = int(q_cuda.masked_fill(~mask, float("-inf")).argmax(dim=1).item())
    q_cpu = q_cuda.detach().cpu().numpy()[0].astype(np.float64, copy=False)
    cpu_action = int(np.where(legal, q_cpu, -np.inf).argmax())
    if cuda_action != cpu_action:
        raise RuntimeError("CUDA B8 masked action differs from same-Q CPU convention")
    export_json(
        out / "cuda-action-audit.json",
        {
            "status": "PASS",
            "purpose": "family-canary CUDA action evidence only; excluded from CPU ruler timing",
            "fixture_id": fixture["fixture_id"],
            "device": str(device),
            "q_shape": list(q_cuda.shape),
            "legal_actions": np.flatnonzero(legal).tolist(),
            "cuda_masked_action": cuda_action,
            "same_q_cpu_masked_action": cpu_action,
        },
    )


def _promote(source: Path, destination: Path) -> None:
    """Move a successful fresh child payload into the supervisor export root."""
    if not destination.exists():
        # A nested completed-export tree is one immutable unit to the runtime
        # scanner.  Rename it instead of exposing its sidecars one by one.
        source.rename(destination)
        return
    if not destination.is_dir():
        raise NotADirectoryError(destination)

    def order(item: Path) -> tuple[int, str]:
        if item.name.endswith(".complete"):
            return (2, item.name)
        if item.name.endswith(".sha256"):
            return (1, item.name)
        return (0, item.name)

    for item in sorted(source.iterdir(), key=order):
        target = destination / item.name
        if target.exists():
            raise FileExistsError(f"refusing to overwrite runtime export: {target.name}")
        item.rename(target)
    source.rmdir()


def _merge_audit(source: Path, payload: Path) -> None:
    """Place a completed CUDA audit beside, but never inside, ruler execution."""
    for item in source.iterdir():
        target = payload / item.name
        if target.exists():
            raise FileExistsError(f"refusing to overwrite ruler payload: {target.name}")
        item.rename(target)
    source.rmdir()


def run(*, fixtures: Path, k1: Path, out: Path, max_seconds: float, canary: bool) -> None:
    """Run the internal canary then full ruler, or the standalone family canary."""
    if not 0 < max_seconds <= (CANARY_CAP_SECONDS if canary else MAIN_CAP_SECONDS):
        raise ValueError("invalid native L4 wrapper cap")
    verify_export(fixtures)
    verify_export(k1)
    if not out.is_dir() or any(out.iterdir()):
        raise FileExistsError("L4 output root must be an empty supervisor-created directory")
    started = time.monotonic()
    # The supervisor's export scanner caches every completed file it sees below
    # COLAB_OUTPUT_DIR.  Stage beside that root so promotion never removes a
    # previously scanned immutable export.
    stage = Path(tempfile.mkdtemp(prefix=".native-l4-stage-", dir=out.parent))
    common = ["--fixtures", str(fixtures), "--k1", str(k1), "--platform", "l4"]
    try:
        if canary:
            payload = stage / "canary"
            audit = stage / "canary-audit"
            cuda_action_audit(fixtures, audit)
            _invoke_native(
                [
                    *common,
                    "--out",
                    str(payload),
                    "--canary",
                    "--max-seconds",
                    str(_remaining(started, max_seconds)),
                ]
            )
            _merge_audit(audit, payload)
            _promote(payload, out)
        else:
            canary_payload = stage / "native-canary"
            canary_audit = stage / "native-canary-audit"
            cuda_action_audit(fixtures, canary_audit)
            _invoke_native(
                [
                    *common,
                    "--out",
                    str(canary_payload),
                    "--canary",
                    "--max-seconds",
                    str(min(CANARY_CAP_SECONDS, _remaining(started, max_seconds))),
                ]
            )
            full_payload = stage / "full"
            full_audit = stage / "full-audit"
            cuda_action_audit(fixtures, full_audit)
            _invoke_native(
                [
                    *common,
                    "--out",
                    str(full_payload),
                    "--canary-receipt",
                    str(canary_payload / "ruler-receipt.json"),
                    "--max-seconds",
                    str(_remaining(started, max_seconds)),
                ]
            )
            _merge_audit(canary_audit, canary_payload)
            _merge_audit(full_audit, full_payload)
            _promote(canary_payload, out / "native-canary")
            _promote(full_payload, out)
        stage.rmdir()
    except BaseException:
        # The supervisor collects this bounded failure evidence; never publish a
        # partial receipt at the root where it could be mistaken for a result.
        export_json(out / "stopped.json", {"status": "INCOMPLETE", "stage": "native-l4"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--k1", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    run(
        fixtures=args.fixtures,
        k1=args.k1,
        out=args.out,
        max_seconds=args.max_seconds,
        canary=args.canary,
    )


if __name__ == "__main__":
    main()
