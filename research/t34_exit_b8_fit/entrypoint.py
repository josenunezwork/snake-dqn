"""Bounded ExIt-B8 fit preflight with durable, complete hash-bound output."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


class EntrypointRefused(RuntimeError):
    """The untested or incomplete fit seam must not be treated as production-ready."""


def _write_complete_json(out_dir: Path, report: dict[str, Any]) -> Path:
    if not out_dir.is_dir():
        raise EntrypointRefused("preflight output directory was not initialized")
    payload = out_dir / "fit-preflight.json"
    encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    payload.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    (out_dir / "fit-preflight.json.sha256").write_text(digest + "  fit-preflight.json\n")
    (out_dir / "fit-preflight.json.complete").write_text("complete\n")
    return payload


def _require_export_complete(exports: Path) -> None:
    payload = exports / "payload.json"
    marker = exports / "payload.json.complete"
    sidecar = exports / "payload.json.sha256"
    if not (payload.is_file() and marker.is_file() and sidecar.is_file()):
        raise EntrypointRefused("engine export requires payload.json, .sha256, and .complete")
    expected = sidecar.read_text().split()[0] if sidecar.read_text().split() else ""
    actual = hashlib.sha256(payload.read_bytes()).hexdigest()
    if expected != actual:
        raise EntrypointRefused("engine export payload hash mismatch")


def _binding(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise EntrypointRefused("real fit requires the hash-bound engine binding")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EntrypointRefused("invalid engine binding") from exc
    from .make_manifest import ManifestRefused, validate_binding

    try:
        return validate_binding(value)
    except ManifestRefused as exc:
        raise EntrypointRefused(str(exc)) from exc


def run(
    *,
    exports: Path,
    checkpoint: Path | None,
    out_dir: Path,
    worlds: int,
    updates: int,
    device: str,
    fake_engine: bool = False,
    mode: str = "preflight",
    epochs: int = 5,
    binding_path: Path | None = None,
) -> Path:
    if mode not in {"preflight", "fit"}:
        raise EntrypointRefused("mode must be preflight or fit")
    if mode == "preflight" and (worlds != 2 or updates != 200):
        raise EntrypointRefused("P6 preflight is fixed at exactly 2 worlds and 200 updates")
    if mode == "fit" and not 4 <= epochs <= 6:
        raise EntrypointRefused("production fit requires 4 to 6 epochs")
    if device not in {"cpu", "cuda"}:
        raise EntrypointRefused(
            "device must be cpu for Mac preflight or cuda for the L4 entrypoint"
        )
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if not fake_engine:
        _require_export_complete(exports)
        if checkpoint is None or not checkpoint.is_file():
            raise EntrypointRefused("real preflight requires a read-only B8 checkpoint")
    from research.t34_exit_b8_fit import runner

    binding = None if fake_engine else _binding(binding_path)
    if binding is not None:
        expected = binding["inputs"][0]
        payload = exports / "payload.json"
        if payload.stat().st_size != expected["size_bytes"] or (
            hashlib.sha256(payload.read_bytes()).hexdigest() != expected["sha256"]
        ):
            raise EntrypointRefused("engine export differs from the immutable binding")
    result = runner.run(
        exports=exports,
        checkpoint=checkpoint,
        output_dir=out_dir / "fit-output",
        worlds=worlds,
        updates=updates,
        device=device,
        mode=mode,
        epochs=epochs,
        fake_engine=fake_engine,
        binding=binding,
    )
    if not isinstance(result, dict):
        raise EntrypointRefused("fit adapter must return a JSON object")
    completed = bool(result.get("completed", False))
    tripwire = result.get("tripwire")
    probes_ok = bool(result.get("probes_ok", False))
    stopped = not completed or bool(tripwire) or (not fake_engine and not probes_ok)
    result["status"] = (
        "FIT_STOPPED"
        if stopped
        else ("LOCAL_FAKE_ENGINE_SMOKE_ONLY" if fake_engine else "FIT_COMPLETED")
    )
    result.update(
        worlds=worlds,
        updates=int(result.get("updates", updates)),
        requested_update_limit=updates if mode == "preflight" else None,
        device=device,
        mode=mode,
        epochs=epochs,
        elapsed_seconds=time.monotonic() - started,
        checkpoint=str(checkpoint) if checkpoint else None,
    )
    report = _write_complete_json(out_dir, result)
    if stopped:
        raise EntrypointRefused(
            f"fit stopped: {tripwire or 'probe or completion failure'}; report={report}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fake-engine", action="store_true")
    parser.add_argument("--mode", choices=("preflight", "fit"), default="preflight")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--binding", type=Path)
    args = parser.parse_args()
    result = run(
        exports=args.exports,
        checkpoint=args.checkpoint,
        out_dir=args.out,
        worlds=args.worlds,
        updates=args.updates,
        device=args.device,
        fake_engine=args.fake_engine,
        mode=args.mode,
        epochs=args.epochs,
        binding_path=args.binding,
    )
    print(json.dumps({"report": str(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
