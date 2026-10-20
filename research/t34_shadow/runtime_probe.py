#!/usr/bin/env python3
"""Bounded, batch-one CPU runtime probe for the released B8 checkpoint.

This measurement intentionally compares inference runtimes only.  It does not
exercise a game world, a live service, or a serving policy.  Each backend is
timed one forward at a time after warm-up, and every emitted receipt is
create-only so a repeated probe cannot overwrite earlier evidence.

The ONNX Runtime path is conditional on the existing venv already containing
both ``onnx`` and ``onnxruntime``.  No package installation is attempted.
Apple's BNNS APIs do not provide a supported Python execution binding here;
that fact is retained in the receipt rather than treating Core ML conversion
as a BNNS result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Set before importing NumPy/Torch.  The caller should also export these caps,
# because libraries can initialize before Python imports in some launchers.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.t34_shadow import common  # noqa: E402

DEFAULT_CHECKPOINT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth"
)
DEFAULT_ARTIFACT_DIR = common.ARTIFACTS


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> Mapping[str, str]:
    """Hash the probe and model source used by this receipt."""
    paths = (Path(__file__), REPO / "src/model/ego2s_network.py")
    return {str(path.relative_to(REPO)): sha256_file(path) for path in paths}


def git_revision() -> str | None:
    """Return the worktree revision without making the receipt depend on Git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def percentile_ms(samples_ns: Sequence[int]) -> Mapping[str, float]:
    """Compute declared latency quantiles in milliseconds."""
    values = np.asarray(samples_ns, dtype=np.float64) / 1_000_000.0
    return {
        "count": int(values.size),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def timed(
    forward: Callable[[], np.ndarray], warmup: int, samples: int, budget: common.Budget
) -> tuple[np.ndarray, Mapping[str, float]]:
    """Warm one batch-one callable and return its final output and timings."""
    output: np.ndarray | None = None
    for _ in range(warmup):
        budget.check()
        output = forward()
    elapsed: list[int] = []
    for _ in range(samples):
        budget.check()
        start = time.perf_counter_ns()
        output = forward()
        elapsed.append(time.perf_counter_ns() - start)
    assert output is not None
    return output, percentile_ms(elapsed)


def batch_one_inputs(net: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create fixed, valid-shape B8-like batch-one tensors without game simulation."""
    config = getattr(net, "config", {})
    local_channels = int(config["local_channels"])
    n_scalars = int(config["n_scalars"])
    rng = np.random.default_rng(34)
    return (
        torch.from_numpy(rng.integers(0, 256, (1, local_channels, 31, 31), dtype=np.uint8)),
        torch.from_numpy(rng.integers(0, 256, (1, 4, 37, 37), dtype=np.uint8)),
        torch.from_numpy(rng.standard_normal((1, n_scalars)).astype(np.float32)),
    )


def torch_backend(
    net: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    warmup: int,
    samples: int,
    budget: common.Budget,
) -> tuple[np.ndarray, Mapping[str, float]]:
    """Run eager PyTorch CPU under inference mode."""

    def forward() -> np.ndarray:
        with torch.inference_mode():
            return net(*inputs).detach().cpu().numpy().copy()

    return timed(forward, warmup, samples, budget)


def onnx_backend(
    net: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    warmup: int,
    samples: int,
    export_path: Path,
    budget: common.Budget,
) -> tuple[np.ndarray, Mapping[str, float]]:
    """Export once then measure an existing ONNX Runtime CPU provider."""
    import onnxruntime as ort

    torch.onnx.export(
        net,
        inputs,
        export_path,
        input_names=["local", "global", "scalars"],
        output_names=["q"],
        dynamo=False,
        opset_version=18,
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(
        str(export_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"ONNX Runtime did not select CPU only: {session.get_providers()}")
    feeds = {"local": inputs[0].numpy(), "global": inputs[1].numpy(), "scalars": inputs[2].numpy()}

    def forward() -> np.ndarray:
        return np.asarray(session.run(["q"], feeds)[0]).copy()

    output, timing = timed(forward, warmup, samples, budget)
    return output, {
        **timing,
        "session_options": {
            "intra_op_num_threads": options.intra_op_num_threads,
            "inter_op_num_threads": options.inter_op_num_threads,
            "execution_mode": "ORT_SEQUENTIAL",
            "intra_op_allow_spinning": False,
            "inter_op_allow_spinning": False,
            "providers": session.get_providers(),
        },
    }


def comparison(reference: np.ndarray, candidate: np.ndarray) -> Mapping[str, Any]:
    """Describe numerical/action differences; this is deliberately not an identity claim."""
    diff = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return {
        "max_abs_q_diff": float(diff.max()),
        "mean_abs_q_diff": float(diff.mean()),
        "reference_action": int(np.argmax(reference, axis=1)[0]),
        "candidate_action": int(np.argmax(candidate, axis=1)[0]),
        "action_difference_count": int(
            np.count_nonzero(np.argmax(reference, 1) != np.argmax(candidate, 1))
        ),
    }


def available(module: str) -> bool:
    """Check installed capability without importing or installing it."""
    return importlib.util.find_spec(module) is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=5_000)
    parser.add_argument("--budget-seconds", type=float, default=600.0)
    parser.add_argument(
        "--receipt", type=Path, required=True, help="New raw JSON filename; must not exist"
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("--warmup must be non-negative and --samples must be positive")
    if not 0 < args.budget_seconds <= 600:
        parser.error("--budget-seconds must be in (0, 600]")
    if args.receipt.parent == Path("."):
        args.receipt = args.artifact_dir / args.receipt
    if args.receipt.exists():
        parser.error(f"refusing to overwrite existing receipt: {args.receipt}")
    if not args.checkpoint.is_file():
        parser.error(f"missing checkpoint: {args.checkpoint}")
    return args


def main() -> int:
    args = parse_args()
    budget = common.Budget(args.budget_seconds)
    shared_provenance = common.provenance()
    budget.check()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from src.model.ego2s_network import load_ego2s_checkpoint

    net = load_ego2s_checkpoint(str(args.checkpoint), "cpu").eval()
    budget.check()
    inputs = batch_one_inputs(net)
    reference, torch_timing = torch_backend(net, inputs, args.warmup, args.samples, budget)
    backends: dict[str, Any] = {
        "pytorch_cpu": {"status": "measured", "timing": torch_timing},
        "onnxruntime_cpu": {"status": "unavailable", "reason": None},
        "accelerate_bnns": {
            "status": "unsupported",
            "reason": (
                "No supported direct Python BNNS/Accelerate execution binding is present; "
                "Core ML is not a BNNS measurement."
            ),
        },
    }
    if available("onnx") and available("onnxruntime"):
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        export_path = args.artifact_dir / (args.receipt.stem + ".onnx")
        if export_path.exists():
            raise FileExistsError(f"refusing to overwrite existing ONNX export: {export_path}")
        output, timing = onnx_backend(net, inputs, args.warmup, args.samples, export_path, budget)
        backends["onnxruntime_cpu"] = {
            "status": "measured",
            "timing": timing,
            "export_path": str(export_path),
            "comparison_to_pytorch": comparison(reference, output),
        }
    else:
        missing = [name for name in ("onnx", "onnxruntime") if not available(name)]
        backends["onnxruntime_cpu"][
            "reason"
        ] = f"Existing venv lacks: {', '.join(missing)}; no install attempted."
    receipt = {
        "schema": "t34-shadow-m4-runtime-probe/v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "argv": sys.argv,
        "git_revision": git_revision(),
        "common_provenance": shared_provenance,
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "thread_limits": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "torch": {
            "version": torch.__version__,
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "device": "cpu",
        },
        "measurement": {
            "batch_size": 1,
            "warmup_calls_excluded": args.warmup,
            "timed_calls": args.samples,
            "wall_budget_seconds": args.budget_seconds,
            "elapsed_seconds": time.monotonic() - budget.started,
            "host_contention": (
                "not measured by this probe; inspect concurrent Mac work before interpreting."
            ),
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            "config": dict(net.config),
        },
        "source_sha256": source_hashes(),
        "backends": backends,
        "pytorch_output": {
            "q": reference.tolist(),
            "greedy_action": int(np.argmax(reference, axis=1)[0]),
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps({"receipt": str(args.receipt), "backends": backends}, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
