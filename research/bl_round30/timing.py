"""R30 timing-only calibration. Scientific values exist only in worker memory.

This is deliberately separate from job.py: no scientific exports, resume ledger,
solver run, endpoint selection or objective-quality decision is available here.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import math
import os
import re
import signal
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

RULING_SHA = "9b07b66663d8a7ac168f80dbfc35889b3551df8797f347335c10d83a8d81801f"
SCHEMA = "r30-timing-only/v1"
SCHEDULE = {
    "adam_blocks": [[0, 10], [10, 20], [20, 30]],
    "closure_updates": [0, 10, 30],
    "fit_batches": 2000,
    "fit_seed": 3403,
    "audit_seed": 3404,
    "certificate_update": 30,
    "certificate_cap_seconds": 600,
}
PHASES = (
    "imports",
    "unpack",
    "input_checks_load",
    "feature_cache",
    "plan_coefficients",
    "equality",
    "adam_setup",
    "adam_0_10",
    "adam_10_20",
    "adam_20_30",
    "closure_0",
    "closure_10",
    "closure_30",
    "solver_auxiliary",
    "bookkeeping_probe",
    "certificate",
    "diagnostics",
    "bindings",
    "fusion_serialization",
    "main_export_io",
    "cleanup",
    "metadata_export_probe",
)
ROOT = Path(__file__).resolve().parents[2]


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_report(value: dict) -> None:
    """Exact typed allowlist; no recursive scientific dictionary can pass."""
    if set(value) != {
        "schema",
        "complete",
        "synthetic",
        "ruling_sha256",
        "config_sha256",
        "source_commit",
        "source_hashes",
        "input_hashes",
        "control_sha256",
        "bundle_sha256",
        "schedule",
        "device",
        "threads",
        "row_count",
        "feature_shape",
        "parameter_count",
        "phase_seconds",
        "worker_seconds",
    }:
        raise ValueError("timing export keys differ")
    if (
        value["schema"] != SCHEMA
        or value["complete"] is not True
        or type(value["synthetic"]) is not bool
        or value["ruling_sha256"] != RULING_SHA
        or value["schedule"] != SCHEDULE
        or value["device"] not in {"cuda", "cpu"}
        or type(value["threads"]) is not int
        or value["threads"] != 1
        or value["parameter_count"] != 1542
    ):
        raise ValueError("timing export identity differs")
    if not value["synthetic"] and value["device"] != "cuda":
        raise ValueError("production calibration requires CUDA features")
    if not re.fullmatch(r"[0-9a-f]{40}", str(value["source_commit"])):
        raise ValueError("source commit differs")
    for name in ("config_sha256", "control_sha256", "bundle_sha256"):
        if not _sha(value[name]):
            raise ValueError("identity digest differs")
    from research.bl_round30.timing_pack import identity_paths

    if set(value["source_hashes"]) != set(identity_paths()):
        raise ValueError("source allowlist differs")
    from research.bl_round30.private_bundle import _BASE_NAMES

    if not {"train.npz", "b8.pth", "sampler-plan.npz", "sampler-plan.json"} <= set(
        value["input_hashes"]
    ) or not set(value["input_hashes"]) <= _BASE_NAMES | {"b8.pth"}:
        raise ValueError("input allowlist differs")
    for name in ("source_hashes", "input_hashes"):
        if not isinstance(value[name], dict) or not all(_sha(v) for v in value[name].values()):
            raise ValueError("identity hashes differ")
    n = value["row_count"]
    if type(n) is not int or n <= 0 or value["feature_shape"] != [n, 256]:
        raise ValueError("shape metadata differs")
    phases = value["phase_seconds"]
    if not isinstance(phases, dict) or set(phases) != set(PHASES):
        raise ValueError("incomplete timing phases")
    for duration in (*phases.values(), value["worker_seconds"]):
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
            raise ValueError("invalid timing duration")
    if phases["certificate"] >= 600 or value["worker_seconds"] >= 2400:
        raise ValueError("incomplete calibration: timeout")
    if sum(phases.values()) > value["worker_seconds"] + 0.1:
        raise ValueError("overlapping timing phases")


@contextlib.contextmanager
def silence() -> Iterator[None]:
    """Suppress Python and native-library stdout/stderr, including exception text."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(fd) for fd in (1, 2)]
    with open(os.devnull, "w") as sink:
        try:
            for fd in (1, 2):
                os.dup2(sink.fileno(), fd)
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                yield
        finally:
            for fd, original in zip((1, 2), saved):
                os.dup2(original, fd)
                os.close(original)


@contextlib.contextmanager
def deadline(seconds: float) -> Iterator[None]:
    """Timeout never yields a partial completion-time report."""

    def expired(signum: int, frame: Any) -> None:
        raise TimeoutError("timing-only incomplete")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class Timer:
    """Nonoverlapping phase timings, never numerical return values."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    def call(self, name: str, operation: Callable[[], Any]) -> Any:
        if name not in PHASES or name in self.seconds:
            raise ValueError("duplicate/foreign phase")
        started = time.monotonic()
        result = operation()
        self.seconds[name] = time.monotonic() - started
        return result


def workload(
    cache: dict,
    base: Any,
    timer: Timer,
    frozen_plan: Path | None = None,
    table: dict | None = None,
) -> int:
    """Run the fixed full-plan workload; retain no scientific return payload."""
    import torch

    from research.bl_round30 import objective as obj
    from research.bl_round30.certificate import gradient_certificate
    from research.bl_round30.job import _checkpoint

    recipe = obj.ObjectiveRecipe()

    def plans() -> tuple:
        plan = obj.reproduce_batch_plan(cache, recipe)
        audit = obj.audit_batch_plan(cache, recipe)
        if frozen_plan is not None:
            import numpy as np

            with np.load(frozen_plan, allow_pickle=False) as archive:
                for key, actual in (("fit_indices", plan), ("audit_indices", audit)):
                    if not np.array_equal(archive[key], actual.indices.numpy()):
                        raise ValueError("full cache sampler differs from frozen plan")
        coefficients = obj.compile_coefficients(cache, plan, recipe)
        return plan, audit, coefficients

    plan, audit, coefficients = timer.call("plan_coefficients", plans)

    # Both full plans are reproduced and checked against the frozen input by
    # _load_training before this point; no indices or cache hash are exported.
    def equality() -> None:
        if not obj.verify_objective_equivalence(
            obj.zero_parameters(dtype=torch.float64), cache, plan, recipe
        ).passed:
            raise RuntimeError("incomplete equality workload")

    timer.call("equality", equality)

    def setup() -> tuple:
        theta = obj.zero_parameters().requires_grad_(True)
        return (
            theta,
            torch.optim.Adam((theta,), lr=recipe.learning_rate),
            obj._tensor_cache(cache, dtype=torch.float32),
        )

    theta, optimizer, data = timer.call("adam_setup", setup)
    ephemeral_losses: list[float] = []

    def closure(update: int) -> None:
        timer.call(
            f"closure_{update}",
            lambda: obj.evaluate_objective(
                theta.detach().double(), cache, plan, recipe, coefficients=coefficients
            ),
        )  # numerical result immediately discarded

    closure(0)
    for begin, end in SCHEDULE["adam_blocks"]:

        def block() -> None:
            for index in plan.indices[begin:end]:
                optimizer.zero_grad(set_to_none=True)
                loss = obj._component_losses(theta, data, index, recipe).total
                if not torch.isfinite(loss):
                    raise FloatingPointError("incomplete Adam workload")
                loss.backward()
                if theta.grad is None or not torch.isfinite(theta.grad).all():
                    raise FloatingPointError("incomplete Adam workload")
                optimizer.step()
                ephemeral_losses.append(float(loss.detach().cpu()))  # original Adam bookkeeping

        timer.call(f"adam_{begin}_{end}", block)
        if end in (10, 30):
            closure(end)

    def auxiliary() -> None:
        # Main's origin value, post-solve evaluation, Adam objective and setup.
        torch.optim.LBFGS(
            (theta.detach().double().requires_grad_(True),),
            max_iter=500,
            max_eval=500,
            tolerance_grad=0,
            tolerance_change=0,
            line_search_fn="strong_wolfe",
        )
        obj.finite_objective(
            obj.zero_parameters(dtype=torch.float64), cache, plan, recipe, coefficients=coefficients
        )
        obj.evaluate_objective(
            theta.detach().double(), cache, plan, recipe, coefficients=coefficients
        )
        obj.finite_objective(theta.detach().double(), cache, plan, recipe)
        # Snapshot/deepcopy/state conversion overhead, without publishing it.
        import copy

        copy.deepcopy(optimizer.state_dict())

    timer.call("solver_auxiliary", auxiliary)
    timer.call("bookkeeping_probe", bookkeeping_probe)
    timer.call(
        "certificate",
        lambda: gradient_certificate(theta.detach().double(), cache, plan, recipe, max_seconds=600),
    )  # no certified flag or bound survives
    timer.call(
        "diagnostics",
        lambda: obj.component_diagnostics(theta.detach().double(), cache, plan, recipe),
    )

    def bindings() -> None:
        obj._procedure_binding("adam", "zero-residual", obj.zero_parameters(), cache, plan, recipe)
        obj._procedure_binding("lbfgs", "r30-1-adam-endpoint", theta.detach(), cache, plan, recipe)

    timer.call("bindings", bindings)

    def fusion() -> None:
        # Main's full sampler lists/counters and two endpoint/history objects.
        # Scientific bytes remain in RAM. Only synthetic zero-filled I/O is
        # written to a private temporary directory, then removed before export.
        history = [torch.zeros(1542, dtype=torch.float64) for _ in range(200)]
        endpoints = {
            name: _checkpoint(base.cpu(), theta.detach(), {"history": history})
            for name in ("adam", "lbfgs")
        }
        fit_indices = plan.indices.tolist()
        audit_indices = audit.indices.reshape(-1).tolist()
        sampler = {
            "fit_indices": fit_indices,
            "audit_indices": audit_indices,
            "fit_multiplicities": dict(Counter(i for batch in fit_indices for i in batch)),
            "audit_multiplicities": dict(Counter(audit_indices)),
            "batch_rng_state": plan.rng_state.tolist(),
            "audit_rng_state": audit.rng_state.tolist(),
        }
        if table is not None:
            sampler["row_ids_by_index"] = table["row_id"].tolist()
            sampler["source_life_ids_by_index"] = table.get(
                "source_life_id", table["life_id"]
            ).tolist()
        with io.BytesIO() as buffer:
            torch.save(
                {"endpoints": endpoints, "sampler": sampler, "adam_state": optimizer.state_dict()},
                buffer,
            )

    timer.call("fusion_serialization", fusion)

    def export_io() -> None:
        from research.bl_ppo.exports import export_bytes

        # Charge main's entire 128 MiB file allowance with synthetic zeros.
        with tempfile.TemporaryDirectory(prefix="r30-timing-synthetic-io-") as scratch:
            export_bytes(Path(scratch) / "synthetic-io.bin", bytes(128 * 1024**2))

    timer.call("main_export_io", export_io)
    return int(cache["features"].shape[0])


def bookkeeping_probe() -> None:
    """Outcome-independent worst-history two-loop/line-search proxy.

    Forecast charges 10x this complete 500-step/100-history/25-trial proxy,
    with a 300-second floor. This is a planning allowance, not a proof of a
    Python/BLAS wall-time bound. Real strong-Wolfe closures are charged separately.
    """
    import torch

    vector = torch.ones(1542, dtype=torch.float64)
    history = [torch.ones_like(vector) / (i + 1) for i in range(100)]
    for _ in range(500):
        work = vector.clone()
        for entry in reversed(history):
            work.add_(entry, alpha=-float(torch.dot(entry, work)))
        for entry in history:
            work.add_(entry, alpha=float(torch.dot(entry, work)))
        for trial in range(25):
            vector + work * (1.0 / (trial + 1))


def run(input_dir: Path, output: Path, *, config_sha256: str, synthetic: bool = False) -> dict:
    """Only this function can publish; every phase must complete first."""
    from research.bl_ppo.exports import digest, export_json

    started = time.monotonic()
    launch_age = max(
        0.0, time.time() - float(os.environ.get("R30_TIMING_LAUNCH_EPOCH", time.time()))
    )
    timer = Timer()
    timer.seconds["imports"] = launch_age
    config = json.loads((input_dir / "timing-config.json").read_text())
    if digest(input_dir / "timing-config.json") != config_sha256:
        raise ValueError("config binding differs")
    if (
        config["schema"] != "r30-timing-config/v1"
        or config["ruling_sha256"] != RULING_SHA
        or config["schedule"] != SCHEDULE
        or config["synthetic"] is not synthetic
    ):
        raise ValueError("fixed timing schedule differs")
    from research.bl_round30.timing_pack import identity_paths

    if set(config["source_hashes"]) != set(identity_paths()):
        raise ValueError("source allowlist differs")
    if any(digest(ROOT / name) != sha for name, sha in config["source_hashes"].items()):
        raise ValueError("source binding differs")
    if output.is_symlink():
        raise ValueError("unsafe output")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("nonempty timing output")
    with silence(), deadline(max(0.001, 2390 - launch_age)):
        import torch

        from research.bl_residual.data import tensor_cache
        from research.bl_round30.job import _load_training, preflight_inputs
        from research.bl_round30.private_bundle import unpack
        from src.model.ego2s_network import Ego2sNet, load_ego2s_checkpoint

        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if synthetic:
            from research.bl_round30.fixture import synthetic_table

            timer.call("unpack", lambda: None)

            def load() -> tuple:
                torch.manual_seed(3403)
                base = Ego2sNet(local_channels=7, n_scalars=60).eval()
                return base, synthetic_table(base)

            base, table = timer.call("input_checks_load", load)
            device = "cpu"
        else:
            timer.call(
                "unpack",
                lambda: unpack(
                    input_dir / "private-inputs.tar.gz", input_dir, config["bundle_sha256"]
                ),
            )

            def load() -> tuple:
                # Old control/source bytes stay frozen; the new source is bound separately.
                if digest(input_dir / "control.json") != config["control_sha256"]:
                    raise ValueError("frozen control differs")
                if digest(input_dir / "timing-ruling.md") != RULING_SHA:
                    raise ValueError("ruling differs")
                control = preflight_inputs(input_dir, mode="r30-1")
                if control["input_hashes"] != config["input_hashes"]:
                    raise ValueError("frozen training inputs differ")
                return load_ego2s_checkpoint(str(input_dir / "b8.pth")), _load_training(
                    input_dir, control, updates=2000
                )

            base, table = timer.call("input_checks_load", load)
            device = "cuda"

        def features() -> dict:
            result = tensor_cache(table, base.to(device), batch_size=1)
            if device == "cuda":
                torch.cuda.synchronize()
            return result

        cache = timer.call("feature_cache", features)
        rows = workload(
            cache, base, timer, None if synthetic else input_dir / "sampler-plan.npz", table
        )

        def cleanup() -> None:
            cache.clear()
            table.clear()
            base.cpu()
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        timer.call("cleanup", cleanup)

        def export_probe() -> None:
            with tempfile.TemporaryDirectory(prefix="r30-timing-metadata-") as scratch:
                # Only fixed metadata filler, never a serialized scientific result.
                export_json(
                    Path(scratch) / "timing.json", {"timing_metadata_padding": "0" * (512 * 1024)}
                )

        timer.call("metadata_export_probe", export_probe)
    report = {
        "schema": SCHEMA,
        "complete": True,
        "synthetic": synthetic,
        "ruling_sha256": RULING_SHA,
        "config_sha256": config_sha256,
        "source_commit": config["source_commit"],
        "source_hashes": config["source_hashes"],
        "input_hashes": config["input_hashes"],
        "control_sha256": config["control_sha256"],
        "bundle_sha256": config["bundle_sha256"],
        "schedule": SCHEDULE,
        "device": device,
        "threads": 1,
        "row_count": rows,
        "feature_shape": [rows, 256],
        "parameter_count": 1542,
        "phase_seconds": timer.seconds,
        "worker_seconds": time.monotonic() - started + launch_age,
    }
    validate_report(report)
    export_json(output / "timing.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    # No traceback, native warning, exception payload or scientific log can escape.
    try:
        with silence():
            run(
                args.input_dir,
                args.output,
                config_sha256=args.config_sha256,
                synthetic=args.synthetic,
            )
    except BaseException:
        print('{"schema":"r30-timing-operational/v1","complete":false}', flush=True)
        raise SystemExit(1) from None
    print('{"schema":"r30-timing-operational/v1","complete":true}', flush=True)


if __name__ == "__main__":
    main()
