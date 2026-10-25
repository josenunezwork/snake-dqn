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

from research.bl_round30.timing_telemetry import OBSERVABILITY_SHA

RULING_SHA = "9b07b66663d8a7ac168f80dbfc35889b3551df8797f347335c10d83a8d81801f"
AMENDMENT_SHA = "447609b306bde15de4014640e1248a4349f8cb2feb811e455511a66886e943ff"
CERTIFICATE_AMENDMENT_SHA = "d21347475aba7b4e3be8e4e615b97d6a9f72d487d024a701aa81bf269d1bc837"
SCHEMA = "r30-timing-only/v2"
TIMING_PAYLOAD_BYTES = 524_288
DEADLINES = {
    "whole_session_seconds": 2400,
    "worker_seconds": 2100,
    "cleanup_export_reserve_seconds": 300,
}
SCHEDULE = {
    "adam_blocks": [[0, 10], [10, 20], [20, 30]],
    "closure_updates": [0, 10, 30],
    "fit_batches": 2000,
    "fit_seed": 3403,
    "audit_seed": 3404,
    "certificate_update": 30,
    "certificate_cap_seconds": 1200,
}
PHASES = (
    "imports",
    "unpack",
    "input_checks_load",
    "feature_cache",
    "b8_masked_action_decision",
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
        "amendment_sha256",
        "certificate_amendment_sha256",
        "observability_ruling_sha256",
        "deadlines",
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
        or value["amendment_sha256"] != AMENDMENT_SHA
        or value["certificate_amendment_sha256"] != CERTIFICATE_AMENDMENT_SHA
        or value["observability_ruling_sha256"] != OBSERVABILITY_SHA
        or value["deadlines"] != DEADLINES
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
    if (
        phases["certificate"] >= SCHEDULE["certificate_cap_seconds"]
        or value["worker_seconds"] >= DEADLINES["worker_seconds"]
    ):
        raise ValueError("incomplete calibration: timeout")
    if sum(phases.values()) > value["worker_seconds"] + 0.1:
        raise ValueError("overlapping timing phases")
    if (
        len((json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())
        > TIMING_PAYLOAD_BYTES
    ):
        raise ValueError("timing payload exceeds 512 KiB")


def validate_output_inventory(output: Path, *, complete: bool = True) -> None:
    """Admit exactly the timing payload and its immutable publication sidecars."""
    from research.bl_ppo.exports import verify_export
    from research.bl_round30.timing_telemetry import MAX_BYTES, validate_inventory

    expected = {"timing.json", "timing.json.sha256", "timing.json.complete", "telemetry"}
    if output.is_symlink() or {p.name for p in output.iterdir()} != expected:
        raise ValueError("timing output inventory differs")
    validate_inventory(output, complete=complete)
    if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > MAX_BYTES:
        raise ValueError("timing output exceeds 512 KiB")
    for path in output.iterdir():
        if path.name == "telemetry":
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("unsafe timing output")
    if (output / "timing.json").stat().st_size > TIMING_PAYLOAD_BYTES:
        raise ValueError("timing payload exceeds 512 KiB")
    verify_export(output / "timing.json")


def publish_report(output: Path, report: dict) -> None:
    """Enforce timing-only policy before publishing any payload or sidecar."""
    from research.bl_ppo.exports import export_json, limit_exports

    validate_report(report)
    if any(p.name != "telemetry" for p in output.iterdir()):
        raise ValueError("timing output inventory differs")
    with limit_exports(output, total_bytes=TIMING_PAYLOAD_BYTES, file_bytes=TIMING_PAYLOAD_BYTES):
        export_json(output / "timing.json", report)
    validate_output_inventory(output, complete=False)


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
        from research.bl_round30.timing_telemetry import WorkerDeadline, deliver_signal

        deliver_signal(WorkerDeadline())

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class Timer:
    """Nonoverlapping phase timings, never numerical return values."""

    def __init__(self, output: Path | None = None) -> None:
        from research.bl_round30.timing_telemetry import Telemetry

        self.telemetry = Telemetry(output) if output is not None else None
        self.seconds: dict[str, float] = {} if self.telemetry is None else self.telemetry.seconds

    def call(self, name: str, operation: Callable[[], Any]) -> Any:
        if name not in PHASES or name in self.seconds:
            raise ValueError("duplicate/foreign phase")
        if self.telemetry is not None:
            self.telemetry.start(name)
        started = time.monotonic()
        result = operation()
        self.seconds[name] = (
            time.monotonic() - started if self.telemetry is None else self.telemetry.end()
        )
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
        lambda: gradient_certificate(
            theta.detach().double(),
            cache,
            plan,
            recipe,
            max_seconds=SCHEDULE["certificate_cap_seconds"],
        ),
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


def _b8_masked_action_decision(base: Any, table: dict, *, device: str, synthetic: bool) -> None:
    """Make and discard one fixed-row, masked B8 six-action decision.

    Row zero is fixed before execution and is part of the already admitted
    training table.  In the real probe the CUDA observer wraps ``argmax`` and
    writes only caller/shape/device/count metadata; no result survives this
    function.  The CPU branch exists only for the hermetic synthetic test.
    """
    import torch

    if not synthetic and device != "cuda":
        raise ValueError("real timing probe requires one CUDA B8 decision")
    mask = torch.as_tensor(table["action_mask"][:1], dtype=torch.bool, device=device)
    from research.bl_round30.objective import resolved_action_mask

    mask = resolved_action_mask(mask)
    with torch.no_grad():
        q = base(
            *[
                torch.as_tensor(table[key][:1], device=device)
                for key in ("local", "global", "scalars")
            ]
        )
        if q.shape != (1, 6) or not torch.isfinite(q).all():
            raise ValueError("foreign or nonfinite B8 action tensor")
        action = q.masked_fill(~mask, -torch.inf).argmax(dim=1)
        if action.shape != (1,):
            raise ValueError("B8 action decision shape differs")
        if device == "cuda":
            torch.cuda.synchronize()
    # Deliberately discard Q-values and action before any later phase/export.
    del action, q, mask


def _worker_deadline_seconds(*, input_dir: Path, synthetic: bool, launch_age: float) -> float:
    """Reserve the final five minutes of the fixed 40-minute session.

    Production timing uses supervisor-owned immutable start/timeout metadata,
    while synthetic tests have no supervisor root and use the same fixed worker
    ceiling.  Neither metadata file or its contents enters the timing export.
    """
    remaining = float(DEADLINES["worker_seconds"]) - launch_age
    if synthetic:
        return max(0.001, remaining)
    root_value = os.environ.get("COLAB_CLI_ROOT")
    if not root_value:
        raise ValueError("real timing probe requires supervisor runtime metadata")
    root = Path(root_value).resolve()
    try:
        spec = json.loads((root / "spec.json").read_text())
        state = json.loads((root / "state.json").read_text())
        timeout = spec["timeout_seconds"]
        started = state["started_at"]
    except (OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("real timing probe runtime metadata is invalid") from exc
    if (
        isinstance(timeout, bool)
        or isinstance(started, bool)
        or not isinstance(timeout, (int, float))
        or not isinstance(started, (int, float))
        or not math.isfinite(float(timeout))
        or not math.isfinite(float(started))
        or timeout <= 0
        or timeout > DEADLINES["whole_session_seconds"]
    ):
        raise ValueError("real timing probe deadline metadata differs")
    remaining = min(
        remaining,
        float(started)
        + float(timeout)
        - time.time()
        - float(DEADLINES["cleanup_export_reserve_seconds"]),
    )
    return max(0.001, remaining)


def _run(
    input_dir: Path, output: Path, *, config_sha256: str, synthetic: bool, timer: Timer
) -> dict:
    """Only this function can publish; every phase must complete first."""
    from research.bl_ppo.exports import digest, export_json

    started = time.monotonic()
    launch_age = max(
        0.0, time.time() - float(os.environ.get("R30_TIMING_LAUNCH_EPOCH", time.time()))
    )
    config = json.loads((input_dir / "timing-config.json").read_text())
    if digest(input_dir / "timing-config.json") != config_sha256:
        raise ValueError("config binding differs")
    if (
        config["schema"] != "r30-timing-config/v2"
        or config["ruling_sha256"] != RULING_SHA
        or config.get("amendment_sha256") != AMENDMENT_SHA
        or config.get("certificate_amendment_sha256") != CERTIFICATE_AMENDMENT_SHA
        or config.get("observability_ruling_sha256") != OBSERVABILITY_SHA
        or config.get("deadlines") != DEADLINES
        or config["schedule"] != SCHEDULE
        or config["synthetic"] is not synthetic
    ):
        raise ValueError("fixed timing schedule differs")
    from research.bl_round30.timing_pack import identity_paths

    if set(config["source_hashes"]) != set(identity_paths()):
        raise ValueError("source allowlist differs")
    if any(digest(ROOT / name) != sha for name, sha in config["source_hashes"].items()):
        raise ValueError("source binding differs")
    cutoff = _worker_deadline_seconds(
        input_dir=input_dir, synthetic=synthetic, launch_age=launch_age
    )
    timer.telemetry.end()
    with (
        silence(),
        deadline(cutoff),
    ):
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
                if digest(input_dir / "timing-observability-ruling.md") != OBSERVABILITY_SHA:
                    raise ValueError("observability ruling differs")
                if (
                    digest(input_dir / "timing-certificate-amendment.md")
                    != CERTIFICATE_AMENDMENT_SHA
                ):
                    raise ValueError("certificate amendment differs")
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
        timer.call(
            "b8_masked_action_decision",
            lambda: _b8_masked_action_decision(base, table, device=device, synthetic=synthetic),
        )
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
        "amendment_sha256": AMENDMENT_SHA,
        "certificate_amendment_sha256": CERTIFICATE_AMENDMENT_SHA,
        "observability_ruling_sha256": OBSERVABILITY_SHA,
        "deadlines": DEADLINES,
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
    timer.telemetry.start("publication")
    publish_report(output, report)
    timer.telemetry.end()
    timer.telemetry.emit("complete")
    validate_output_inventory(output)
    return report


def run(input_dir: Path, output: Path, *, config_sha256: str, synthetic: bool = False) -> dict:
    """Persist phase records even on BaseException; publish a forecast only on success."""
    from research.bl_round30.timing_telemetry import Interrupted, deliver_signal

    if output.is_symlink():
        raise ValueError("unsafe output")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("nonempty timing output")
    timer = Timer(output)

    def terminated(signum: int, frame: Any) -> None:
        deliver_signal(Interrupted())

    previous = signal.signal(signal.SIGTERM, terminated)
    try:
        # The launcher timestamp accounts for startup before this module could emit.
        launch_age = max(
            0.0, time.time() - float(os.environ.get("R30_TIMING_LAUNCH_EPOCH", time.time()))
        )
        timer.telemetry.phase = "imports"
        timer.telemetry.started -= launch_age
        timer.telemetry.wall_started -= launch_age
        timer.telemetry.emit("start")
        timer.telemetry.end()
        timer.telemetry.start("preflight")
        return _run(
            input_dir, output, config_sha256=config_sha256, synthetic=synthetic, timer=timer
        )
    except BaseException as exc:
        timer.telemetry.emit("failure", error=exc)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


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
