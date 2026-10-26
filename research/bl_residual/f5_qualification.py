"""Sealed-disjoint CUDA qualification runner for the F5 transport canary.

This executable is separate from ``f5_job``: it never opens the selected-training
archive or produces a science acceptance. Its twelve sealed timing fixtures drive
the production GridBatchSim -> ego2s-bv8 -> B8 six-action decision path. Full
qualification also performs a separately labelled, timing-only optimizer
calibration on source-derived fixture observations; its output is not fit state,
science data, or a production forecast.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.bl_ppo.exports import digest, export_json, export_torch
from research.bl_residual import f5_timing
from research.bl_residual.f5_progress import QualificationProgress

SCHEMA = "bl-f5-cuda-qualification/v1"
FIXTURE_SCHEMA = "bl-f5-sealed-timing-fixtures/v1"
ACTION_COUNT = 6
PARITY_ATOL = 2.0e-4
PARITY_RTOL = 2.0e-4
LONG_BODY_MINIMUM = 700
NATURAL_HORIZON = 5_000
BRANCH_HORIZON = 300
WATCH_FIXTURE_FRAMES = 14_400


class QualificationRefusal(ValueError):
    """The disjoint CUDA qualification cannot produce an admissible receipt."""


def _refuse(reason: str) -> None:
    raise QualificationRefusal("F5 CUDA qualification refused: " + reason)


def _fixture_rows(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the exact registered timing-only allocation, including seeds."""
    if set(value) != {"schema", "fixtures"}:
        _refuse("fixture document has unknown or missing fields")
    if value.get("schema") != FIXTURE_SCHEMA:
        _refuse("fixture schema differs")
    rows = value.get("fixtures")
    if not isinstance(rows, list):
        _refuse("fixtures must be a list")
    try:
        canonical = f5_timing._timing_parents(rows)
    except f5_timing.TimingHold as exc:
        _refuse(str(exc))
    from research.bl_residual.f5_r10a_annex import allocate

    expected_seeds = {
        (row["allocation_index"], row["parent_id"]): row["seed"]
        for row in allocate(0)
        if row["timing_only"] is True
    }
    result: list[dict[str, Any]] = []
    for row in canonical:
        seed = row.get("seed")
        if type(seed) is not int or seed < 0 or seed >= 2**63:
            _refuse("fixture seed is missing or invalid")
        if expected_seeds.get((row["allocation_index"], row["parent_id"])) != seed:
            _refuse("fixture seed differs from frozen allocation")
        # Fixtures contain only identity/seed metadata.  Payload-like keys are
        # forbidden so a timing receipt cannot become a covert scientific look.
        allowed = {"allocation_index", "group", "group_index", "parent_id", "timing_only", "seed"}
        if set(row) != allowed:
            _refuse("fixture has unknown or payload field")
        result.append(dict(row))
    return result


def load_fixtures(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load a sealed descriptor without touching any selected-training payload."""
    if path.is_symlink() or not path.is_file():
        _refuse("fixture descriptor is absent or unsafe")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QualificationRefusal("F5 CUDA qualification refused: malformed fixture JSON") from exc
    if not isinstance(value, dict):
        _refuse("fixture document is not an object")
    return _fixture_rows(value), hashlib.sha256(raw).hexdigest()


def _require_cuda(device: str) -> None:
    import torch

    if device != "cuda:0" or not torch.cuda.is_available():
        _refuse("requires an available explicit cuda:0 device")


def _native_observation(seed: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build one real production observation from a timing-only deterministic world."""
    from src.simd_env.batch_sim import BatchSimConfig
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.grid_sim import GridBatchSim

    sim = GridBatchSim(
        BatchSimConfig(num_envs=1, num_snakes=6),
        seeds=[seed],
        train_mode=False,
        allow_respawn=False,
    )
    rows = np.asarray([[0, 0]], dtype=np.int64)
    observation = build_ego_raster(sim, EgoRasterConfig(version="bv8"), backend="numba", rows=rows)
    mask = np.asarray(sim.get_resolved_action_mask()[0, 0], dtype=bool)
    if observation["local"].shape[0] != 1 or mask.shape != (ACTION_COUNT,) or not mask.any():
        _refuse("source-native timing fixture has no legal six-action decision")
    return observation, mask


def _select_cuda(
    net: Any, observation: Mapping[str, np.ndarray], mask: np.ndarray
) -> tuple[np.ndarray, int]:
    """Run the genuine B8 fp32 CUDA forward and masked argmax selection."""
    import torch

    from src.model.ego2s_network import obs_tensors

    device = torch.device("cuda:0")
    with torch.inference_mode():
        q = net(*obs_tensors(observation, device)).float()
        if tuple(q.shape) != (1, ACTION_COUNT) or not bool(torch.isfinite(q).all()):
            _refuse("B8 CUDA forward did not produce finite six-action Q values")
        legal = torch.as_tensor(mask, device=device, dtype=torch.bool).unsqueeze(0)
        selected = int(q.masked_fill(~legal, float("-inf")).argmax(dim=1).item())
        torch.cuda.synchronize(device)
    return q.detach().cpu().numpy()[0].astype(np.float32, copy=False), selected


def _cpu_action(
    net: Any, observation: Mapping[str, np.ndarray], mask: np.ndarray
) -> tuple[np.ndarray, int]:
    import torch

    from src.model.ego2s_network import obs_tensors

    net = net.cpu().eval()
    with torch.inference_mode():
        q = net(*obs_tensors(observation, torch.device("cpu"))).float().numpy()[0]
    if q.shape != (ACTION_COUNT,) or not np.isfinite(q).all():
        _refuse("B8 CPU parity forward did not produce finite six-action Q values")
    return q.astype(np.float32, copy=False), int(np.where(mask, q, -np.inf).argmax())


def _redacted_export_payload(*, fixture_sha256: str, count: int) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "kind": "TIMING_ONLY_PRODUCER_EXPORT_NOT_A_FIT",
        "fixture_sha256": fixture_sha256,
        "fixtures_completed": count,
        "fit_usable": False,
    }


def _producer_exports(
    output: Path,
    *,
    fixture_sha256: str,
    count: int,
    assignment_environment: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Create scanner-recoverable producer paths, explicitly unusable as fit state.

    The raw same-assignment environment receipt is emitted alongside the producer
    paths so later fit snapshots bind only its digest instead of duplicating it.
    """
    payload = _redacted_export_payload(fixture_sha256=fixture_sha256, count=count)
    environment = (
        dict(assignment_environment)
        if assignment_environment is not None
        else {**payload, "kind": "TIMING_ONLY_CALIBRATION_ENVIRONMENT_PLACEHOLDER"}
    )
    names = (
        "paired/fit-C.pth",
        "paired/fit-W.pth",
        "paired/f5-state-C.pt",
        "paired/f5-state-W.pt",
    )
    # Use the same immutable Torch export mechanism as the actual producer.  A
    # deliberately non-loadable calibration envelope prevents these artifacts
    # from being substituted for an F5 fit checkpoint or full-state snapshot.
    return [
        export_json(output / "paired/fit-receipt.json", payload),
        export_json(output / "paired/assignment-environment.json", environment),
        *[export_torch(output / name, payload) for name in names],
    ]


def _bootstrap_admission(
    inputs: Path, output_root: Path, fixture_sha256: str, checkpoint: Path
) -> dict[str, Any]:
    """Admit the pre-control transport probe without weakening source/env checks."""
    from research.bl_residual import f5_environment, f5_job
    from research.bl_residual.f5_package import ROOT, source_inventory

    path = inputs / "bootstrap-control.json"
    if path.is_symlink() or not path.is_file():
        _refuse("bootstrap control is absent or unsafe")
    try:
        control = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise QualificationRefusal(
            "F5 CUDA qualification refused: malformed bootstrap control"
        ) from exc
    required = {
        "schema",
        "source_hashes",
        "environment_policy",
        "qualification_fixture_sha256",
        "b8_sha256",
    }
    if not isinstance(control, dict) or set(control) != required:
        _refuse("bootstrap control fields differ")
    if (
        control["schema"] != "bl-f5-bootstrap-qualification/v1"
        or control["source_hashes"] != source_inventory(ROOT)
        or control["environment_policy"] != f5_environment.policy_binding()
        or control["qualification_fixture_sha256"] != fixture_sha256
        or control["b8_sha256"] != f5_job.B8_SHA256
        or digest(checkpoint) != f5_job.B8_SHA256
    ):
        _refuse("bootstrap source, environment policy, fixture, or B8 binding differs")
    previous = output_root / "environment.json"
    if previous.is_symlink() or not previous.is_file():
        _refuse("bootstrap requires the preceding environment capture receipt")
    try:
        captured = json.loads(previous.read_text())
    except json.JSONDecodeError as exc:
        raise QualificationRefusal(
            "F5 CUDA qualification refused: malformed environment receipt"
        ) from exc
    actual = f5_job._environment_admission(None)
    return f5_environment.compare_environment(captured, actual, mode="same-assignment")


def _verify_uploaded_checkpoint_pool(binding: object) -> dict[str, str]:
    """Refuse ambient checkpoint paths; all evaluator opponents must be uploads."""
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    expected = {name: sha for name, sha in gid._context(1)["ds"].POOL}
    expected[gid.FRP3_S12[0]] = gid.FRP3_S12[1]
    if not isinstance(binding, dict) or binding != expected:
        _refuse("qualification checkpoint-pool binding differs")
    for name, expected_sha in expected.items():
        path = gid.CHECKPOINT_DIR / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected_sha:
            _refuse("uploaded qualification checkpoint pool differs")
    return expected


def _fixture_mix(row: Mapping[str, Any]) -> str:
    """Map the sealed timing stream to its registered evaluator roster mix."""
    stream = str(row["group"]).split("/", 1)[1]
    return {"watch6": "mixed", "scripted": "scripted", "frozen": "frozen", "mixed": "mixed"}[stream]


def _watch_world(seed: int, checkpoint: Path) -> tuple[Any, list[tuple[str, str]]]:
    """Construct the uploaded-pool Grid mirror used by the Watch6 source seam."""
    import copy
    from unittest.mock import patch

    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_rollout_labels_20261015 import bank_engine
    from src.simd_env import eval_engine as ee
    from src.simd_env.grid_sim import GridBatchSim

    opponent = gid.CHECKPOINT_DIR / gid.FRP3_S12[0]
    specs = [("checkpoint", str(opponent))] * 5
    with patch.object(ee, "_terminal_hero_sim_class", return_value=GridBatchSim):
        world = bank_engine.build_world(
            "b8_s3", {"seed": seed, "mix": "mixed", "specs": specs}, hero_checkpoint=checkpoint
        )
    for slot, policy in enumerate(world.opponents, 1):
        configured = copy.copy(policy)
        configured._configure_veto((slot,), "v8", 8.0, None)
        world.opponents[slot - 1] = configured
    return world, specs


def _source_root_observation(
    row: Mapping[str, Any], checkpoint: Path
) -> tuple[Any, dict[str, np.ndarray], float]:
    """Build one registered exact root and its B8 observation without rollout work."""
    from research.bl_residual import gameplay
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_rollout_labels_20261015 import bank_engine
    from research.t34_shadow.shadow import clone_world_exact, full_digest

    seed = int(row["seed"])
    mix = _fixture_mix(row)
    context = gid._context(1)
    root_started = time.monotonic()
    if str(row["group"]) == "timing/watch6":
        root, _ = _watch_world(seed, checkpoint)
        prepared = root.clone()
        if full_digest(clone_world_exact(prepared)) != full_digest(prepared):
            _refuse("Watch6 exact root clone differs")
    else:
        plan = bank_engine.planned_rosters(context, [seed], [mix])[0]
        root = bank_engine.build_world("b8_s3", plan, hero_checkpoint=checkpoint)
        # This exact digest/clone seam is the registered snapshot/restore timing
        # component; its digest itself is deliberately not emitted.
        gameplay._root_digest(plan, b8_path=checkpoint)
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    root_observation = build_ego_raster(
        root.sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
    )
    if root_observation["local"].shape != (1, 7, 31, 31):
        _refuse("exact timing root produced an invalid B8 observation")
    return root, root_observation, time.monotonic() - root_started


def _full_source_timing(
    row: Mapping[str, Any], checkpoint: Path, progress: QualificationProgress | None = None
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Measure the registered exact-world seams without exporting any outcome.

    The booleans are coverage evidence only.  They are intentionally insufficient
    to establish a scientific comparison and become a HOLD when a natural world
    terminates before its registered horizon.
    """
    from research.bl_residual import gameplay
    from research.t3_rollout_labels_20261015 import bank_engine

    if progress is not None:
        progress.phase("source-root")
    root, root_observation, t_root = _source_root_observation(row, checkpoint)
    seed = int(row["seed"])

    if progress is not None:
        progress.phase("source-h300")
    branch_started = time.monotonic()
    branch = root.clone()
    _, advisory, _ = bank_engine.choose_and_step(branch, capture_obs=False)
    legal = np.flatnonzero(np.asarray(advisory, dtype=bool)).tolist()
    if not legal:
        _refuse("registered branch fixture has no legal action")
    branches = bank_engine.rollout_all(branch, legal, depth=BRANCH_HORIZON, record_outcomes=False)
    t_branch = time.monotonic() - branch_started
    full_h300 = all(
        int(value.get("frames_executed", 0)) == BRANCH_HORIZON for value in branches.values()
    )

    if progress is not None:
        progress.phase("source-h5000")
    natural_started = time.monotonic()
    natural = gameplay._run_exact_arm_from_root(root, root.hero, horizon=NATURAL_HORIZON)
    t_nat = time.monotonic() - natural_started
    trace = natural["action_trace"]
    full_h5000 = natural["decisions"] == NATURAL_HORIZON
    long_body = bool(trace) and max(item["hero_length"] for item in trace) >= LONG_BODY_MINIMUM

    watch_frames = 0
    watch_seconds = 0.0
    if str(row["group"]) == "timing/watch6":
        from research.t34_pathb.adapter_v2 import RunSession

        if progress is not None:
            progress.phase("source-watch")
        watch_started = time.monotonic()
        # This is ``open_watch``'s Grid-continuation construction with its
        # ambient serving paths replaced by the manifest-uploaded, hash-checked
        # checkpoint pool under SNAKE_CKPT_DIR.
        watch_world, specs = _watch_world(seed, checkpoint)
        session = RunSession(watch_world, "bare", specs=specs)
        try:
            for _ in range(WATCH_FIXTURE_FRAMES):
                session.step()
                watch_frames += 1
        finally:
            session.close()
        watch_seconds = time.monotonic() - watch_started
    return {
        "t_root": t_root,
        "t_branch": t_branch,
        "t_nat": t_nat,
        "full_h300": full_h300,
        "full_h5000_natural": full_h5000,
        "long_body": long_body,
        "watch_fixture_frames": watch_frames,
        "watch_seconds": watch_seconds,
        # The actual source workloads measured above are the primitive timing
        # basis.  Expansion to registered stage counts belongs to f5_timing.
    }, root_observation


def _calibration_schema_sha256() -> str:
    """Return the shipped timing-only calibration schema digest."""
    path = Path(__file__).with_name("f5_calibration_schema_v1.json")
    if path.is_symlink() or not path.is_file():
        _refuse("calibration schema asset is absent or unsafe")
    return digest(path)


def run(
    inputs: Path,
    fixtures: Path,
    output: Path,
    *,
    device: str = "cuda:0",
    bootstrap: bool = False,
    transport_only: bool = False,
    ledger: Path | None = None,
    canary_tiny_workload: bool = False,
    progress: QualificationProgress | None = None,
) -> dict[str, Any]:
    """Execute all twelve timing-only source-native CUDA fixture workloads."""
    if progress is not None:
        progress.phase("admission")
    _require_cuda(device)
    # The supervisor owns ``$COLAB_CLI_ROOT/output``.  The executable owns a
    # receipt child and the real producer paths directly below that root.
    root = output
    receipt_dir = root / "qualification"
    if receipt_dir.exists() or (root / "paired").exists():
        raise FileExistsError("qualification receipt or producer output is not fresh")
    if inputs.is_symlink() or not inputs.is_dir():
        _refuse("input directory is absent or unsafe")
    checkpoint = inputs / "b8.pth"
    if checkpoint.is_symlink() or not checkpoint.is_file():
        _refuse("requires the pinned B8 checkpoint in inputs")
    if progress is not None:
        progress.phase("fixture-load")
    rows, fixture_sha256 = load_fixtures(fixtures)
    # Admission is deliberately real rather than a test seam: it binds the
    # current source and exact B8 input before the first CUDA decision.
    from research.bl_residual import f5_job

    if bootstrap:
        if progress is not None:
            progress.phase("bootstrap-admission")
        environment_evidence = _bootstrap_admission(inputs, root, fixture_sha256, checkpoint)
        if progress is not None:
            progress.phase("environment-capture")
        assignment = f5_job._environment_admission(None)
    else:
        if progress is not None:
            progress.phase("control-source-checks")
        control = f5_job._control(inputs)
        f5_job._source_admission(control)
        if control.get("qualification_fixture_sha256") != fixture_sha256:
            _refuse("control does not bind this timing-fixture descriptor")
        if (
            digest(checkpoint) != f5_job.B8_SHA256
            or control.get("qualification_b8_sha256") != f5_job.B8_SHA256
        ):
            _refuse("control or checkpoint differs from the fixed B8 input")
        if not transport_only:
            if progress is not None:
                progress.phase("checkpoint-pool")
            _verify_uploaded_checkpoint_pool(control.get("qualification_checkpoint_pool"))
        if progress is not None:
            progress.phase("environment-capture")
        assignment = f5_job._environment_admission(None)
        if progress is not None:
            progress.phase("environment-compare")
        environment_evidence = f5_job._environment_evidence(control, assignment)
    if progress is not None:
        progress.phase("initial-boundary")
    f5_job._assignment_boundary(assignment)
    from src.model.ego2s_network import load_ego2s_checkpoint, spec_for

    if progress is not None:
        progress.phase("cuda-load")
    model = load_ego2s_checkpoint(str(checkpoint), "cuda:0").eval()
    if spec_for(model) != "ego2s-bv8":
        _refuse("qualification requires the served ego2s-bv8 B8 checkpoint")
    root.mkdir(parents=True, exist_ok=True)
    if canary_tiny_workload and not transport_only:
        _refuse("tiny workload is CANARY_ONLY and cannot be a timing operation")
    # The full timing operation is resumable at the registered completed-unit
    # boundaries only.  Transport-only probes intentionally have no timing
    # units and cannot be mistaken for completed timing evidence.
    completed_units: list[dict[str, Any]] = []
    verified_units: dict[int | str, dict[str, Any]] = {}
    resume_binding: dict[str, str] | None = None
    if not transport_only or canary_tiny_workload:
        from research.bl_residual import f5_timing_resume

        if ledger is None:
            ledger = inputs / "verified-ledger.json"
        if ledger.is_symlink() or not ledger.is_file():
            _refuse("full timing requires a hash-bound verified ledger")
        if control.get("ledger_sha256") != digest(ledger):
            _refuse("timing ledger differs from the sealed control")
        sealed = f5_job._identity(control)
        input_sha256 = control.get("timing_input_sha256")
        if not isinstance(input_sha256, str) or len(input_sha256) != 64:
            _refuse("full timing requires a sealed non-ledger input inventory digest")
        resume_binding = f5_timing_resume.identity(
            namespace="bl-residual-f5-timing-recovery-20261026",
            allocation=sealed.allocation_sha256,
            source_sha256=sealed.source_sha256,
            environment_sha256=sealed.environment_sha256,
            input_sha256=input_sha256,
        )
        try:
            verified_units = f5_timing_resume.load_ledger(ledger, resume_binding)
            if canary_tiny_workload:
                if verified_units:
                    _refuse("family canary requires the shared empty initial ledger")
                resume_binding = {
                    **resume_binding,
                    "namespace": resume_binding["namespace"] + "/canary",
                }
        except f5_timing_resume.ResumeRefusal as exc:
            _refuse(str(exc))
    samples: list[dict[str, Any]] = []
    started = time.monotonic()
    # The canary uses two tiny units: a preverified one exercises the scanner
    # skip path and the following missing unit exercises the real producer.
    active_rows = rows[:2] if canary_tiny_workload else rows
    skipped_keys = list(verified_units)
    # Revisit the first produced tiny fixture after scanning it. This proves an
    # actual skip through the ordinary loop, then produces the second fixture.
    loop_rows = [active_rows[0], *active_rows] if canary_tiny_workload else active_rows
    for completed, row in enumerate(loop_rows):
        if canary_tiny_workload and completed == 1:
            # Exercise the identical producer -> immutable ledger -> scanner
            # path in this assignment.  The original uploaded ledger is empty;
            # this second scanner pass must skip the just-exported tiny 672.
            from research.bl_residual import f5_timing_resume

            f5_timing_resume.build_ledger(
                root,
                resume_binding,
                completed_units,
                name="timing-canary/verified-ledger.json",
                canary_only=True,
            )
            verified_units.update(
                f5_timing_resume.load_ledger(
                    root / "timing-canary/verified-ledger.json",
                    resume_binding,
                    canary_only=True,
                )
            )
        key = int(row["allocation_index"])
        if key in verified_units:
            payload = verified_units[key]
            if payload.get("allocation_index") != key:
                _refuse("verified fixture record differs from allocation")
            if key not in skipped_keys:
                skipped_keys.append(key)
            if not canary_tiny_workload:
                samples.append(payload)
            continue
        if progress is not None:
            progress.phase("cuda-parity", completed=completed)
        fixture_started = time.monotonic()
        observation, mask = _native_observation(int(row["seed"]))
        feature_seconds = time.monotonic() - fixture_started
        cuda_started = time.monotonic()
        cuda_q, cuda_action = _select_cuda(model, observation, mask)
        cuda_seconds = time.monotonic() - cuda_started
        cpu_started = time.monotonic()
        cpu_q, cpu_action = _cpu_action(model, observation, mask)
        cpu_seconds = time.monotonic() - cpu_started
        model = model.cuda().eval()
        if (
            not np.allclose(cuda_q, cpu_q, rtol=PARITY_RTOL, atol=PARITY_ATOL)
            or cuda_action != cpu_action
        ):
            _refuse("same-source CPU/CUDA B8 parity differs")
        if transport_only:
            timing = {}
        else:
            timing, _ = _full_source_timing(row, checkpoint, progress)
        total = time.monotonic() - fixture_started
        # All fields are elapsed measurements of this exact fixture.  Forecast
        # expansion remains the validator's job; this runner never invents a
        # feasibility claim from a missing fixture.
        sample = {
            **row,
            "stream": str(row["group"]).split("/", 1)[1],
            "feature_seconds": feature_seconds,
            "cuda_selection_seconds": cuda_seconds,
            "cpu_parity_seconds": cpu_seconds,
            "elapsed_seconds": total,
            "parity": "PASS",
            "q_shape": [1, ACTION_COUNT],
            **timing,
        }
        samples.append(sample)
        if resume_binding is not None:
            from research.bl_residual import f5_timing_resume

            completed_units.append(
                f5_timing_resume.publish_unit(
                    root, resume_binding, key, sample, canary_only=canary_tiny_workload
                )
            )
        if progress is not None:
            progress.phase("fixture-complete", completed=completed + 1)
    calibration = None
    if not transport_only:
        from research.bl_residual import f5_timing_resume
        from research.bl_residual.f5_calibration import run as run_calibration

        existing = verified_units.get(f5_timing_resume.CALIBRATION_KEY)
        if existing is not None:
            calibration = existing
        else:
            if progress is not None:
                progress.phase("calibration-source")
            calibration = run_calibration(
                model,
                checkpoint,
                rows,
                fixture_sha256,
                source_hashes=control["source_hashes"],
                environment_sha256=resume_binding["environment_sha256"],
                progress=progress,
            )
            completed_units.append(
                f5_timing_resume.publish_unit(
                    root, resume_binding, f5_timing_resume.CALIBRATION_KEY, calibration
                )
            )
    # The after-boundary covers every actual qualification workload, including
    # the calibration fit; transport-only performs no calibration workload.
    if progress is not None:
        progress.phase("final-boundary", completed=len(samples))
    f5_job._assignment_boundary(assignment)
    if progress is not None:
        progress.phase("export", completed=len(samples))
    exports = _producer_exports(
        root,
        fixture_sha256=fixture_sha256,
        count=len(samples),
        assignment_environment=assignment,
    )
    if resume_binding is not None:
        new_keys = [entry["key"] for entry in completed_units]
        required_keys = (
            {672, 673}
            if canary_tiny_workload
            else {*range(672, 684), f5_timing_resume.CALIBRATION_KEY}
        )
        if set(skipped_keys) | set(new_keys) != required_keys:
            _refuse("completed-unit union does not cover the registered operation")
        export_json(
            root / "resume" / "verified-skip-proof.json",
            {
                "schema": "bl-f5-timing-verified-skip-proof/v1",
                "classification": "CANARY_ONLY" if canary_tiny_workload else "FULL_TIMING_UNITS",
                "identity": resume_binding,
                "input_ledger_sha256": digest(ledger),
                "skipped_keys": skipped_keys,
                "new_unit_exports": completed_units,
                "complete_keys": (
                    [*range(672, 674)]
                    if canary_tiny_workload
                    else [*range(672, 684), f5_timing_resume.CALIBRATION_KEY]
                ),
                "all_completed_durations_are_uninterrupted": True,
            },
        )
    report = {
        "schema": SCHEMA,
        "status": (
            "COMPLETE_TRANSPORT_ONLY_NOT_TIMING_QUALIFIED"
            if transport_only
            else "COMPLETE_L4_TIMING_PENDING_MAC_QUALIFICATION"
        ),
        "fixture_schema": FIXTURE_SCHEMA,
        "fixture_sha256": fixture_sha256,
        "b8_sha256": digest(checkpoint),
        "environment_evidence": environment_evidence,
        "device": device,
        "precision": "fp32",
        "action_count": ACTION_COUNT,
        "parity_atol": PARITY_ATOL,
        "parity_rtol": PARITY_RTOL,
        "samples": samples,
        "elapsed_seconds": time.monotonic() - started,
        "producer_exports": exports,
        "fit_or_optimizer_executed": calibration is not None,
        "timing_only_calibration": calibration,
        "completed_timing_units": completed_units,
        "mac_timing_status": (
            "NOT_MEASURED_TRANSPORT_ONLY" if transport_only else "HOLD_REQUIRES_MAC_COMPANION"
        ),
    }
    if canary_tiny_workload:
        # This path deliberately uses the real scanner admission and real CUDA
        # producer seam, but marks its shortened record ineligible for either
        # a ledger or the 13-unit reducer.
        from research.bl_residual import f5_timing_resume

        if not completed_units:
            _refuse("tiny canary must execute a missing producer unit")
        report["canary_only"] = completed_units
    receipt_dir.mkdir(parents=True)
    export_json(receipt_dir / "qualification.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--transport-only", action="store_true")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--canary-tiny-workload", action="store_true")
    args = parser.parse_args()
    progress = QualificationProgress()
    progress.start()
    success = False
    try:
        run(
            args.inputs,
            args.fixtures,
            args.output,
            device=args.device,
            bootstrap=args.bootstrap,
            transport_only=args.transport_only,
            ledger=args.ledger,
            canary_tiny_workload=args.canary_tiny_workload,
            progress=progress,
        )
        success = True
    except BaseException as exc:
        progress.failure(exc)
        raise
    finally:
        progress.close(success=success)


if __name__ == "__main__":
    main()
