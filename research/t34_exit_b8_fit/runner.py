"""High-level, artifact-owning assembly for the P6 fit entrypoint.

The low-level optimizer and streaming loader stay in :mod:`train`; this module
binds only immutable engine factories, the read-only B8 checkpoint, and durable
output artifacts.  It never chooses a public URL or dispatches a runtime.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
from pathlib import Path
from typing import Any

import torch

from . import train
from .cache import CacheValidationError
from .data import ReplayDivergenceError
from .diagnostics import FitTripwire
from .interface import FakeReplaySource


class RunnerRefused(RuntimeError):
    """A factory/binding/artifact seam is incomplete."""


def _factory(value: str):
    module, separator, name = value.partition(":")
    if not separator or not module or not name:
        raise RunnerRefused("factory must be module:function")
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise RunnerRefused(f"factory unavailable: {value}") from exc


def _load_b8(checkpoint: Path, device: str) -> torch.nn.Module:
    from src.model.ego2s_network import load_ego2s_checkpoint

    from .network import ARCH, load_exit_fit_tower_checkpoint

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if blob.get("arch") == ARCH:
        return load_exit_fit_tower_checkpoint(checkpoint, torch.device(device))
    return load_ego2s_checkpoint(checkpoint, torch.device(device))


def _save_selected(out_dir: Path, result: dict[str, Any], model: torch.nn.Module) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=False)
    state = result.pop("selected_state_dict", None)
    if state is None:
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    checkpoint = out_dir / "selected_fit.pth"
    model.load_state_dict(state)
    from src.model.ego2s_network import save_ego2s_checkpoint

    from .network import ExitFitTowerNet, save_exit_fit_tower_checkpoint

    meta = {
        "selected": result.get("selected"),
        "updates": result.get("selected_updates", result.get("updates")),
        "smoke_only_fake_engine": result.get("smoke_only_fake_engine", False),
    }
    if isinstance(model, ExitFitTowerNet):
        save_exit_fit_tower_checkpoint(checkpoint, model, meta=meta, strip_aux=True)
    else:
        save_ego2s_checkpoint(checkpoint, model, meta=meta)
    training = out_dir / "selected_training_state.pth"
    torch.save(
        {
            "selected_state_dict": state,
            "optimizer_state_dict": result.pop("selected_optimizer_state", None),
            "torch_rng_state": result.pop("selected_torch_rng_state", None),
            "updates": result.pop("selected_updates", result.get("updates")),
            "selected_epoch": result.get("selected", {}).get("epoch"),
            "exact_resume_qualified": False,
        },
        training,
    )
    for artifact in (checkpoint, training):
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        artifact.with_name(artifact.name + ".sha256").write_text(digest + f"  {artifact.name}\n")
        artifact.with_name(artifact.name + ".complete").write_text("complete\n")
    return {
        "path": str(Path(out_dir.name) / checkpoint.name),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_state": str(Path(out_dir.name) / training.name),
        "training_state_sha256": hashlib.sha256(training.read_bytes()).hexdigest(),
    }


def run(
    *,
    exports: Path,
    checkpoint: Path | None,
    output_dir: Path,
    worlds: int,
    updates: int,
    device: str,
    mode: str,
    epochs: int,
    fake_engine: bool,
    binding: dict[str, Any] | None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Run either the real P6 preflight or a bounded production fit artifact."""
    seed = int((binding or {}).get("seed", 0))
    torch.manual_seed(seed)
    if fake_engine:
        from src.model.ego2s_network import Ego2sNet

        model = _load_b8(checkpoint, device) if checkpoint is not None else Ego2sNet(7, 60)
        source, probe = FakeReplaySource(worlds=worlds, rows_per_world=4), None
    else:
        if checkpoint is None or binding is None:
            raise RunnerRefused("real fit requires a B8 checkpoint and immutable binding")
        model = _load_b8(checkpoint, device)
        from .network import ExitFitTowerNet

        if binding.get("arm") == "S1a":
            if not isinstance(model, ExitFitTowerNet):
                model = ExitFitTowerNet(model)
            elif binding.get("aux_head", True) and model.free_region_head is None:
                model.free_region_head = torch.nn.Linear(256, 1)
                model.new_parameter_names = tuple(
                    name
                    for name, _ in model.named_parameters()
                    if name.startswith("free_region_head.")
                )
        elif isinstance(model, ExitFitTowerNet):
            raise RunnerRefused("S1b must start from a no-tower checkpoint")
    model = model.to(device)
    previous = copy.deepcopy(model).eval()
    if not fake_engine:
        source = _factory(str(binding["source_factory"]))(
            exports=exports, binding=binding, previous_model=previous
        )
        probe = _factory(str(binding["probe_factory"]))(binding=binding, previous_model=previous)
    # Bind immutable data/recipe inputs plus every repository Python source.
    # A replay factory supplied outside the repository is bound separately.
    repo = Path(__file__).resolve().parents[2]
    paths = sorted(
        {path for base in (repo / "src", repo / "research") for path in base.rglob("*.py")}
    )
    closure = hashlib.sha256()
    for path in paths:
        closure.update(str(path.relative_to(repo)).encode())
        closure.update(path.read_bytes())
    factory_file = inspect.getsourcefile(type(source))
    include_tower = (binding or {}).get("arm") == "S1a"
    cache_identity = {
        "binding": binding,
        "checkpoint_sha256": (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None
        ),
        "python_source_sha256": closure.hexdigest(),
        "source_file_sha256": (
            hashlib.sha256(Path(factory_file).read_bytes()).hexdigest() if factory_file else None
        ),
        "fake_engine": fake_engine,
        "include_tower": include_tower,
    }
    cache_args = {
        "cache_dir": cache_dir or output_dir.parent / "observation-cache",
        "cache_identity": cache_identity,
        "include_tower": include_tower,
    }
    try:
        if mode == "preflight":
            result = train.run_fit_preflight(
                source=source,
                model=model,
                previous_model=previous,
                device=device,
                worlds=worlds,
                updates=updates,
                batch_size=4 if fake_engine else 1024,
                allow_fake=fake_engine,
                probe_hook=probe,
                seed=seed,
                **cache_args,
            )
        else:
            result = train.fit(
                source=source,
                model=model,
                previous_model=previous,
                device=device,
                epochs=epochs,
                probe_hook=probe,
                allow_fake=False,
                seed=seed,
                **cache_args,
            )
    except (FitTripwire, ReplayDivergenceError, CacheValidationError, ValueError) as exc:
        return {
            "completed": False,
            "tripwire": str(exc),
            "probes_ok": False,
            "smoke_only_fake_engine": fake_engine,
        }
    result = dict(result)
    result["seed"] = seed
    result["completed"] = bool(result.get("completed", True))
    result["tripwire"] = result.get("tripwire")
    result["probes_ok"] = bool(result.get("probes_ok", not fake_engine))
    if result["completed"] and not result["tripwire"]:
        result["selected_checkpoint"] = _save_selected(output_dir, result, model)
    else:
        # These are binary training artifacts. A stopped run retains only the
        # JSON-safe stop reason/metrics and never writes a candidate.
        result.pop("selected_state_dict", None)
        result.pop("selected_optimizer_state", None)
        result.pop("selected_torch_rng_state", None)
    result["smoke_only_fake_engine"] = fake_engine
    return result
