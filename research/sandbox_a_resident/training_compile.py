"""Sandbox-local compiled tensor blocks for resident PPO training.

The mutable simulator remains outside graphs.  This adapter compiles only the
current A-3 pure-tensor feature/observation ports and actor forward callable;
the actor module itself is never wrapped, so its state dict and export format
remain ordinary ego2s-compatible bytes.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from research.sandbox_a_resident.runtime_tools import CompiledBlock, PhaseProfiler
from research.sandbox_a_resident.tensor_blocks import TensorBlocks


class _EagerBlock:
    """CompiledBlock-shaped callable for an explicitly requested eager mode."""

    compiled = None

    def __init__(self, eager: Callable, report: dict[str, Any], name: str) -> None:
        self.eager = eager
        self.entry = report[name] = {
            "status": "disabled_eager",
            "mode": "none",
            "backend": None,
            "fallback_reason": "requested mode=none",
            "validation_policy": "not applicable: explicit eager mode",
            "compile_factory_seconds": 0.0,
            "compile_specialization_seconds": 0.0,
            "compile_specialization_calls": 0,
            "validation_seconds": 0.0,
            "validation_calls": 0,
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.eager(*args, **kwargs)


class TrainingCompile:
    """Compile static resident blocks with per-block fallback and validation reports."""

    def __init__(
        self,
        actor: torch.nn.Module,
        task: Any | None = None,
        *,
        mode: str = "default",
        backend: str | Callable | None = None,
    ) -> None:
        if mode not in ("default", "reduce-overhead", "max-autotune", "none"):
            raise ValueError("unknown compilation mode")
        self.actor, self.task, self.mode = actor, task, mode
        self.report: dict[str, Any] = {}
        # CompiledBlock catches factory/specialization/validation errors and
        # permanently records an explicit eager fallback for each block.
        if mode == "none":
            self.actor_block = _EagerBlock(actor.forward, self.report, "actor")
        else:
            self.actor_block = CompiledBlock(
                actor.forward, mode, self.report, "actor", backend=backend
            )
        self.blocks: TensorBlocks | None = None
        self.v8_block: CompiledBlock | _EagerBlock | None = None
        self.observation_block: CompiledBlock | _EagerBlock | None = None
        if task is not None:
            self.blocks = TensorBlocks(task.sim)
            if mode == "none":
                self.v8_block = _EagerBlock(self.blocks.v8_features, self.report, "v8_features")
                self.observation_block = _EagerBlock(
                    self.blocks.observation, self.report, "observation"
                )
            else:
                self.v8_block = CompiledBlock(
                    self.blocks.v8_features, mode, self.report, "v8_features", backend=backend
                )
                self.observation_block = CompiledBlock(
                    self.blocks.observation, mode, self.report, "observation", backend=backend
                )

    def actor_forward(
        self, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        """Compiled/eager actor callable; actor serialization remains untouched."""
        return self.actor_block(local, glob, scalars)

    @torch.inference_mode()
    def observe(self, profiler: PhaseProfiler | None = None) -> tuple[torch.Tensor, ...]:
        """Build the current hero B8 observation through pure tensor ports."""
        if (
            self.task is None
            or self.blocks is None
            or self.v8_block is None
            or self.observation_block is None
        ):
            raise RuntimeError("task-backed observation requires a ResidentTrainingTask")
        sim = self.task.sim
        sim.prepare()
        rows = torch.stack((self.task._world, torch.zeros_like(self.task._world)), dim=1)
        legal = sim.advisory_mask()[:, 0]
        state = self.blocks.state(sim)
        all_occ, other_occ, own_cells, own_valid = self.blocks.v8_inputs(sim, rows)
        context = profiler.phase("v8_features") if profiler is not None else _null_phase()
        with context:
            features = self.v8_block(
                *state,
                rows=rows,
                mask=legal,
                all_occ=all_occ,
                other_occ=other_occ,
                own_cells=own_cells,
                own_valid=own_valid,
            )
        ttl, region, food_grid, heads = self.blocks.observation_inputs(sim, rows)
        context = profiler.phase("observation") if profiler is not None else _null_phase()
        with context:
            local, glob, scalars = self.observation_block(
                *state,
                ttl=ttl,
                region=region,
                food_grid=food_grid,
                heads=heads,
                rows=rows,
                features=features,
            )
        return local, glob, scalars, legal

    def gradient_probe(
        self, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
    ) -> dict[str, Any]:
        """Check eager/compiled logits and parameter gradients before the 100M run.

        The probe changes only transient ``.grad`` fields, then clears them.
        It is an initialization-time self-check, never a throughput claim.
        """
        if self.actor_block.compiled is None:
            return {
                "status": self.report["actor"]["status"],
                "reason": self.report["actor"]["fallback_reason"],
            }
        self.actor.zero_grad(set_to_none=True)
        eager = self.actor(local, glob, scalars)
        eager.square().mean().backward()
        eager_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in self.actor.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        self.actor.zero_grad(set_to_none=True)
        compiled = self.actor_forward(local, glob, scalars)
        compiled.square().mean().backward()
        torch.testing.assert_close(compiled, eager.detach(), rtol=2e-5, atol=2e-5)
        for name, parameter in self.actor.named_parameters():
            if name in eager_grads:
                torch.testing.assert_close(parameter.grad, eager_grads[name], rtol=2e-5, atol=2e-5)
        self.actor.zero_grad(set_to_none=True)
        self.report["actor"][
            "gradient_check"
        ] = "eager/compiled logits and trainable gradients match"
        return {"status": self.report["actor"]["status"], "rows": int(local.shape[0])}

    def runtime_self_check(self, rows: int = 32) -> dict[str, Any]:
        """Run the fixed small actor validation used before training allocation."""
        if rows < 1:
            raise ValueError("rows must be positive")
        device = next(self.actor.parameters()).device
        generator = torch.Generator(device=device).manual_seed(9182)
        local = torch.randint(
            0, 256, (rows, 7, 31, 31), device=device, dtype=torch.uint8, generator=generator
        )
        glob = torch.randint(
            0, 256, (rows, 4, 37, 37), device=device, dtype=torch.uint8, generator=generator
        )
        scalars = torch.rand((rows, 60), device=device, generator=generator)
        return self.gradient_probe(local, glob, scalars)


class _null_phase:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False
