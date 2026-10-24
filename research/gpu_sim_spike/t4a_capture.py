"""Observe actual policy calls without changing their numerical execution order."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import torch

from research.gpu_sim_spike import device_policy


class Capture:
    """Capture real inputs, raw Q, every v8 result, and masks on the device.

    Model identity is not seat identity: multiple seats may call one shared net.
    Hooks assign outputs in the ordinary hero-then-opponent call order and check
    that each call actually used the module belonging to that seat.
    """

    def __init__(self, case: tuple) -> None:
        self.sim, self.policy = case[:2]
        self.slots = [(0, self.policy.b8_network)] + [
            (slot, net)
            for slot, net in enumerate(self.policy.opponent_networks, 1)
            if net is not None
        ]
        self.stack = ExitStack()
        self.values: dict[str, torch.Tensor] = {}
        self.calls = 0
        self.v8_calls = 0
        self.mask_calls = 0
        self.config_digest = self._configuration()

    def _configuration(self) -> torch.Tensor:
        cfg = dataclasses.asdict(self.sim.cfg)
        cfg.pop("num_envs", None)
        runtime = self.policy.vector_runtime
        values: dict[str, Any] = {
            "cfg": cfg,
            "runtime_class": type(runtime).__name__,
            "runtime_params": repr(runtime.params),
            "anchor_kinds": self.policy.anchor_kinds,
            "lambda": self.policy.v8_lambda,
        }
        for name in ("_cpu_sector_rx", "_cpu_sector_ry", "_cpu_sector_lut", "_free_ratio"):
            value = getattr(runtime, name, None)
            values[name] = (
                hashlib.sha256(value.contiguous().cpu().numpy().tobytes()).hexdigest()
                if isinstance(value, torch.Tensor)
                else repr(value)
            )
        values["tables"] = [
            (
                hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()
                if isinstance(t, torch.Tensor)
                else repr(t)
            )
            for t in self.policy.anchor_tables
        ]
        raw = hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).digest()
        return torch.tensor(list(raw), dtype=torch.uint8, device=self.sim.device)

    def begin_tick(self) -> None:
        self.values = {}
        self.calls = self.v8_calls = self.mask_calls = 0

    def _model_hook(self, module: Any, inputs: tuple, output: torch.Tensor) -> None:
        if self.calls >= len(self.slots):
            raise ValueError("unexpected learned-policy forward")
        slot, expected = self.slots[self.calls]
        if module is not expected:
            raise ValueError("learned-seat call order/module identity differs")
        self.calls += 1
        if (
            output.shape != (self.sim.E, 6)
            or output.dtype != torch.float32
            or not bool(torch.isfinite(output).all())
        ):
            raise ValueError("learned-seat Q shape/nonfinite evidence failure")
        self.values[f"output.q.{slot}"] = output.detach().clone()
        for index, tensor in enumerate(inputs):
            if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != self.sim.E:
                raise ValueError("learned observation is not aligned to physical rows")
            self.values[f"output.obs.{slot}.{index}"] = tensor.detach().clone()

    def __enter__(self) -> "Capture":
        seen = set()
        for _slot, net in self.slots:
            if id(net) not in seen:
                seen.add(id(net))
                handle = net.register_forward_hook(self._model_hook)
                self.stack.callback(handle.remove)
        original_v8 = device_policy.v8_decide

        def v8(
            sim: Any,
            rows: torch.Tensor,
            q: torch.Tensor,
            mask: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            value = original_v8(sim, rows, q, mask, *args, **kwargs)
            if sim is self.sim:
                index = self.v8_calls
                self.v8_calls += 1
                if rows.shape != (sim.E, 2) or not torch.equal(
                    rows[:, 0], torch.arange(sim.E, device=sim.device)
                ):
                    raise ValueError("unexpected compacted v8 row order")
                self.values[f"output.v8.{index}.mask"] = mask.detach().clone()
                for field in dataclasses.fields(value):
                    tensor = getattr(value, field.name)
                    if isinstance(tensor, torch.Tensor):
                        if tensor.ndim == 0 or tensor.shape[0] != sim.E:
                            raise ValueError("uninventoried v8 tensor axis")
                        self.values[f"output.v8.{index}.{field.name}"] = tensor.detach().clone()
                    elif tensor is not None:
                        raise ValueError("uninventoried v8 result value")
            return value

        original_mask = self.sim.advisory_mask

        def advisory(*args: Any, **kwargs: Any) -> torch.Tensor:
            value = original_mask(*args, **kwargs)
            self.values[f"output.advisory.{self.mask_calls}"] = value.detach().clone()
            self.mask_calls += 1
            return value

        advisory.__t4a_instrumentation__ = True
        self.stack.enter_context(patch.object(device_policy, "v8_decide", v8))
        self.stack.enter_context(patch.object(self.sim, "advisory_mask", advisory))
        original_actions = self.policy.actions

        def actions(*args: Any, **kwargs: Any) -> torch.Tensor:
            value = original_actions(*args, **kwargs)
            self.values["output.actions"] = value.detach().clone()
            return value

        self.stack.enter_context(patch.object(self.policy, "actions", actions))
        runtime = self.policy.vector_runtime
        original_prepare = runtime.prepare

        def prepare(sim: Any, rows_mask: Any = None) -> Any:
            affected = runtime.controlled
            if rows_mask is not None:
                affected = affected & rows_mask[:, None]
            ready = affected & sim.alive
            respawned = (sim.alive & ~runtime._alive_post).any(1)
            carried = ready & runtime._carry_valid & ~respawned[:, None]
            fresh = ready & ~carried
            before_f = torch.as_tensor(runtime.fresh_rows, device=sim.device).clone()
            before_c = torch.as_tensor(runtime.carried_rows, device=sim.device).clone()
            result = original_prepare(sim, rows_mask)
            if not (
                torch.equal(runtime.fresh_rows - before_f, fresh.sum())
                and torch.equal(runtime.carried_rows - before_c, carried.sum())
            ):
                raise ValueError("E-dependent feature accounting does not reconcile")
            self.values["accounting.reconciled"] = torch.ones(
                (sim.E,), dtype=torch.bool, device=sim.device
            )
            return result

        prepare.__t4a_instrumentation__ = True
        self.stack.enter_context(patch.object(runtime, "prepare", prepare))
        return self

    def __exit__(self, *args: Any) -> None:
        self.stack.__exit__(*args)

    def fields(self, rows: tuple[int, ...]) -> dict[str, torch.Tensor]:
        if self.calls != len(self.slots) or self.v8_calls != len(self.slots) or not self.mask_calls:
            raise ValueError("required policy/observation/v8/mask call was omitted")
        row_index = torch.tensor(rows, device=self.sim.device, dtype=torch.long)
        values = {
            name: value.index_select(0, row_index).contiguous()
            for name, value in sorted(self.values.items())
        }
        # Lifecycle flags are scalars but independent of physical E.
        values["lifecycle.sim_prepared"] = torch.tensor(self.sim._prepared, device=self.sim.device)
        values["lifecycle.vector_prepared"] = torch.tensor(
            self.policy.vector_prepared, device=self.sim.device
        )
        values["configuration.sha256"] = self.config_digest
        return values

    def q_tape(self, actions: torch.Tensor, rows: tuple[int, ...]) -> dict[str, Any]:
        index = torch.tensor(rows, device=self.sim.device, dtype=torch.long)
        return {
            "actions_hex": actions.index_select(0, index)
            .contiguous()
            .cpu()
            .numpy()
            .tobytes()
            .hex(),
            "q_hex": {
                str(slot): self.values[f"output.q.{slot}"]
                .index_select(0, index)
                .contiguous()
                .cpu()
                .numpy()
                .tobytes()
                .hex()
                for slot, _net in self.slots
            },
        }
