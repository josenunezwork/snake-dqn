"""Sandbox-local food-maintenance adapter for the resident A-3 assay.

The certified ``gpu_sim_spike.DeviceSim`` remains unmodified.  This subclass
only recognizes the empty-food rows created by the resident assay reset and
places their first ambient-food stream in one ordered batched append.  It runs
at the inherited ``prepare()`` food phase, so reset state and the observable
step lifecycle are unchanged.
"""

from __future__ import annotations

import torch

from research.gpu_sim_spike.device_sim import DeviceSim as _CertifiedDeviceSim

# Explicit environment-axis ownership.  Do not infer it from ``shape[0]``:
# the resident recipe deliberately permits E == S, where slot constants have
# the same leading size but must never be compacted as environment state.
_ENV_TENSOR_FIELDS = (
    "bodies",
    "head_ptr",
    "seg_count",
    "length",
    "alive",
    "direction",
    "boost_frames",
    "frames_since_food",
    "respawn_timer",
    "_reward_prev_length",
    "frame",
    "rng_counter",
    "rng_key",
    "food_cells",
    "food_present",
    "corpse_food",
    "food_count",
    "_last_reward",
    "_last_done",
    "_last_death_cause",
    "_last_kills",
    "_last_food_ate",
    "_last_ambient_food_ate",
    "_last_corpse_food_ate",
    "_last_transition_valid",
    "_boosted_this_step",
    "terminal_slots",
    "capacity_overflow",
)


class ResidentDeviceSim(_CertifiedDeviceSim):
    """Use a batched, exact ordered append for empty-row initial food.

    ``DeviceSim._maintain_food_impl`` draws exactly ``max_food`` counter
    candidates for every environment per maintenance phase.  For a newly
    reset empty row, the number of candidates equals the target, so the whole
    candidate stream can be generated at once and passed to ``_append_many``.
    The latter retains the original FIFO acceptance/rejection order (including
    duplicate and occupied-cell rejection).  Nonempty rows retain the frozen
    implementation verbatim.
    """

    def _row_subset(self, rows: torch.Tensor) -> _CertifiedDeviceSim:
        """Return a private compact view over complete environment rows."""
        sub = object.__new__(_CertifiedDeviceSim)
        sub.__dict__ = self.__dict__.copy()
        for name in _ENV_TENSOR_FIELDS:
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                setattr(sub, name, value.index_select(0, rows))
        sub.E = int(rows.numel())
        return sub

    def _copy_food_rows(self, rows: torch.Tensor, sub: _CertifiedDeviceSim) -> None:
        """Copy only changed food state back to its original environment rows."""
        for name in ("food_cells", "food_present", "corpse_food", "food_count"):
            getattr(self, name).index_copy_(0, rows, getattr(sub, name))

    def _append_initial_food(self, rows: torch.Tensor, trials: int) -> None:
        """Append the old counter stream for blank rows in one ordered batch."""
        sub = self._row_subset(rows)
        counters = sub.rng_counter[:, None] + torch.arange(
            1, trials + 1, dtype=torch.long, device=self.device
        )
        mixed = (
            sub.rng_key[:, None] * 6364136223846793005 + counters * 1442695040888963407
        ) & 0x7FFF_FFFF_FFFF_FFFF
        candidates = torch.stack(
            (mixed.remainder(sub.grid_w), (mixed // sub.grid_w).remainder(sub.grid_h)), dim=-1
        )
        occupancy = sub.occupancy()
        env = torch.arange(sub.E, device=self.device)[:, None]
        free = occupancy[env, candidates[..., 1], candidates[..., 0]] < 0
        # `_append_many` scans the candidate axis in source order and uses the
        # same duplicate/occupied rejection semantics as repeated `_append_food`.
        previous_cells = sub.food_cells
        sub._append_many(candidates, free, False)
        # A naturally depleted row can retain inactive cell bytes. Sequential
        # append overwrites accepted slots only; preserve that same padding.
        sub.food_cells = torch.where(sub.food_present[..., None], sub.food_cells, previous_cells)
        self._copy_food_rows(rows, sub)

    def _maintain_food_impl(self) -> None:
        """Keep frozen maintenance for live rows; batch only blank initial rows."""
        trials = int(self.cfg.max_food)
        if self.device.type == "meta":
            # The certified shape oracle remains the source for meta execution.
            return super()._maintain_food_impl()
        ambient = (self.food_present & ~self.corpse_food).sum(dim=1)
        active = ambient < trials
        # Food count zero identifies a true reset row.  Ambient zero alone can
        # still carry v2 corpse food and must retain the frozen maintenance path.
        empty = active & self.food_count.eq(0)
        if not bool(empty.any().item()):
            return super()._maintain_food_impl()

        empty_rows = empty.nonzero(as_tuple=False).flatten()
        self._append_initial_food(empty_rows, trials)

        # Nonempty rows need the original early-stop behavior because fewer
        # than `trials` candidates can be necessary.  Run that code on the
        # compact complement and copy just its food tensors back.
        nonempty_rows = (active & ~empty).nonzero(as_tuple=False).flatten()
        if nonempty_rows.numel():
            sub = self._row_subset(nonempty_rows)
            _CertifiedDeviceSim._maintain_food_impl(sub)
            self._copy_food_rows(nonempty_rows, sub)

        # Both frozen paths advance every world by this fixed counter budget,
        # including worlds already at capacity and empty rows whose append was
        # batched above.
        self.rng_counter += trials


@torch.no_grad()
def food_mechanics_parity_probe(
    cfg: object,
    *,
    device: str | torch.device = "cpu",
    ticks: int = 4,
    rng_keys: torch.Tensor | None = None,
) -> dict[str, int | str]:
    """Exercise reset-food placement and subsequent mechanics against frozen code.

    This is intentionally a small same-device development proof, suitable for
    the A100 receipt before a timing cell.  It is not a promotion/serving
    parity harness and does not alter either simulator instance.
    """
    if ticks < 1:
        raise ValueError("ticks must be positive")
    requested = torch.device(device)
    backend = "indexed" if requested.type == "cuda" else "reference"
    frozen = _CertifiedDeviceSim(
        cfg, device=device, train_mode=False, allow_respawn=True, move_backend=backend
    )
    candidate = ResidentDeviceSim(
        cfg, device=device, train_mode=False, allow_respawn=True, move_backend=backend
    )
    if rng_keys is not None:
        frozen.rng_key.copy_(rng_keys.to(frozen.device))
        candidate.rng_key.copy_(rng_keys.to(candidate.device))
    frozen.rng_counter.fill_(32)
    candidate.rng_counter.fill_(32)
    actions = torch.arange(frozen.S, device=frozen.device)[None].remainder(3).expand(frozen.E, -1)
    for tick in range(ticks):
        frozen.step(actions)
        candidate.step(actions)
        for name, value in frozen.__dict__.items():
            if isinstance(value, torch.Tensor):
                other = getattr(candidate, name)
                if (
                    value.dtype != other.dtype
                    or value.shape != other.shape
                    or not torch.equal(value, other)
                ):
                    raise AssertionError(f"A-3 food/mechanics mismatch at tick={tick} field={name}")
    return {"status": "PASS", "ticks": ticks, "worlds": frozen.E, "device": frozen.device.type}
