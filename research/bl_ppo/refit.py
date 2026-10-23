"""Bounded complete-life, world-disjoint critic-refit qualification evidence."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

from research.bl_ppo.learner import PPOLearner, fit_time_length_baseline

REGISTERED_MIXES = ("scripted", "frozen", "mixed")


@dataclass
class _Sample:
    priority: int
    feature: list[float]
    length: float
    horizon: float
    prefix: float
    source: str = "natural"
    world: int = 0
    life: int = 0
    mix: str = "unknown"


@dataclass
class _Life:
    ordinal: int
    total: float = 0.0
    rows: int = 0
    samples: list[_Sample] = field(default_factory=list)
    pinned_late: _Sample | None = None


class CompleteLifeRefitBank:
    """Keep a fixed deterministic sample per natural life and completed split.

    It never stores all rollout rows: open lives keep ``samples_per_life`` rows,
    and each split keeps at most ``reservoir_size`` completed rows.  Suffix
    returns are exact gamma-one returns, assigned only at death/H5000.
    """

    def __init__(
        self,
        *,
        holdout_modulus: int = 5,
        holdout_remainder: int = 0,
        samples_per_life: int = 8,
        reservoir_size: int = 10_000,
    ) -> None:
        if holdout_modulus < 2 or not 0 <= holdout_remainder < holdout_modulus:
            raise ValueError("invalid deterministic world split")
        if samples_per_life < 1 or reservoir_size < 1:
            raise ValueError("refit reservoirs must be positive")
        self.holdout_modulus, self.holdout_remainder = holdout_modulus, holdout_remainder
        self.samples_per_life, self.reservoir_size = samples_per_life, reservoir_size
        self._open: dict[int, _Life] = {}
        self._ordinals: dict[int, int] = {}
        self._rows_complete = {"fit": 0, "holdout": 0}
        self._reservoir: dict[str, list[tuple[int, _Sample, float]]] = {"fit": [], "holdout": []}
        self._sources = {"fit": {}, "holdout": {}}

    def _split(self, world: int) -> str:
        return "holdout" if world % self.holdout_modulus == self.holdout_remainder else "fit"

    @staticmethod
    def _priority(world: int, life: int, row: int) -> int:
        return int.from_bytes(
            hashlib.blake2b(f"bl-e8/{world}/{life}/{row}".encode(), digest_size=8).digest(), "big"
        )

    def _offer(self, split: str, item: tuple[int, _Sample, float]) -> None:
        rows = self._reservoir[split]
        if len(rows) < self.reservoir_size:
            rows.append(item)
            return
        worst = max(range(len(rows)), key=lambda index: rows[index][0])
        if item[0] < rows[worst][0]:
            rows[worst] = item

    def add_batch(
        self,
        world_ids: torch.Tensor,
        critic_features: torch.Tensor,
        lengths: torch.Tensor,
        remaining_horizon: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        *,
        mixes: Sequence[str] | None = None,
        split_override: str | None = None,
        pin_first_late: bool = False,
        source: str = "natural",
    ) -> int:
        vectors = (world_ids, lengths, remaining_horizon, rewards, terminals)
        if any(x.ndim != 1 for x in vectors) or len({x.numel() for x in vectors}) != 1:
            raise ValueError("refit evidence fields must be equal-length vectors")
        if critic_features.ndim != 2 or critic_features.shape != (world_ids.numel(), 257):
            raise ValueError("critic_features must have shape [N, 257]")
        if (
            world_ids.dtype.is_floating_point
            or world_ids.dtype == torch.bool
            or terminals.dtype.is_floating_point
        ):
            raise ValueError("world_ids must be integral and terminals boolean/integral")
        if not all(
            torch.isfinite(x).all() for x in (critic_features, lengths, remaining_horizon, rewards)
        ):
            raise ValueError("refit evidence must be finite")
        if not ((remaining_horizon >= 0).all() and (remaining_horizon <= 1).all()):
            raise ValueError("remaining_horizon must lie in [0, 1]")
        if split_override not in {None, "fit", "holdout"}:
            raise ValueError("split_override must be fit, holdout, or None")
        if not source:
            raise ValueError("refit evidence source must be non-empty")
        if mixes is None:
            mixes = ("unknown",) * world_ids.numel()
        if len(mixes) != world_ids.numel() or any(not str(mix) for mix in mixes):
            raise ValueError("mixes must contain one non-empty string per refit row")
        completed = 0
        for i in range(world_ids.numel()):
            world = int(world_ids[i])
            life = self._open.setdefault(world, _Life(self._ordinals.get(world, 0)))
            sample = _Sample(
                self._priority(world, life.ordinal, life.rows),
                critic_features[i].detach().cpu().float().tolist(),
                float(lengths[i]),
                float(remaining_horizon[i]),
                life.total,
                source,
                world,
                life.ordinal,
                str(mixes[i]),
            )
            life.total += float(rewards[i])
            life.rows += 1
            if len(life.samples) < self.samples_per_life:
                life.samples.append(sample)
            else:
                worst = max(range(len(life.samples)), key=lambda j: life.samples[j].priority)
                if sample.priority < life.samples[worst].priority:
                    life.samples[worst] = sample
            if pin_first_late and life.pinned_late is None and sample.length >= 800.0:
                # The fixed support bank has an explicit initial long-body row.
                # Keep it outside random reservoir replacement so qualification
                # cannot lose all L>=800 support through sampling.
                life.pinned_late = sample
            if bool(terminals[i]):
                split = split_override or self._split(world)
                self._rows_complete[split] += life.rows
                for kept in life.samples:
                    if life.pinned_late is not None and kept.priority == life.pinned_late.priority:
                        continue
                    self._offer(split, (kept.priority, kept, life.total - kept.prefix))
                    completed += 1
                if life.pinned_late is not None:
                    # Negative priority dominates ordinary reservoir entries;
                    # the fixed bank is bounded far below the 10k capacity.
                    pinned = life.pinned_late
                    self._offer(split, (-1 - pinned.priority, pinned, life.total - pinned.prefix))
                    completed += 1
                by_source = self._sources[split]
                by_source[source] = int(by_source.get(source, 0)) + life.rows
                self._ordinals[world] = life.ordinal + 1
                del self._open[world]
        return completed

    def counts(self) -> dict[str, int]:
        return dict(self._rows_complete)

    def support_counts(self) -> dict[str, int]:
        return {key: len(value) for key, value in self._reservoir.items()}

    def source_counts(self) -> dict[str, dict[str, int]]:
        """Completed-row provenance retained with the immutable qualification state."""
        return {split: dict(rows) for split, rows in self._sources.items()}

    def late_support_by_source(self, split: str = "holdout") -> dict[str, int]:
        """Retained L>=800 reservoir rows, grouped by immutable provenance."""
        if split not in self._reservoir:
            raise ValueError("unknown split")
        result: dict[str, int] = {}
        for _priority, sample, _target in self._reservoir[split]:
            if sample.length >= 800.0:
                result[sample.source] = result.get(sample.source, 0) + 1
        return result

    @staticmethod
    def _length_bucket(length: float) -> str:
        if length < 800.0:
            return "<800"
        if length < 1000.0:
            return "800-999"
        if length < 1200.0:
            return "1000-1199"
        return "1200+"

    @staticmethod
    def _horizon_bucket(horizon: float) -> str:
        # ``horizon`` is a normalized remaining horizon under P1's fixed H5000.
        return "<=1500" if horizon <= 0.3000001 else ">1500"

    def _late_rows(self, split: str, *, natural_only: bool) -> list[tuple[int, _Sample, float]]:
        if split not in self._reservoir:
            raise ValueError("unknown split")
        return [
            row
            for row in self._reservoir[split]
            if row[1].length >= 800.0 and (not natural_only or row[1].source == "natural")
        ]

    def natural_support(self, split: str) -> dict[str, object]:
        """Return natural completed late-life support for the prospective E8 rule.

        This is deliberately independent of constructed fixture rows.  A life is
        counted once even when its deterministic reservoir retains several late
        rows, and the per-mix rule is evaluated from those distinct lives.
        """
        rows = self._late_rows(split, natural_only=True)
        lives_by_mix: dict[str, set[tuple[int, int]]] = {}
        for _priority, sample, _target in rows:
            lives_by_mix.setdefault(sample.mix, set()).add((sample.world, sample.life))
        lives = {(sample.world, sample.life) for _priority, sample, _target in rows}
        minimum_lives, minimum_rows = (160, 640) if split == "fit" else (40, 160)
        minimum_per_mix = 0 if split == "fit" else 8
        per_mix = {mix: len(lives_by_mix.get(mix, set())) for mix in REGISTERED_MIXES}
        return {
            "split": split,
            "completed_late_lives": len(lives),
            "late_rows": len(rows),
            "lives_by_mix": per_mix,
            "minimum_completed_late_lives": minimum_lives,
            "minimum_late_rows": minimum_rows,
            "minimum_lives_per_mix": minimum_per_mix,
            "meets_minimum": len(lives) >= minimum_lives
            and len(rows) >= minimum_rows
            and (minimum_per_mix == 0 or min(per_mix.values()) >= minimum_per_mix),
        }

    def support_table(self, split: str) -> list[dict[str, object]]:
        """Group retained rows by provenance, mix, length and horizon.

        The output is JSON-ready and records distinct completed world/life pairs
        so status receipts can diagnose support without pooling fixture rows.
        """
        grouped: dict[tuple[str, str, str, str], dict[str, object]] = {}
        for _priority, sample, _target in self._reservoir[split]:
            key = (
                sample.source,
                sample.mix,
                self._length_bucket(sample.length),
                self._horizon_bucket(sample.horizon),
            )
            item = grouped.setdefault(
                key,
                {
                    "provenance": sample.source,
                    "mix": sample.mix,
                    "length_bucket": key[2],
                    "remaining_horizon_bucket": key[3],
                    "rows": 0,
                    "_lives": set(),
                },
            )
            item["rows"] = int(item["rows"]) + 1
            item["_lives"].add((sample.world, sample.life))  # type: ignore[index]
        result: list[dict[str, object]] = []
        for key in sorted(grouped):
            item = grouped[key]
            lives = item.pop("_lives")
            item["distinct_completed_lives"] = len(lives)  # type: ignore[arg-type]
            result.append(item)
        return result

    def diagnostic_rows(self, split: str = "holdout") -> dict[str, object]:
        """Return all natural completed rows for the registered offline diagnostic.

        Fixture rows are quarantined by construction: this API emits only
        natural rows and includes each row's immutable provenance and life key.
        """
        rows = [row for row in self._reservoir[split] if row[1].source == "natural"]
        features = (
            torch.tensor([sample.feature for _, sample, _ in rows], dtype=torch.float32)
            if rows
            else torch.empty((0, 257), dtype=torch.float32)
        )
        return {
            "features257": features,
            "returns": torch.tensor([target for _, _, target in rows], dtype=torch.float32),
            "lengths": torch.tensor([sample.length for _, sample, _ in rows], dtype=torch.float32),
            "remaining_horizon": torch.tensor(
                [sample.horizon for _, sample, _ in rows], dtype=torch.float32
            ),
            "mixes": [sample.mix for _, sample, _ in rows],
            "provenance": [sample.source for _, sample, _ in rows],
            "world_ids": torch.tensor([sample.world for _, sample, _ in rows], dtype=torch.int64),
            "life_ids": torch.tensor([sample.life for _, sample, _ in rows], dtype=torch.int64),
        }

    def _columns(
        self, split: str, *, natural_only: bool = True, late_only: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = (
            self._late_rows(split, natural_only=natural_only)
            if late_only
            else [
                row
                for row in self._reservoir[split]
                if not natural_only or row[1].source == "natural"
            ]
        )
        if not rows:
            raise ValueError("critic qualification requires completed fit and held-out lives")
        return (
            torch.tensor([x[1].feature for x in rows], dtype=torch.float32),
            torch.tensor([x[1].length for x in rows]),
            torch.tensor([x[1].horizon for x in rows]),
            torch.tensor([x[2] for x in rows]),
        )

    def qualification(
        self, learner: PPOLearner, *, require_support_minimum: bool = False
    ) -> dict[str, object]:
        """Score natural held-out late rows, optionally applying P1's support rule.

        The floor is prospective and P1-only.  Keeping it opt-in preserves the
        earlier E2 registration while allowing P1 runtime to make support part
        of actor admission.
        """
        fit_support = self.natural_support("fit")
        holdout_support = self.natural_support("holdout")
        sufficient = bool(fit_support["meets_minimum"] and holdout_support["meets_minimum"])
        fit_rows = [row for row in self._reservoir["fit"] if row[1].source == "natural"]
        holdout_late = self._late_rows("holdout", natural_only=True)
        if not fit_rows or not holdout_late:
            # Status output stays numeric even when a collection failure left no
            # evaluable rows. ``metrics_available`` makes zero placeholders
            # unambiguous and prevents their use as a gate result.
            result: dict[str, object] = {
                "late_body_rows": float(len(holdout_late)),
                "explained_variance": 0.0,
                "mse": 0.0,
                "baseline_mse": 0.0,
                "bias": 0.0,
                "passed": False,
                "metrics_available": False,
            }
            inputs: dict[str, torch.Tensor] | None = None
        else:
            # The registered comparator stays fit on every natural completed
            # refit row (1, h, L); only the held-out estimand is late-body.
            _, fl, fh, fr = self._columns("fit", natural_only=True, late_only=False)
            features, hl, hh, hr = self._columns("holdout", natural_only=True, late_only=True)
            coef = fit_time_length_baseline(fl, fh, fr)
            device = next(learner.policy.critic.parameters()).device
            with torch.no_grad():
                predictions = learner.policy.value_from_critic_features(features.to(device)).cpu()
            inputs = dict(
                predictions=predictions,
                returns=hr,
                lengths=hl,
                remaining_horizon=hh,
                baseline_coefficients=coef,
            )
            result = learner.critic_quality(**inputs)
            result["metrics_available"] = True
            if not all(
                math.isfinite(float(value))
                for value in result.values()
                if not isinstance(value, bool)
            ):
                raise FloatingPointError("non-finite E8 qualification metrics")
        accuracy_passed = bool(result["passed"])
        admitted = accuracy_passed and (sufficient or not require_support_minimum)
        if admitted and inputs is not None:
            learner.qualify_critic(**inputs)
        result.update(
            count=result["late_body_rows"],
            fit_support=float(fit_support["late_rows"]),
            holdout_support=float(holdout_support["late_rows"]),
            support_sufficient=float(sufficient),
            accuracy_passed=accuracy_passed,
            passed=admitted,
            status=(
                "PASS"
                if admitted
                else (
                    "INSUFFICIENT_LATE_BODY_SUPPORT"
                    if require_support_minimum and not sufficient
                    else "CRITIC_QUALITY_REFUSAL"
                )
            ),
        )
        return result

    def score_fixture_holdout(
        self, learner: PPOLearner, fixture_bank: "CompleteLifeRefitBank"
    ) -> dict[str, object]:
        """Score labelled fixture rows with the natural-fit baseline, never admitting.

        This is descriptive evidence only.  It neither updates PopArt nor calls
        ``qualify_critic`` and it cannot contribute to natural support.
        """
        fit_rows = [row for row in self._reservoir["fit"] if row[1].source == "natural"]
        fixture_rows = fixture_bank._late_rows("holdout", natural_only=False)
        if not fit_rows or not fixture_rows:
            return {
                "fixture_late_body_rows": float(len(fixture_rows)),
                "explained_variance": 0.0,
                "mse": 0.0,
                "baseline_mse": 0.0,
                "bias": 0.0,
                "metrics_available": False,
            }
        _, fl, fh, fr = self._columns("fit", natural_only=True)
        features = torch.tensor([row[1].feature for row in fixture_rows], dtype=torch.float32)
        lengths = torch.tensor([row[1].length for row in fixture_rows])
        horizon = torch.tensor([row[1].horizon for row in fixture_rows])
        returns = torch.tensor([row[2] for row in fixture_rows])
        coef = fit_time_length_baseline(fl, fh, fr)
        device = next(learner.policy.critic.parameters()).device
        with torch.no_grad():
            predictions = learner.policy.value_from_critic_features(features.to(device)).cpu()
        result: dict[str, object] = learner.critic_quality(
            predictions, returns, lengths, horizon, baseline_coefficients=coef
        )
        if not all(
            math.isfinite(float(value)) for value in result.values() if not isinstance(value, bool)
        ):
            raise FloatingPointError("non-finite fixture late-body diagnostic metrics")
        result["fixture_late_body_rows"] = result.pop("late_body_rows")
        result["metrics_available"] = True
        result["diagnostic_only"] = True
        result["actor_admission"] = False
        return result

    def fit_complete_targets(self, learner: PPOLearner) -> float:
        """One bounded exact-return head-refit pass over fit-world reservoir rows."""
        # Fixed holdout fixtures are never allowed into PopArt.  The separately
        # registered, fit-only supplemental roots remain permitted critic data.
        rows = [
            row
            for row in self._reservoir["fit"]
            if row[1].source in {"natural", "refit_long_body_support_supplement"}
        ]
        if not rows:
            raise ValueError("critic refit requires completed natural or supplemental fit rows")
        features = torch.tensor([x[1].feature for x in rows], dtype=torch.float32)
        returns = torch.tensor([x[2] for x in rows])
        return learner.fit_completed_critic(features, returns)

    def solve_e8_critic_convergence_amendment_03(self, learner: PPOLearner) -> dict[str, object]:
        """Run the prospective exact-return head solve on natural fit rows only."""
        rows = [row for row in self._reservoir["fit"] if row[1].source == "natural"]
        if not rows:
            raise ValueError("Amendment-03 critic solve requires completed natural fit rows")
        if len(rows) > self.reservoir_size or len(rows) > 10_000:
            raise ValueError("Amendment-03 critic solve exceeds the bounded fit reservoir")
        # Do not route targets through ``_columns``: its historic default dtype
        # is float32.  Preserve raw stored suffix returns until the solver's
        # explicit NumPy float64 conversion.
        features = torch.tensor([row[1].feature for row in rows], dtype=torch.float64)
        returns = torch.tensor([row[2] for row in rows], dtype=torch.float64)
        return learner.install_exact_return_critic_solution(features, returns)

    def solve_e8_critic_convergence_amendment_04(self, learner: PPOLearner) -> dict[str, object]:
        """Run Amendment-04's zero-sum solve on natural fit rows only."""
        rows = [row for row in self._reservoir["fit"] if row[1].source == "natural"]
        if not rows:
            raise ValueError("Amendment-04 critic solve requires completed natural fit rows")
        if len(rows) > self.reservoir_size or len(rows) > 10_000:
            raise ValueError("Amendment-04 critic solve exceeds the bounded fit reservoir")
        features = torch.tensor([row[1].feature for row in rows], dtype=torch.float64)
        returns = torch.tensor([row[2] for row in rows], dtype=torch.float64)
        return learner.install_exact_return_critic_solution(features, returns, zero_sum_basis=True)

    qualify = qualification
