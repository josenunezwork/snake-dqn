"""Count existing fit masks; no actions, labels, margins or inference are recomputed."""

from __future__ import annotations

from typing import Any, Mapping

import torch

COUNT_MASKS = {
    "N_natural": "natural",
    "N_changed": "changed",
    "N_changed_both_exact": "changed_both_exact",
    "N_changed_B8_exact_survives_selected_exact": "changed_b8_exact_survives_selected_exact",
    "C_confirmed": "harmful_exact",
    "U_selected_only": "unknown_selected_only",
    "U_B8_only": "unknown_b8_only",
    "U_both": "unknown_both",
    "U_unknown": "harmful_unknown",
    "H": "harmful",
}
BOUNDS = ((0.0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, None))


def _counts(masks: Mapping[str, Any], subset: torch.Tensor) -> dict[str, int]:
    return {
        name: int((masks[mask] & masks["natural"] & subset).sum().item())
        for name, mask in COUNT_MASKS.items()
    }


def _bins(masks: Mapping[str, Any], subset: torch.Tensor) -> list[dict[str, Any]]:
    return [
        {"lower": lower, "upper": upper, "counts": _counts(masks, subset & in_bin)}
        for (lower, upper), in_bin in zip(BOUNDS, masks["margin_bins"], strict=True)
    ]


class AggregateObserver:
    """Consume masks once, retain only full-table/parent aggregate counters."""

    def __init__(self, parents: list[str]) -> None:
        if parents != sorted(set(parents)) or not parents:
            raise ValueError("expected sorted unique census parents")
        self.parents = parents
        self.result: dict[str, Any] | None = None
        self.total_rows = 0

    def __call__(self, masks: Mapping[str, Any], parents: tuple[str, ...]) -> None:
        if self.result is not None:
            raise ValueError("one observer invocation per candidate")
        natural = masks["natural"]
        if (
            natural.ndim != 1
            or natural.dtype != torch.bool
            or natural.device.type != "cpu"
            or len(parents) != len(natural)
            or sorted(set(parents)) != self.parents
            or any(
                masks[name].shape != natural.shape or masks[name].dtype != torch.bool
                for name in COUNT_MASKS.values()
            )
        ):
            raise ValueError("observer masks/parent census mismatch")
        self.total_rows = len(parents)
        all_rows = torch.ones_like(natural)
        self.result = {
            "totals": _counts(masks, all_rows),
            "margin_bins": _bins(masks, all_rows),
            "parents": [
                {
                    "parent_world": parent,
                    "totals": _counts(masks, selected),
                    "margin_bins": _bins(masks, selected),
                }
                for parent in self.parents
                for selected in [torch.tensor([p == parent for p in parents], dtype=torch.bool)]
            ],
        }


def valid_counts(counts: dict[str, Any]) -> bool:
    """Check arithmetic only, without defining or changing a scientific mask."""
    return (
        set(counts) == set(COUNT_MASKS)
        and all(type(value) is int and value >= 0 for value in counts.values())
        and counts["C_confirmed"] + counts["U_unknown"] == counts["H"]
        and counts["U_selected_only"] + counts["U_B8_only"] + counts["U_both"]
        == counts["U_unknown"]
        and counts["H"] <= counts["N_changed"] <= counts["N_natural"]
        and counts["N_changed_B8_exact_survives_selected_exact"]
        <= counts["N_changed_both_exact"]
        <= counts["N_changed"]
        and counts["C_confirmed"] <= counts["N_changed_B8_exact_survives_selected_exact"]
    )


def valid_bins(group: dict[str, Any]) -> bool:
    bins = group["margin_bins"]
    return (
        len(bins) == 4
        and all(
            set(entry) == {"lower", "upper", "counts"}
            and (entry["lower"], entry["upper"]) == bounds
            and valid_counts(entry["counts"])
            for entry, bounds in zip(bins, BOUNDS, strict=True)
        )
        and valid_counts(group["totals"])
        and all(
            sum(entry["counts"][name] for entry in bins) == group["totals"][name]
            for name in COUNT_MASKS
        )
    )


def valid_dose(dose: dict[str, Any], parents: list[str], natural_rows: int = 560) -> bool:
    """Validate exact aggregate schema and every parent/bin reconciliation."""
    return (
        set(dose) == {"dose", "checkpoint_sha256", "totals", "margin_bins", "parents"}
        and valid_bins(dose)
        and dose["totals"]["N_natural"] == natural_rows
        and [parent["parent_world"] for parent in dose["parents"]] == parents
        and all(
            set(parent) == {"parent_world", "totals", "margin_bins"} and valid_bins(parent)
            for parent in dose["parents"]
        )
        and all(
            sum(parent["totals"][name] for parent in dose["parents"]) == dose["totals"][name]
            for name in COUNT_MASKS
        )
        and all(
            sum(parent["margin_bins"][i]["counts"][name] for parent in dose["parents"])
            == dose["margin_bins"][i]["counts"][name]
            for name in COUNT_MASKS
            for i in range(4)
        )
    )


def dose_shape_valid(dose: dict[str, Any], parents: list[str]) -> bool:
    """Reject nonaggregate fields even when an arithmetic disagreement is INVALID."""

    def group_shape(group: dict[str, Any]) -> bool:
        def counts_shape(counts: Any) -> bool:
            return (
                isinstance(counts, dict)
                and set(counts) == set(COUNT_MASKS)
                and all(type(v) is int and v >= 0 for v in counts.values())
            )

        return (
            counts_shape(group.get("totals"))
            and isinstance(group.get("margin_bins"), list)
            and len(group["margin_bins"]) == 4
            and all(
                isinstance(b, dict)
                and set(b) == {"lower", "upper", "counts"}
                and (b["lower"], b["upper"]) == bounds
                and counts_shape(b["counts"])
                for b, bounds in zip(group["margin_bins"], BOUNDS, strict=True)
            )
        )

    return (
        isinstance(dose, dict)
        and set(dose) == {"dose", "checkpoint_sha256", "totals", "margin_bins", "parents"}
        and group_shape(dose)
        and isinstance(dose.get("parents"), list)
        and [p.get("parent_world") for p in dose["parents"]] == parents
        and all(
            set(p) == {"parent_world", "totals", "margin_bins"} and group_shape(p)
            for p in dose["parents"]
        )
    )
