"""Fail-closed analysis and registered descriptive readouts for T11 LOCAL30.

The screen decision is deliberately separate from diagnostics.  In particular,
one halted LOCAL30 lane is sufficient for ``KILL`` and makes the five-seed
gameplay comparisons unavailable; four completed lanes must never be silently
presented as a replacement screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.t2_screen_20261015.analyze_t2 import _mean, _number_map, _se, _survival_maps

ARM = "LOCAL30"
SEEDS = tuple(range(10, 15))
MIXES = ("scripted", "frozen", "mixed")
INITIAL_ROUND, EXTENSION_ROUND, FRAMES = 74, 75, 5000
DRIFT_REFERENCE_PERCENT = 0.99
DEFAULT_ARTIFACT_ROOT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t11-mac-20261017"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _group(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    result: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        try:
            key = (str(row["name"]), int(row["round_index"]), str(row["mix"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("every T11 row needs name, round_index, and mix") from exc
        if key in result:
            raise ValueError("duplicate T11 probe row: %r" % (key,))
        result[key] = row
    return result


def _metric(row: Mapping[str, Any], metric: str) -> Mapping[str, float]:
    if metric == "episode_survival":
        return _survival_maps(row)[0]
    if metric == "alive_survival":
        return _survival_maps(row)[1]
    return _number_map(row, metric)


def _paired(
    grouped: Mapping[tuple[str, int, str], Mapping[str, Any]], mix: str, metric: str
) -> dict[str, Any]:
    """Return seed-paired LOCAL30-minus-CTRL effects without imputing absent seeds."""
    per_seed, pooled, observed = [], [], []
    rounds = (INITIAL_ROUND, EXTENSION_ROUND) if mix == "scripted" else (INITIAL_ROUND,)
    for seed in SEEDS:
        left_all: dict[str, float] = {}
        right_all: dict[str, float] = {}
        for round_index in rounds:
            left = grouped.get((f"{ARM}_s{seed}", round_index, mix))
            right = grouped.get((f"CTRL_s{seed}", round_index, mix))
            if left is None or right is None:
                continue
            a, b = _metric(left, metric), _metric(right, metric)
            if set(a) != set(b) or set(left_all) & set(a):
                raise ValueError("T11 paired worlds must match and not repeat")
            left_all.update(a)
            right_all.update(b)
        if left_all:
            delta = [left_all[key] - right_all[key] for key in sorted(left_all)]
            per_seed.append(_mean(delta))
            pooled.extend(delta)
            observed.append(seed)
    return {
        "mean": _mean(per_seed),
        "exact_numerator": (
            sum(int(value) for value in pooled) if metric == "episode_survival" else None
        ),
        "exact_denominator": len(pooled) if metric == "episode_survival" else None,
        "seed_se": _se(per_seed),
        "pooled_world_se": _se(pooled),
        "seeds": observed,
        "seed_n": len(per_seed),
        "world_n": len(pooled),
        "complete": observed == list(SEEDS),
    }


def _shared_lengths(grouped: Mapping[tuple[str, int, str], Mapping[str, Any]]) -> dict[str, Any]:
    """Condition length differences on both matched heroes being alive at each frame."""
    output: dict[str, Any] = {}
    for frame in range(1000, 5001, 1000):
        frame_rows = []
        for mix in MIXES:
            per_seed, pooled = [], []
            for seed in SEEDS:
                candidate = grouped.get((f"{ARM}_s{seed}", INITIAL_ROUND, mix))
                control = grouped.get((f"CTRL_s{seed}", INITIAL_ROUND, mix))
                if candidate is None or control is None:
                    continue
                lengths_a = _frame_lengths(candidate, frame)
                lengths_b = _frame_lengths(control, frame)
                alive_a, alive_b = _alive_at(candidate, frame), _alive_at(control, frame)
                if (
                    set(lengths_a) != set(lengths_b)
                    or set(alive_a) != set(lengths_a)
                    or set(alive_b) != set(lengths_a)
                ):
                    raise ValueError("T11 length/alive maps must align with paired worlds")
                shared = [key for key in sorted(lengths_a) if alive_a[key] and alive_b[key]]
                values = [lengths_a[key] - lengths_b[key] for key in shared]
                per_seed.append(
                    {"seed": seed, "shared_survivors": len(shared), "mean_delta": _mean(values)}
                )
                pooled.extend(values)
            finite = [item["mean_delta"] for item in per_seed if math.isfinite(item["mean_delta"])]
            frame_rows.append(
                {
                    "mix": mix,
                    "per_seed": per_seed,
                    "shared_survivors": len(pooled),
                    "pooled_mean_delta": _mean(pooled),
                    "descriptive_seed_mean_delta": _mean(finite),
                    "descriptive_seed_n": len(finite),
                }
            )
        output[str(frame)] = {item.pop("mix"): item for item in frame_rows}
    return {"frames": output, "convention": "both paired heroes alive at the named post-frame"}


def _frame_lengths(row: Mapping[str, Any], frame: int) -> Mapping[str, float]:
    values = row.get("length_at_frames", {})
    if not isinstance(values, Mapping):
        return {}
    return _number_map({"values": values.get(str(frame), values.get(frame, {}))}, "values")


def _alive_at(row: Mapping[str, Any], frame: int) -> Mapping[str, bool]:
    alive = _number_map(row, "alive_frames")
    return {key: value >= frame for key, value in alive.items()}


def _decision(statuses: Mapping[str, int], gameplay: Mapping[str, Any], extended: bool) -> str:
    if any(code != 0 for code in statuses.values()):
        return "KILL"
    primary = gameplay["scripted"]["episode_survival"]
    if not primary["complete"]:
        return "INCOMPLETE"
    if primary["exact_numerator"] <= 0:
        return "KILL"
    if (
        primary["mean"] >= 0.06
        and all(gameplay[mix]["mass"]["mean"] >= -30.0 for mix in MIXES)
        and all(gameplay[mix]["episode_survival"]["mean"] >= -0.05 for mix in ("frozen", "mixed"))
    ):
        return "GO"
    return "INCONCLUSIVE" if extended else "EXTEND"


def _validate_complete_screen(
    grouped: Mapping[tuple[str, int, str], Mapping[str, Any]], statuses: Mapping[str, int]
) -> None:
    """Require the complete preregistered roster before a nonhalt decision."""
    names = {f"{ARM}_s{seed}" for seed in SEEDS}
    if set(statuses) != names or any(code != 0 for code in statuses.values()):
        raise ValueError("T11 nonhalt analysis requires five successful terminal receipts")
    from research.redesign_m2_20261008 import dev_probe

    registered = (
        "baseline",
        *(f"{ARM}_s{seed}" for seed in SEEDS),
        *(f"CTRL_s{seed}" for seed in SEEDS),
    )
    required = {(name, INITIAL_ROUND, mix) for name in registered for mix in MIXES}
    if not required.issubset(grouped):
        raise ValueError("T11 initial round lacks registered arms, seeds, or mixes")
    extended = any(round_index == EXTENSION_ROUND for _, round_index, _ in grouped)
    if extended:
        extension = {
            (name, EXTENSION_ROUND, "scripted") for name in registered if name != "baseline"
        }
        if not extension.issubset(grouped):
            raise ValueError("T11 extension requires all ten registered scripted rows")
    for (name, round_index, mix), row in grouped.items():
        valid = (name in registered and round_index == INITIAL_ROUND and mix in MIXES) or (
            name in registered
            and name != "baseline"
            and round_index == EXTENSION_ROUND
            and mix == "scripted"
        )
        if not valid:
            raise ValueError("T11 rejects unregistered probe arms, rounds, or mixes")
        expected_worlds = 96 if mix == "scripted" else 32
        expected_seeds = set(map(str, dev_probe.probe_seeds(expected_worlds, round_index)))
        maps = [_metric(row, metric) for metric in ("episode_survival", "alive_survival", "mass")]
        if row.get("frames") != FRAMES or any(len(values) != expected_worlds for values in maps):
            raise ValueError("T11 probe row has wrong registered world count")
        if any(set(values) != expected_seeds for values in maps):
            raise ValueError("T11 probe metrics must share exactly one world roster")


def analyze(
    rows: Sequence[Mapping[str, Any]],
    statuses: Mapping[str, int],
    telemetry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply T11's rule. A halt wins over missing game probes and readouts."""
    expected = {f"{ARM}_s{seed}" for seed in SEEDS}
    if not set(statuses).issubset(expected) or any(
        type(value) is not int for value in statuses.values()
    ):
        raise ValueError("T11 statuses must be integer LOCAL30 exits")
    grouped = _group(rows)
    if not any(code != 0 for code in statuses.values()):
        _validate_complete_screen(grouped, statuses)
    extended = any(key[1] == EXTENSION_ROUND for key in grouped)
    gameplay = {
        mix: {
            metric: _paired(grouped, mix, metric)
            for metric in ("episode_survival", "alive_survival", "mass")
        }
        for mix in MIXES
    }
    if extended and not any(code != 0 for code in statuses.values()):
        initial = {key: row for key, row in grouped.items() if key[1] == INITIAL_ROUND}
        initial_gameplay = {
            mix: {
                metric: _paired(initial, mix, metric)
                for metric in ("episode_survival", "alive_survival", "mass")
            }
            for mix in MIXES
        }
        if _decision(statuses, initial_gameplay, False) != "EXTEND":
            raise ValueError("T11 extension requires initial EXTEND")
    halted = any(code != 0 for code in statuses.values())
    data: dict[str, Any] = {
        "decision": _decision(statuses, gameplay, extended),
        "decision_basis": (
            "halt/nonzero exit is preregistered KILL" if halted else "registered gameplay rule"
        ),
        "failed_runs": [
            {"name": name, "exit": code} for name, code in sorted(statuses.items()) if code != 0
        ],
        "unstarted_runs": [f"{ARM}_s{seed}" for seed in SEEDS if f"{ARM}_s{seed}" not in statuses],
        "gameplay": gameplay if not halted else None,
        "gameplay_unavailable_reason": (
            "five-seed roster invalid after preregistered halt; no evaluation is substituted"
            if halted
            else None
        ),
        "four_seed_descriptive_gameplay": gameplay if halted and grouped else None,
        "shared_survivor_lengths": _shared_lengths(grouped) if grouped else None,
        "consequential_action_change": None,
        "consequential_action_change_reason": (
            "not measurable: no eligible fresh gameplay after halt"
        ),
        "death_within_30": None,
        "death_within_30_reason": (
            "not measurable: no validated branch replay trajectories were generated"
        ),
    }
    if telemetry is not None:
        data["telemetry"] = telemetry
    return _json_safe({"schema": "t11-local30-analysis/v1", "arms": {ARM: data}})


def _records(path: Path) -> list[Mapping[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_training_telemetry(roots: Mapping[str, Path]) -> dict[str, Any]:
    """Read all lanes, keeping the halted lane and its final-window absence explicit."""
    per_seed: dict[str, Any] = {}
    curves: list[dict[str, Any]] = []
    diagnostics = (
        "td",
        "anchor",
        "bank_margin_loss",
        "qmax",
        "tight_frac",
        "deaths_per_1k",
        "death_len_mean",
    )
    for seed in SEEDS:
        name, root = f"{ARM}_s{seed}", Path(roots[f"{ARM}_s{seed}"])
        summary_path, log_path = root / "summary.json", root / "log.jsonl"
        if not summary_path.is_file() or not log_path.is_file():
            per_seed[str(seed)] = {"status": "missing", "final_window": None}
            continue
        summary, records = json.loads(summary_path.read_text()), _records(log_path)
        halt = summary.get("halt")
        final = records[-1] if records else {}
        checkpoint = root / "final.pth"
        completed = (
            not halt
            and summary.get("actor_exitcodes") == [0, 0]
            and float(summary.get("transitions", 0)) >= 8_000_000
            and checkpoint.is_file()
        )
        final_window = final if completed else None
        per_seed[str(seed)] = {
            "status": "complete" if completed else "halted",
            "halt": halt,
            "transitions": summary.get("transitions"),
            "actor_exitcodes": summary.get("actor_exitcodes"),
            "summary_sha256": _sha256(summary_path),
            "log_sha256": _sha256(log_path),
            "final_checkpoint_sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
            "aborted_checkpoint_sha256": (
                _sha256(root / "aborted.pth") if (root / "aborted.pth").is_file() else None
            ),
            "final_window": _point(final_window, diagnostics) if final_window else None,
        }
        seed_windows = []
        for record in records:
            if isinstance(record.get("transitions"), Real):
                point = {"seed": seed, **_point(record, diagnostics)}
                curves.append(point)
                seed_windows.append(point)
        per_seed[str(seed)]["per_100k_windows"] = seed_windows
    by_point: dict[int, list[Mapping[str, Any]]] = {}
    for item in curves:
        nominal = int(round(float(item["transitions"]) / 100000.0)) * 100000
        by_point.setdefault(nominal, []).append(item)
    curve_summary = []
    for nominal, values in sorted(by_point.items()):
        curve_summary.append(
            {
                "transitions": nominal,
                "n": len(values),
                "bank_agree_decisive_mean": _mean(_numbers(values, "bank_agree_decisive")),
                "bank_agree_counter_mean": _mean(_numbers(values, "bank_agree_counter")),
                "bank_agree_decisive_five_seed_mean": (
                    _mean(_numbers(values, "bank_agree_decisive")) if len(values) == 5 else None
                ),
                "bank_agree_counter_five_seed_mean": (
                    _mean(_numbers(values, "bank_agree_counter")) if len(values) == 5 else None
                ),
                "greedy_boost_rate_mean": _mean(_numbers(values, "greedy_boost_rate")),
                "greedy_boost_rate_lt_700_mean": _mean(
                    _numbers(values, "greedy_boost_rate_lt_700")
                ),
                "td_mean": _mean(_numbers(values, "td")),
                "anchor_mean": _mean(_numbers(values, "anchor")),
                "bank_margin_loss_mean": _mean(_numbers(values, "bank_margin_loss")),
                "qmax_mean": _mean(_numbers(values, "qmax")),
            }
        )
    completed = [value for value in per_seed.values() if value.get("status") == "complete"]
    artifact_root = next(iter(roots.values())).parent if roots else None
    failure_sources = {}
    if artifact_root:
        for name in ("lane0-status.jsonl", "lane1-status.jsonl", "KILL.json"):
            source = artifact_root / name
            failure_sources[name] = _sha256(source) if source.is_file() else None
    return _json_safe(
        {
            "schema": "t11-training-telemetry/v1",
            "per_seed": per_seed,
            "per_100k": curve_summary,
            "all_windows": curves,
            "failure_source_sha256": failure_sources,
            "final_8m_four_seed_descriptive_n": len(completed),
            "final_8m_family_agreement": {
                "decisive_mean": _mean(
                    _nested_numbers(completed, "final_window", "bank_agree_decisive")
                ),
                "counter_mean": _mean(
                    _nested_numbers(completed, "final_window", "bank_agree_counter")
                ),
                "interpretation": (
                    "descriptive four-completed-lane summary; halted seed has null final window"
                ),
            },
            "tripwires": {"greedy_boost_rate": 0.02, "greedy_boost_rate_lt_700": 0.01},
        }
    )


def _point(record: Mapping[str, Any], diagnostics: Sequence[str]) -> dict[str, Any]:
    keys = (
        "transitions",
        "bank_agree_decisive",
        "bank_agree_counter",
        "greedy_boost_rate",
        "greedy_boost_rate_lt_700",
        "greedy_boost_rate_by_length",
        "greedy_boost_length_decisions",
        *diagnostics,
    )
    return {key: record.get(key) for key in keys}


def _numbers(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), Real) and math.isfinite(float(row[key]))
    ]


def _nested_numbers(rows: Sequence[Mapping[str, Any]], parent: str, key: str) -> list[float]:
    return [
        float(row[parent][key])
        for row in rows
        if isinstance(row.get(parent), Mapping) and isinstance(row[parent].get(key), Real)
    ]


def markdown(result: Mapping[str, Any]) -> str:
    """Render an auditable result while making the pre-evaluation KILL unambiguous."""
    data = result["arms"][ARM]
    telemetry = data.get("telemetry", {})
    lines = [
        "# T11 LOCAL30 screen result (2026-10-17)",
        "",
        f"Decision: **{data['decision']}**.",
        "",
        data["decision_basis"] + ".",
    ]
    if data["failed_runs"]:
        lines += ["", "## Halt", ""]
        for failed in data["failed_runs"]:
            lines.append(f"- {failed['name']}: exit {failed['exit']}.")
    lines += [
        "",
        "## Registered gameplay",
        "",
        data["gameplay_unavailable_reason"] or "Available in analysis JSON.",
        "",
        "## Registered descriptive readouts",
        "",
    ]
    agreement = (
        telemetry.get("final_8m_family_agreement", {}) if isinstance(telemetry, Mapping) else {}
    )
    lines.append(
        "Final 8M family agreement (four completed lanes only): decisive %s; counter %s. %s"
        % (
            _show(agreement.get("decisive_mean")),
            _show(agreement.get("counter_mean")),
            agreement.get("interpretation", ""),
        )
    )
    lines += [
        "",
        "Consequential flips: " + data["consequential_action_change_reason"] + ".",
        "",
        "Death within 30 decisions: " + data["death_within_30_reason"] + ".",
    ]
    return "\n".join(lines) + "\n"


def telemetry_svg(telemetry: Mapping[str, Any], path: Path) -> None:
    """Write a dependency-free provenance chart for the two boost tripwires.

    The graphic intentionally plots means only.  The JSON remains the authority
    for individual lanes, including the halted lane that triggered the KILL.
    """
    points = list(telemetry.get("per_100k", []))
    width, height, pad = 960, 360, 60
    usable = [point for point in points if isinstance(point.get("transitions"), Real)]
    maximum = max((float(point["transitions"]) for point in usable), default=1.0)

    def x(value: float) -> float:
        return pad + (width - 2 * pad) * value / maximum

    def y(value: float) -> float:
        return height - pad - (height - 2 * pad) * min(max(value, 0.0), 0.025) / 0.025

    def line(key: str, color: str) -> str:
        pairs = [
            (x(float(row["transitions"])), y(float(row[key])))
            for row in usable
            if isinstance(row.get(key), Real)
        ]
        return '<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (
            color,
            " ".join("%.1f,%.1f" % pair for pair in pairs),
        )

    low = y(0.01)
    high = y(0.02)
    body = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="360" viewBox="0 0 960 360">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<text x="60" y="28" font-family="sans-serif" font-size="16">'
            "T11 greedy boost telemetry (per-100k mean)</text>"
        ),
        '<line x1="60" y1="300" x2="900" y2="300" stroke="black"/>',
        '<line x1="60" y1="60" x2="60" y2="300" stroke="black"/>',
        '<line x1="60" y1="%.1f" x2="900" y2="%.1f" stroke="#c33" stroke-dasharray="5 4"/>'
        % (low, low),
        '<line x1="60" y1="%.1f" x2="900" y2="%.1f" stroke="#c33" stroke-dasharray="5 4"/>'
        % (high, high),
        line("greedy_boost_rate_mean", "#1769aa"),
        line("greedy_boost_rate_lt_700_mean", "#e07a20"),
        (
            '<text x="700" y="50" font-family="sans-serif" font-size="12" '
            'fill="#1769aa">all lengths</text>'
        ),
        (
            '<text x="790" y="50" font-family="sans-serif" font-size="12" '
            'fill="#e07a20">&lt;700</text>'
        ),
        '<text x="5" y="%.1f" font-family="sans-serif" font-size="11">1%%</text>' % (low + 4),
        '<text x="5" y="%.1f" font-family="sans-serif" font-size="11">2%%</text>' % (high + 4),
        "</svg>",
    ]
    path.write_text("\n".join(body) + "\n")


def _show(value: Any) -> str:
    return (
        "null"
        if not isinstance(value, Real) or not math.isfinite(float(value))
        else f"{float(value):.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--out-json", type=Path, default=Path("research/t11_screen_20261017/results/analysis.json")
    )
    parser.add_argument(
        "--out-md", type=Path, default=Path("docs/research/t11_screen_result_2026-10-17.md")
    )
    parser.add_argument(
        "--out-svg",
        type=Path,
        default=Path("research/t11_screen_20261017/results/boost_curves.svg"),
    )
    args = parser.parse_args()
    roots = {f"{ARM}_s{seed}": args.artifact_root / f"{ARM}_s{seed}" for seed in SEEDS}
    telemetry = read_training_telemetry(roots)
    from research.t11_screen_20261017.watch import training_status

    statuses = training_status(args.artifact_root)
    result = analyze([], statuses, telemetry)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_svg.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    args.out_md.write_text(markdown(result))
    telemetry_svg(telemetry, args.out_svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
