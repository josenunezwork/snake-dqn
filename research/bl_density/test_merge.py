"""Synthetic sidecar-only coverage for the DENSITY-2 Mac merge."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from research.bl_density.analysis import _identity_replay, merge
from research.bl_density.completion import SCIENCE_COMMIT
from research.bl_density.runner import _seed
from research.bl_ppo.exports import digest, export_json

NAMESPACE = "bl-density-round14b-20261022"
ARMS = ("A", "B70m", "B70p", "C-P1")
MIXES = ("scripted", "frozen", "mixed")


def _record(arm: str, mix: str, index: int, *, cpu: float = 1.0) -> dict:
    value = {
        "episode_id": f"{NAMESPACE}/{arm}/{mix}/{index:03d}",
        "namespace": NAMESPACE,
        "arm": arm,
        "mix": mix,
        "complete": True,
        "seed": _seed(arm, mix, index, namespace=NAMESPACE),
        "plan": {"seed": _seed(arm, mix, index, namespace=NAMESPACE)},
        "horizon": 5000,
        "sample_stride": 200,
        "decisions": 100,
        "cpu_seconds": cpu,
        "wall_seconds": 1.0,
        "h300_calls": 1,
        "h300_branch_decisions": 1,
        "h300_cpu_seconds": 0.1,
        "lengths": [800],
        "sampled": [
            {
                "length": 800,
                "prefilter": True,
                "h300_doom": True,
                "natural_doom": True,
                "features": [0.0],
                "witnessed_avoidable": True,
                "death_cause": "DEATH_SELF",
                "all_snake_lengths": [800, 20, 20, 20, 20, 20],
                "total_occupied_cells": 900,
                "map_cells": 12035,
                "hero_occupancy": 800 / 12035,
                "nearest_opponent_head_manhattan": 5,
                "nearest_opponent_body_manhattan": 3,
                "wall_distance_ahead": 10,
            }
        ],
    }
    if arm == "C-P1":
        value["construction"] = {"fixture": {"body_lengths": [900, 600, 400, 300, 150, 150]}}
    env = _bindings()
    value.update({field: env[key] for field, key in _field_keys().items()})
    value.update(
        metrics_commit=SCIENCE_COMMIT,
        metrics_science_commit=SCIENCE_COMMIT,
        metrics_completion_amendment_sha256=digest(
            Path(__file__).with_name("METRICS_AMENDMENT_05.md")
        ),
    )
    return value


def _field_keys():
    return {
        "metrics_sha256": "BL_DENSITY_METRICS_SHA256",
        "metrics_amendment_sha256": "BL_DENSITY_AMENDMENT_SHA256",
        "metrics_capacity_amendment_sha256": "BL_DENSITY_CAPACITY_AMENDMENT_SHA256",
        "metrics_round14_amendment_sha256": "BL_DENSITY_ROUND14_AMENDMENT_SHA256",
        "metrics_replication_amendment_sha256": "BL_DENSITY_REPLICATION_AMENDMENT_SHA256",
    }


def _bindings():
    names = ["METRICS.md"] + [f"METRICS_AMENDMENT_0{i}.md" for i in range(1, 5)]
    return {
        key: digest(Path(__file__).with_name(name))
        for key, name in zip(_field_keys().values(), names)
    }


def _receipt(root):
    paths = [
        p
        for p in root.rglob("*.json")
        if p.name not in {"manifest.json", "receipt.json", "poll-000015.json"}
    ]
    inventory = [
        {"path": p.relative_to(root).as_posix(), "sha256": digest(p), "size": p.stat().st_size}
        for p in paths
    ]
    (root / "receipt.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "stop_verified": True,
                "commit": SCIENCE_COMMIT,
                "job_id": root.name,
                "manifest_sha256": digest(root / "manifest.json"),
                "generation": "g",
                "exports": inventory,
            }
        )
    )
    if json.loads((root / "manifest.json").read_text())["density"]["phase"] == "production":
        (root / "poll-000015.json").write_text(
            json.dumps(
                {
                    "exports": inventory,
                    "terminal": {"generation": "g", "job_id": root.name, "returncode": 0},
                }
            )
        )


def _tree(root: Path, *, completion: bool = False) -> Path:
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "commit": SCIENCE_COMMIT,
                "job_id": root.name,
                "env": _bindings(),
                "density": {
                    "namespace": NAMESPACE,
                    "total_episodes": 492,
                    "horizon": 5000,
                    "sample_stride": 200,
                    "phase": "production-completion" if completion else "production",
                    "science_commit": SCIENCE_COMMIT,
                },
            }
        )
    )
    (root / ("episodes-replay" if completion else "episodes")).mkdir()
    if completion:
        export_json(
            root / "attempt.json",
            {
                "schema": "bl-density-attempt/v1",
                "namespace": NAMESPACE,
                "b8_sha256": "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8",
                "attempted_episode_count": 16,
                "attempt_cpu_overhead_seconds": 0.0,
            },
        )
    _receipt(root)
    return root


def _write(root: Path, record: dict, *, replay: bool = False) -> None:
    leaf = "episodes-replay" if replay else "episodes"
    export_json(root / leaf / (record["episode_id"].replace("/", "_") + ".json"), record)
    _receipt(root)


def _patched_allocation(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, int]]:
    import research.bl_density.runner as runner

    allocation = {arm: {mix: 4 for mix in MIXES} for arm in ARMS}
    monkeypatch.setattr(runner, "DENSITY2_ALLOCATION", allocation)
    return allocation


def _complete_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    allocation = _patched_allocation(monkeypatch)
    original, completion = _tree(tmp_path / "original"), _tree(
        tmp_path / "completion", completion=True
    )
    for arm, by_mix in allocation.items():
        for mix, count in by_mix.items():
            for index in range(count):
                row = _record(arm, mix, index, cpu=1.0)
                # One fixed, completion-session cell makes CPU rescaling observable.
                if (arm, mix) == ("B70p", "mixed"):
                    row["cpu_seconds"] = 0.4
                    _write(completion, row)
                else:
                    _write(original, row)
    for mix in MIXES:
        for index in range(4):
            replay = _record("A", mix, index, cpu=0.5)
            _write(completion, replay, replay=True)
    return original, completion


def _session_identity_records(session: str) -> list[dict]:
    records = []
    for mix in MIXES:
        for index in range(4):
            row = _record("A", mix, index)
            specs = [
                ["checkpoint", f"/content/snake-cli/{session}/inputs/checkpoints/opponent.pth"],
                ["scripted", "random_safe"],
            ]
            row["plan"]["specs"] = specs
            row["construction"] = {"roster_specs": deepcopy(specs)}
            records.append(row)
    return records


def test_identity_replay_ignores_only_checkpoint_session_prefixes() -> None:
    original = _session_identity_records("11111111-1111-4111-8111-111111111111")
    replay = _session_identity_records("22222222-2222-4222-8222-222222222222")
    before = deepcopy((original, replay))
    value = _identity_replay(original, replay, NAMESPACE)
    assert value["status"] == "IDENTICAL"
    assert value["matched_episodes"] == 12
    assert value["mismatched_episode_ids"] == []
    assert (original, replay) == before


@pytest.mark.parametrize("changed", ["plan_basename", "roster_basename", "decisions", "scripted"])
def test_identity_replay_refuses_basename_or_gameplay_changes(changed: str) -> None:
    original = _session_identity_records("11111111-1111-4111-8111-111111111111")
    replay = _session_identity_records("22222222-2222-4222-8222-222222222222")
    if changed == "plan_basename":
        replay[0]["plan"]["specs"][0][1] = "/content/snake-cli/other/different.pth"
    elif changed == "roster_basename":
        replay[0]["construction"]["roster_specs"][0][1] = "/content/snake-cli/other/different.pth"
    elif changed == "decisions":
        replay[0]["decisions"] += 1
    else:
        replay[0]["plan"]["specs"][1][1] = "greedy_food"
    value = _identity_replay(original, replay, NAMESPACE)
    assert value["status"] == "IDENTITY_FAIL"
    assert value["matched_episodes"] == 11
    assert value["mismatched_episode_ids"] == [replay[0]["episode_id"]]


def test_merge_requires_verified_union_and_applies_session_confound_rule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original, completion = _complete_fixture(monkeypatch, tmp_path)
    value = merge(
        [original, completion],
        completion_tree=completion,
        feature_names=["q_gap"],
    )
    assert value["status"] == "COMPLETE"
    identity = next(iter(value["identity_replay"]["sessions"].values()))
    assert identity["status"] == "IDENTICAL"
    assert identity["cpu_calibration_ratio"] == 0.5
    assert value["completion_overhead"]["charged_uniformly_over_registered_episodes"] == 48
    assert value["completion_overhead"]["charged_overhead_cpu_seconds"] == 0.0
    assert value["session_confound"]["outside_registered_band"] is True
    decision = value["session_confound"]["effective_decisions"]["B70p"]["mixed"]
    assert decision["status"] == "INSUFFICIENT_SESSION"
    assert decision["raw"]["status"] == "INSUFFICIENT"
    assert decision["completion_cpu_rescaled_by_1_over_r"]["status"] == "REJECTED"


def test_merge_refuses_duplicate_and_foreign_normal_identities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original, completion = _complete_fixture(monkeypatch, tmp_path)
    duplicate = _record("A", "scripted", 0)
    _write(completion, duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        merge([original, completion], completion_tree=completion)

    foreign = _tree(tmp_path / "foreign")
    row = _record("A", "scripted", 0)
    row["namespace"] = "foreign"
    _write(foreign, row)
    with pytest.raises(ValueError, match="foreign"):
        merge([foreign, completion], completion_tree=completion)


def test_identity_mismatch_voids_without_analyzing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original, completion = _complete_fixture(monkeypatch, tmp_path)
    path = completion / "episodes-replay" / f"{NAMESPACE}_A_scripted_000.json"
    # Re-publish a distinct synthetic replay in a fresh fixture rather than mutating an export.
    path.unlink()
    Path(str(path) + ".sha256").unlink()
    Path(str(path) + ".complete").unlink()
    changed = _record("A", "scripted", 0, cpu=0.5)
    changed["decisions"] = 101
    export_json(path, changed)
    _receipt(completion)
    value = merge([original, completion], completion_tree=completion)
    assert value["status"] == "IDENTITY_FAIL"
    assert value["completion_void"] is True


def _remove_export(path):
    for suffix in ("", ".sha256", ".complete"):
        Path(str(path) + suffix).unlink()


def _replace_attempt(tree, **changes):
    path = tree / "attempt.json"
    value = json.loads(path.read_text())
    _remove_export(path)
    value.update(changes)
    export_json(path, value)
    _receipt(tree)


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 123),
        ("arm", "C-P1"),
        ("horizon", 8),
        ("sample_stride", 50),
        ("metrics_sha256", "f" * 64),
        ("metrics_commit", "a" * 40),
    ],
)
def test_merge_refuses_frozen_science_binding_changes(monkeypatch, tmp_path, field, value):
    original, completion = _complete_fixture(monkeypatch, tmp_path)
    path = original / "episodes" / f"{NAMESPACE}_A_scripted_000.json"
    record = json.loads(path.read_text())
    _remove_export(path)
    record[field] = value
    export_json(path, record)
    _receipt(original)
    with pytest.raises(ValueError, match="identity|binding|commit"):
        merge([original, completion], completion_tree=completion)


def test_merge_refuses_uninventoried_payload_before_json_parse(monkeypatch, tmp_path):
    from research.bl_ppo.exports import export_bytes

    original, completion = _complete_fixture(monkeypatch, tmp_path)
    export_bytes(completion / "episodes" / "not-json.json", b"invalid JSON")
    with pytest.raises(ValueError, match="absent from immutable receipt"):
        merge([original, completion], completion_tree=completion)


def test_terminal_missing_replay_voids_and_missing_units_stay_incomplete(monkeypatch, tmp_path):
    original, completion = _complete_fixture(monkeypatch, tmp_path)
    _remove_export(completion / "episodes" / f"{NAMESPACE}_B70p_mixed_000.json")
    value = merge([original, completion], completion_tree=completion)
    assert value["status"] == "INCOMPLETE"
    assert value["completion_counts"]["B70p"]["mixed"]["missing_episodes"] == 1
    _remove_export(completion / "episodes-replay" / f"{NAMESPACE}_A_mixed_000.json")
    value = merge([original, completion], completion_tree=completion)
    assert value["status"] == "IDENTITY_FAIL"
    assert value["completion_void"] is True


def test_missing_terminal_accounting_is_explicit_incomplete(monkeypatch, tmp_path):
    original, completion = _complete_fixture(monkeypatch, tmp_path)
    _remove_export(completion / "attempt.json")
    value = merge([original, completion], completion_tree=completion)
    assert value["status"] == "INCOMPLETE"
    assert value["reason"] == "missing_verified_completion_accounting"


def test_two_sessions_have_individual_ratios_weighted_overhead_and_partial_cost(
    monkeypatch, tmp_path
):
    original, completion = _complete_fixture(monkeypatch, tmp_path)
    second = _tree(tmp_path / "second", completion=True)
    # Transfer one entire synthetic cell to a distinct session; never duplicate primary IDs.
    for i in range(4):
        path = original / "episodes" / f"{NAMESPACE}_B70m_scripted_{i:03d}.json"
        row = json.loads(path.read_text())
        _remove_export(path)
        row["cpu_seconds"] = 2.0
        _write(second, row)
    _receipt(original)
    for mix in MIXES:
        for i in range(4):
            _write(second, _record("A", mix, i, cpu=2.0), replay=True)
    _replace_attempt(
        completion,
        attempt_cpu_overhead_seconds=16,
        attempted_episode_count=17,
        partial_episode_cpu_seconds=[
            {"episode_id": f"{NAMESPACE}/B70p/mixed/000", "kind": "primary", "cpu_seconds": 3}
        ],
    )
    _replace_attempt(second, attempt_cpu_overhead_seconds=32)
    value = merge([original, completion, second], completion_tree=[completion, second])
    assert value["status"] == "COMPLETE"
    assert sorted(value["session_confound"]["cpu_calibration_ratios_r"].values()) == [0.5, 2]
    raw = value["session_confound"]["raw_evaluation"]
    scaled = value["session_confound"]["completion_cpu_rescaled_by_1_over_r_evaluation"]
    assert raw["overhead_cpu_seconds"] == pytest.approx((16 + 32) / 33 * 48)
    assert scaled["overhead_cpu_seconds"] == pytest.approx((16 / 0.5 + 32 / 2) / 33 * 48)
    assert raw["cells"]["B70p"]["mixed"]["partial_episode_cpu_seconds"] == 3
    assert scaled["cells"]["B70p"]["mixed"]["partial_episode_cpu_seconds"] == 6


def test_missing_unit_partial_attempt_counts_and_cpu_are_retained(monkeypatch, tmp_path):
    original, completion = _complete_fixture(monkeypatch, tmp_path)
    identity = f"{NAMESPACE}/B70p/mixed/000"
    _remove_export(completion / "episodes" / (identity.replace("/", "_") + ".json"))
    _replace_attempt(
        completion,
        partial_episode_cpu_seconds=[{"episode_id": identity, "kind": "primary", "cpu_seconds": 3}],
    )
    value = merge([original, completion], completion_tree=completion)
    assert value["status"] == "INCOMPLETE"
    cell = value["completion_counts"]["B70p"]["mixed"]
    assert cell["attempted_episodes"] == 4
    assert cell["complete_episodes"] == 3
    assert (
        value["session_confound"]["raw_evaluation"]["cells"]["B70p"]["mixed"][
            "partial_episode_cpu_seconds"
        ]
        == 3
    )
