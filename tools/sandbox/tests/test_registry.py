"""Positive admission and ancestry/path boundary regression tests."""

import copy
import hashlib
import json

import pytest

from tools.sandbox.registry import (
    f4_fold,
    load_registry,
    safe_path,
    validate_input,
    validate_rows,
    world_seed,
)


def test_committed_registry_excludes_protected_ancestry():
    value = load_registry()
    assert len(value["exclusions"]["census_parents"]) == 29
    assert len(value["exclusions"]["r30_validation_parents"]) == 81
    excluded = set(value["excluded_parents"])
    assert all(r["ultimate_parent_id"] not in excluded for r in value["lives"])
    assert all(f4_fold(r["ultimate_parent_id"]) != 0 for r in value["lives"])
    assert value["counts"]["allowed_fit_lives"] == 730


def test_foreign_parent_row_alias_and_duplicate_refused():
    registry = load_registry()
    life = registry["lives"][0]
    row = dict(
        life_id=life["source_life_id"],
        parent_world=life["ultimate_parent_id"],
        row_id=life["source_life_id"] + "/1",
    )
    validate_rows([row], registry)
    for patch in (
        {"parent_world": "foreign"},
        {"life_id": "foreign"},
        {"row_id": "foreign"},
        {"row_id": None},
    ):
        with pytest.raises(ValueError):
            validate_rows([{**row, **patch}], registry)
    with pytest.raises(ValueError):
        validate_rows([row, row], registry)
    excluded = registry["exclusions"]["r30_validation_parents"][0]
    with pytest.raises(ValueError):
        validate_rows([{**row, "parent_world": excluded}], registry)


@pytest.mark.parametrize(
    "path",
    ["/tmp/f5/out", "/tmp/D4096/a", "/tmp/t34-round0-study-01/a", "research/colab_cli/new.py"],
)
def test_protected_paths_refused_before_open(path):
    with pytest.raises(ValueError):
        safe_path(path)


def test_symlink_to_protected_source_refused_before_open(tmp_path):
    alias = tmp_path / "innocent"
    alias.symlink_to(tmp_path / "f5" / "not-present")
    with pytest.raises(ValueError):
        safe_path(alias)


def test_benign_path_component_containing_f5_is_not_a_study(tmp_path):
    benign = tmp_path / "hash-aaf5b3" / "input.npz"
    assert safe_path(benign) == benign


def test_domain_separated_full_uint64_worlds():
    seed = world_seed("sandbox-b", 0)
    assert seed == int.from_bytes(hashlib.sha256(b"a100-dev/sandbox-b/0").digest()[:8], "big")
    assert seed > 2**32
    assert seed != world_seed("sandbox-a", 0)
    registry = load_registry()
    record = {"kind": "worlds", "study": "sandbox-b", "indices": [0], "seeds": [seed]}
    validate_input(record, registry)
    with pytest.raises(ValueError):
        validate_input({**record, "seeds": [seed % 2**32]}, registry)
    with pytest.raises(ValueError):
        world_seed("sandbox-b", True)
    registry = copy.deepcopy(registry)
    registry["excluded_seeds"].append(seed)
    with pytest.raises(ValueError):
        validate_input(record, registry)


def test_modified_data_and_b8_inputs_refused(tmp_path):
    p = tmp_path / "foreign.npz"
    p.write_bytes(b"foreign")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    for kind in ["closed-f0", "frozen-checkpoint"]:
        with pytest.raises(ValueError):
            validate_input({"kind": kind, "dose": "D1024", "path": str(p), "sha256": sha})


def test_mutated_registry_with_protected_life_refused(tmp_path):
    reg = load_registry()
    reg["lives"][0]["ultimate_parent_id"] = reg["excluded_parents"][0]
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(reg))
    with pytest.raises(ValueError):
        load_registry(p)
