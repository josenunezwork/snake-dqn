"""Synthetic metadata tests; no scientific payload or real study is opened."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from research.bl_residual import f5_r10a_annex as annex


def fold(parent: str) -> int:
    return (
        int.from_bytes(
            hashlib.sha256(f"bl-residual-f3-split-v1/{parent}".encode()).digest()[:8], "big"
        )
        % 5
    )


@pytest.fixture
def metadata() -> dict:
    """Build 960 synthetic lives, including a 192-life prefix and ordinary sibling."""
    parents = [f"synthetic-{i}" for i in range(1000) if fold(f"synthetic-{i}") != 0][:40]
    dev = [f"synthetic-{i}" for i in range(1000) if fold(f"synthetic-{i}") == 0][:29]
    worlds = {
        parent: {
            "allocation_id": parent,
            "seed": 1_000_000 + i,
            "allocation_index": i,
            "stream": "watch6",
            "horizon": 14400,
        }
        for i, parent in enumerate(parents + dev)
    }
    unions = {
        dose: {
            "schema": "bl-labels-f0-verified-ledger/v1",
            "namespace": f"test-{dose}",
            "units": [],
            "allocation": list(worlds.values()),
        }
        for dose in ("d256", "d1024")
    }
    catalog, records = [], []
    for i in range(961):
        dose = "d256" if i < 192 else "d1024"
        namespace, parent = unions[dose]["namespace"], parents[i % 40]
        life = f"{namespace}/{parent}/{i}"
        unit = {
            "unit_id": f"{parent}--life-{i:06d}",
            "path": f"lives/{i}.json",
            "sha256": "a" * 64,
            "size": 123,
        }
        unions[dose]["units"].append(unit)
        catalog.append(
            {
                "life_id": life,
                "parent_world": parent,
                "allocation_index": i,
                "rescue_bearing": i < 960,
                "stream": "S-Watch6",
            }
        )
        if i < 960:
            records.append(
                {
                    "source_life_id": life,
                    "ultimate_parent_id": parent,
                    "original_fold": fold(parent),
                    "original_allocation": worlds[parent],
                    "original_allocation_index": i,
                    "payload_sha256": unit["sha256"],
                    "payload_size": unit["size"],
                    "stream": "S-Watch6",
                    "collection_source_commit": "c" * 40,
                }
            )
    selected = [row["life_id"] for row in catalog[:960]]
    current = {
        "catalog": catalog,
        "parents": {r["life_id"]: r["parent_world"] for r in catalog},
        "folds": {r["life_id"]: fold(r["parent_world"]) for r in catalog},
        "doses": {"1024": {"train_lives": selected}, "256": {"train_lives": selected[:192]}},
    }
    prior = {"catalog": catalog[:192], "doses": {"256": {"train_lives": selected[:192]}}}
    result = {
        "d256_allocation": prior,
        "d1024_allocation": current,
        "development": {
            "parent_worlds": dev,
            "parent_count": 29,
            "designation": "DEVELOPMENT_FOREVER",
        },
        "r30_inventory": {"lives": records, "row_totals": {"status": "DEFERRED"}},
    }
    for dose in ("d256", "d1024"):
        result[f"{dose}_union"] = unions[dose]
        result[f"{dose}_collection"] = {
            "allocation": unions[dose]["allocation"],
            "namespace": unions[dose]["namespace"],
            "source_commit": "c" * 40,
        }
        result[f"{dose}_control"] = {"input_hashes": {"b8.pth": "b" * 64}}
    return result


def test_whole_bank_counts_and_exact_encoding() -> None:
    rows = annex.allocate(0)
    assert len(rows) == len({r["seed"] for r in rows}) == len({r["parent_id"] for r in rows}) == 684
    assert [r["allocation_index"] for r in rows] == list(range(684))
    assert sum(r["timing_only"] for r in rows) == 12
    assert sum("ordinary_phase" in r for r in rows) == 128
    first = rows[0]
    assert first["parent_id"] == f"{annex.NAMESPACE}/r=0/archive/watch6/0000"
    assert first["seed"] == int.from_bytes(
        hashlib.sha256(b"bl-f5-predat-v2/20261025|r=0/archive/watch6|0").digest()[:4], "big"
    )
    assert rows == annex.allocate(0)


@pytest.mark.parametrize("round_index", [-1, 1000, True, 1.0])
def test_allocator_refuses_invalid_round(round_index: int) -> None:
    with pytest.raises(ValueError, match="invalid round"):
        annex.allocate(round_index)


def test_phase_records_full_digest_and_covers_schedule() -> None:
    parent = "synthetic-parent"
    record = annex.phase(parent)
    raw = f"{annex.NAMESPACE}|ordinary-phase|{parent}|{record['counter']}".encode()
    assert record["sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["q"] == int(record["sha256"], 16) % 200
    assert len(range(record["q"], 5000, 200)) == 25
    assert sorted(d for q in range(200) for d in range(q, 5000, 200)) == list(range(5000))


def test_collision_audit_uses_seed_alias_and_internal_identity() -> None:
    rows = annex.allocate(0)
    historical = [
        {"parent_id": "renamed", "seed": rows[0]["seed"], "aliases": ["other"]},
        {"parent_id": "physical-ancestor", "seed": -1, "aliases": [rows[1]["parent_id"]]},
    ]
    hits = annex.collision_audit(rows + [rows[0]], historical)
    assert any(r["kind"] == "internal_collision" for r in hits)
    assert {r["old"]["parent_id"] for r in hits if "old" in r} == {"renamed", "physical-ancestor"}
    assert annex.collision_audit(annex.allocate(1), historical) == []


def test_metadata_only_corpus_and_sibling_scope(metadata: dict) -> None:
    corpus, exclusions, _ = annex.corpus_metadata(metadata)
    assert corpus["eligible_selected_life_count"] == 960
    assert corpus["nested_d256_prefix_count"] == 192
    assert corpus["eligible_selected_parent_count"] == 40
    assert corpus["ordinary_only_sibling_life_count"] == 1
    assert corpus["total_eligible_life_metadata_count"] == 961
    assert corpus["row_count"] is None
    assert not corpus["both_row_strata_verified"]
    assert exclusions["permanent_development_count"] == 29
    assert set(exclusions["permanent_development_parents"]) <= set(
        exclusions["original_fold0_parents"]
    )


@pytest.mark.parametrize(
    "corruption",
    ["prefix", "catalog", "ancestry", "fold", "unit_hash", "allocation", "stream", "census"],
)
def test_corrupt_metadata_refuses(metadata: dict, corruption: str) -> None:
    values = copy.deepcopy(metadata)
    if corruption == "prefix":
        values["d256_allocation"]["doses"]["256"]["train_lives"].reverse()
    elif corruption == "catalog":
        values["d1024_allocation"]["catalog"].append(
            {**values["d1024_allocation"]["catalog"][0], "allocation_index": -1}
        )
    elif corruption == "ancestry":
        values["r30_inventory"]["lives"][0]["ultimate_parent_id"] = "unknown"
    elif corruption == "fold":
        values["r30_inventory"]["lives"][0]["original_fold"] = 0
    elif corruption == "unit_hash":
        values["r30_inventory"]["lives"][0]["payload_sha256"] = "f" * 64
    elif corruption == "allocation":
        values["d256_collection"]["allocation"] = []
    elif corruption == "stream":
        values["r30_inventory"]["lives"][0]["stream"] = "foreign"
    else:
        values["development"]["parent_worlds"].pop()
    with pytest.raises(ValueError):
        annex.corpus_metadata(values)


def test_build_cannot_claim_adoption_or_zero_cov(
    metadata: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Path.read_bytes

    def guard(path: Path) -> bytes:
        assert path == Path(annex.__file__), f"unexpected read: {path}"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guard)
    report = annex.build(metadata, [], {})
    assert report["disposition"] == "HOLD_BEFORE_EXPOSURE"
    assert not report["adoption_ready"] and not report["adopted"]
    assert report["cov"]["count_U"] is None and report["cov"]["inventory"] is None
    assert report["cov"]["inventory_sha256"] is None
    assert report["cov"]["ceiling_U"] == 46080
    assert report["allocation"]["accepted_round"] is None
    assert report["allocation"]["accepted_allocation_sha256"] is None
    assert len(report["allocation"]["provisional_parents"]) == 684
    assert report["access_attestation"]["referenced_payload_paths_followed"] is False


def test_collision_rejects_whole_round(metadata: dict) -> None:
    values = copy.deepcopy(metadata)
    blocked = annex.allocate(0)[0]["seed"]
    for role in ("d256_union", "d1024_union"):
        values[role]["allocation"][0]["seed"] = blocked
    report = annex.build(values, [], {})
    bank = report["allocation"]
    assert bank["rejected_rounds_against_inspected_metadata_only"][0]["round"] == 0
    assert bank["first_clear_round_against_inspected_metadata_only"] == 1
    assert bank["provisional_parents"] == annex.allocate(1)
    assert bank["accepted_round"] is None  # partial historical audit cannot accept a bank


@pytest.mark.parametrize("corruption", ["duplicate_selected", "parent", "fold"])
def test_ordinary_sibling_identity_refuses(metadata: dict, corruption: str) -> None:
    values = copy.deepcopy(metadata)
    sibling = values["d1024_allocation"]["catalog"][-1]
    if corruption == "duplicate_selected":
        values["d1024_allocation"]["catalog"][0]["rescue_bearing"] = False
    elif corruption == "parent":
        other = values["d1024_allocation"]["catalog"][1]["parent_world"]
        sibling["parent_world"] = other
        values["d1024_allocation"]["parents"][sibling["life_id"]] = other
    else:
        values["d1024_allocation"]["folds"][sibling["life_id"]] = 0
    with pytest.raises(ValueError, match="stratum|sibling ancestry|sibling fold"):
        annex.corpus_metadata(values)


def test_conflicting_prior_duplicate_is_not_overwritten(metadata: dict) -> None:
    values = copy.deepcopy(metadata)
    prior = values["d256_allocation"]["catalog"]
    prior.insert(0, {**prior[0], "allocation_index": -1})
    with pytest.raises(ValueError, match="conflicting prior catalog"):
        annex.corpus_metadata(values)


def test_main_binds_consumed_manifest_bytes(
    metadata: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "R10A_INPUTS_20261025.json"
    original, replacement = b'{"fixture":"original"}\n', b'{"fixture":"replacement"}\n'
    path.write_bytes(original)
    published = []

    def replace_after_read(plan: dict) -> tuple[dict, list]:
        assert plan == {"fixture": "original"}
        path.write_bytes(replacement)
        return metadata, []

    monkeypatch.setattr(annex, "load_sources", replace_after_read)
    monkeypatch.setattr(annex, "source_closure", lambda repo: {})
    monkeypatch.setattr(annex, "publish", lambda directory, report: published.append(report))
    monkeypatch.setattr(
        sys, "argv", ["annex", "--inputs", str(path), "--publish", str(tmp_path / "publication")]
    )
    annex.main()
    assert published[0]["input_manifest_sha256"] == annex.sha(original)
    assert published[0]["input_manifest_sha256"] != annex.sha(replacement)


@pytest.mark.parametrize(
    "relative",
    [
        "D4096/control.json",
        "t34-round0-study-01/control.json",
        "lives/control.json",
        "outcomes/control.json",
        "f4_postmortem_20261025/manifest.json",
    ],
)
def test_forbidden_paths_before_read(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ValueError):
        annex.safe_path(tmp_path / relative)


def test_symlink_refused(tmp_path: Path) -> None:
    target = tmp_path / "metadata.json"
    target.write_text("{}")
    link = tmp_path / "control.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        annex.safe_path(link)


def test_source_allowlist_and_changed_hash(tmp_path: Path) -> None:
    sources = {}
    for role, basename in annex.FILES.items():
        path = tmp_path / role / basename
        path.parent.mkdir()
        raw = b"{}\n"
        path.write_bytes(raw)
        Path(str(path) + ".sha256").write_text(f"{annex.sha(raw)}  {basename}\n")
        sources[role] = {"path": str(path), "sha256": annex.sha(raw), "seal": "sidecar"}
    plan = {"schema": "bl-f5-r10a-inputs/v1", "sources": sources}
    with pytest.raises(ValueError, match="wrong v2 draft"):
        annex.load_sources(plan)
    sources["d256_union"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="changed metadata"):
        annex.load_sources(plan)
    sources["extra"] = sources["draft"]
    with pytest.raises(ValueError, match="foreign source roles"):
        annex.load_sources(plan)


def test_publication_is_reproducible_create_only_and_bound(metadata: dict, tmp_path: Path) -> None:
    report = annex.build(metadata, [], {})
    left, right = tmp_path / "a", tmp_path / "b"
    hashes = annex.publish(left, report)
    assert annex.publish(right, report) == hashes
    for path in left.iterdir():
        assert path.read_bytes() == (right / path.name).read_bytes()
    completion = json.loads((left / f"{annex.NAME}.complete.json").read_bytes())
    assert completion["files"] == hashes
    for name, digest in hashes.items():
        assert annex.sha((left / name).read_bytes()) == digest
    human = (left / f"{annex.NAME}.md").read_text()
    assert hashes[f"{annex.NAME}.json"] in human
    assert "HOLD" in human and "UNRESOLVED" in human
    before = {p.name: p.read_bytes() for p in left.iterdir()}
    with pytest.raises(ValueError, match="create-only"):
        annex.publish(left, report)
    assert before == {p.name: p.read_bytes() for p in left.iterdir()}


def test_completion_is_published_last(
    metadata: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    linked = []
    original = annex.os.link

    def record(source: str, destination: Path) -> None:
        linked.append(Path(destination).name)
        original(source, destination)

    monkeypatch.setattr(annex.os, "link", record)
    annex.publish(tmp_path, annex.build(metadata, [], {}))
    assert linked[-1] == f"{annex.NAME}.complete.json"
    assert len(linked) == 5
