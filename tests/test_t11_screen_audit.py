"""Focused contracts for the T11 screen audit wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.redesign_m2_20261008 import dev_probe
from research.t11_screen_20261017 import audit
from research.t2_screen_20261015.watch_t2b import row_path


def _base_receipt() -> dict[str, object]:
    rosters = audit.seed_audit.candidate_rosters()
    return {
        "pass": True,
        "checks": {
            key: {"count": 96, "unique": True, "seeds": sorted(values)}
            for key, values in rosters.items()
        },
    }


def _round74_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, mix in sorted(audit._expected_initial_jobs()):
        values = {
            str(seed): 1
            for seed in dev_probe.probe_seeds(audit._MIX_WORLDS[mix], audit.INITIAL_ROUND)
        }
        rows.append(
            {
                "name": name,
                "mix": mix,
                "worlds": audit._MIX_WORLDS[mix],
                "round_index": 74,
                "mass": values,
                "deaths": values,
                "alive_frames": values,
            }
        )
    return rows


def _write_round74_evidence(path: Path) -> Path:
    rows = _round74_rows()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for row in rows:
        job = {key: row[key] for key in ("name", "mix", "worlds", "round_index")}
        raw_path = row_path(path.parent, job)
        raw_path.parent.mkdir(exist_ok=True)
        raw_path.write_text(json.dumps(row))
    receipt = {
        "schema": audit.SCHEMA,
        "round_index": 74,
        "pass": True,
        "checks": {"74": _base_receipt()["checks"]["74"]},
    }
    (path.parent / "seed_bank_audit_round74.json").write_text(json.dumps(receipt))
    return path


def test_initial_round_rejects_prior_evidence() -> None:
    with pytest.raises(ValueError, match="cannot exclude"):
        audit.audit(round_index=74, consumed_round74_evidence="prior.jsonl")


def test_extension_requires_prior_evidence() -> None:
    with pytest.raises(ValueError, match="requires exact"):
        audit.audit(round_index=75)


def test_extension_validates_complete_prior_evidence_and_excludes_only_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = _write_round74_evidence(tmp_path / "probes_round74.jsonl")
    seen: dict[str, object] = {}

    def fake_base(exclusions: tuple[Path, ...], **kwargs: object) -> dict[str, object]:
        seen["exclusions"] = exclusions
        seen["kwargs"] = kwargs
        return _base_receipt()

    monkeypatch.setattr(audit, "_base_audit_with_exclusions", fake_base)
    receipt = audit.audit(tmp_path, tmp_path, tmp_path / "round75.json", 75, prior)

    assert receipt["pass"] is True
    assert [row["kind"] for row in receipt["validated_documentary_exclusions"]] == [
        "validated-owned-round74-probe-transcript",
        "validated-owned-round74-audit-receipt",
        "validated-owned-round74-raw-rows",
    ]
    assert seen["exclusions"] == (
        prior.resolve(),
        (tmp_path / "seed_bank_audit_round74.json").resolve(),
        *[
            row_path(
                tmp_path,
                {"name": name, "mix": mix, "worlds": audit._MIX_WORLDS[mix], "round_index": 74},
            ).resolve()
            for name, mix in sorted(audit._expected_initial_jobs())
        ],
    )


def test_extension_keeps_unvalidated_neighbor_row_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = _write_round74_evidence(tmp_path / "probes_round74.jsonl")
    neighbor = tmp_path / "rows" / "unrelated.json"
    neighbor.write_text(json.dumps({"round_index": 74, "world_seed": 123}))
    seen: dict[str, object] = {}

    def fake_base(exclusions: tuple[Path, ...], **_kwargs: object) -> dict[str, object]:
        seen["exclusions"] = exclusions
        return _base_receipt()

    monkeypatch.setattr(
        audit,
        "_base_audit_with_exclusions",
        fake_base,
    )

    audit.audit(tmp_path, tmp_path, round_index=75, consumed_round74_evidence=prior)

    assert neighbor.resolve() not in seen["exclusions"]


def test_extension_rejects_partial_or_malformed_round74_transcript(tmp_path: Path) -> None:
    prior = _write_round74_evidence(tmp_path / "probes_round74.jsonl")
    rows = prior.read_text().splitlines()
    prior.write_text("\n".join(rows[:-1]) + "\n")

    with pytest.raises(ValueError, match="every registered job"):
        audit._validated_round74_evidence(prior, tmp_path)


def test_wrapper_surfaces_immutable_audit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        audit,
        "_base_audit_with_exclusions",
        lambda *_args, **_kwargs: {"pass": False, "checks": {"74": {"count": 96, "unique": True}}},
    )

    receipt = audit.audit(tmp_path, tmp_path, round_index=74)

    assert receipt["pass"] is False
    assert receipt["status"] == "FAIL"
