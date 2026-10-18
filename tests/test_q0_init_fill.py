"""A2 supplemental initializer-fidelity coverage tests."""

from __future__ import annotations

import hashlib
import json

from research.q0q1_20261017 import audit_b8_init, init_fill_bank
from research.q0q1_20261017.audit_common import canonical


def _row(*, mix: str = "scripted", open_: bool = False) -> dict:
    return {
        "mix": mix,
        "short_lt700": True,
        "long_ge800": True,
        "crowded": True,
        "open": open_,
        "exact_q_tie": True,
        "boost_eligible": True,
        "action_a": 1,
        "action_b": 1,
        "fallback_used": False,
        "mask_disagreement": False,
        "wrapper_disagreement": False,
    }


def test_a2_resets_unmarked_r5_open_flags_before_supplemental_fill(tmp_path, monkeypatch):
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"B8")
    digest = hashlib.sha256(b8.read_bytes()).hexdigest()
    monkeypatch.setattr(audit_b8_init, "B8_SHA256", digest)
    rows = [
        _row(mix=("scripted", "frozen", "mixed")[index % 3], open_=True) for index in range(4095)
    ]
    source = tmp_path / "mass.json"
    source.write_text(
        canonical(
            {
                "b8": {"sha256": digest},
                "development_packet": {
                    "fidelity_corpus": {
                        "rows": rows,
                        "candidate_observations_inspected": len(rows),
                        "bank_audit_pass": True,
                    },
                },
            }
        )
    )

    def fake_fill(_b8, reservoir, **_kwargs):
        reservoir.add(_row(mix="scripted", open_=True))
        return {
            "additional_episodes": 1,
            "additional_candidate_observations": 1,
            "additional_excluded_trapped_observations": 0,
            "open_definition": audit_b8_init.OPEN_DEFINITION,
        }

    monkeypatch.setattr(audit_b8_init, "_run_supplemental_fill", fake_fill)
    result = audit_b8_init.run_observations(
        b8,
        source,
        supplemental_episodes=1,
        supplemental_partials_dir=tmp_path / "partials",
        fill_bank_receipt=tmp_path / "fill.json",
        fill_bank_receipt_sha256="f" * 64,
    )
    assert result["status"] == "COMPLETE"
    assert result["source_open_flags_reset"] is True
    assert result["additional_episodes"] == 1


def test_fill_runs_fixed_three_world_wave_then_stops_when_coverage_closes(tmp_path, monkeypatch):
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"B8")
    receipt = tmp_path / "fill.json"
    receipt.write_text("{}")
    worlds = [
        {"ordinal": index, "world_seed": index, "mix": ("scripted", "frozen", "mixed")[index % 3]}
        for index in range(6)
    ]
    monkeypatch.setattr(
        init_fill_bank,
        "load_receipt",
        lambda **_kwargs: {
            "bank": {"worlds": worlds, "allocation_semantic_sha256": "a" * 64, "namespace": "n"}
        },
    )
    import research.q0q1_20261017.mass_execution as execution

    calls = []

    def execute(tasks, **kwargs):
        calls.append((tasks, kwargs["partials_dir"]))
        kwargs["partials_dir"].mkdir(parents=True)
        (kwargs["partials_dir"] / "manifest.json").write_text("{}")
        return [
            {
                "candidates": [_row(mix=task["world"]["mix"], open_=number == 0)],
                "excluded_trapped": 0,
            }
            for number, task in enumerate(tasks)
        ]

    monkeypatch.setattr(execution, "execute", execute)
    sampler = audit_b8_init.CoverageReservoir()
    sampler.add(_row(open_=False))
    result = audit_b8_init._run_supplemental_fill(
        b8,
        sampler,
        fill_receipt=receipt,
        fill_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        episodes=6,
        workers=2,
        partials_dir=tmp_path / "partials",
        max_seconds=20,
    )
    assert result["additional_episodes"] == 3
    assert len(calls) == 1 and len(calls[0][0]) == 3
    assert result["waves"][0]["missing_coverage_after_wave"] == []


def test_fill_bank_is_twelve_unique_round_robin_worlds():
    bank = init_fill_bank.build_world_bank()
    assert bank["namespace"] == init_fill_bank.NAMESPACE
    assert len(bank["worlds"]) == 12
    assert [row["mix"] for row in bank["worlds"]] == list(init_fill_bank.MIXES) * 4
    assert len({row["world_seed"] for row in bank["worlds"]}) == 12


def test_fill_receipt_rejects_source_or_digest_substitution(tmp_path):
    bank = init_fill_bank.build_world_bank()
    receipt = {
        "schema": init_fill_bank.SCHEMA,
        "source_sha256": init_fill_bank.source_manifest(),
        "bank": bank,
        "audit": {
            "pass": True,
            "registered_collision_seeds": [],
            "world_bank_semantic_sha256": bank["allocation_semantic_sha256"],
        },
    }
    path = tmp_path / "receipt.json"
    raw = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    loaded = init_fill_bank.load_receipt(
        env={
            init_fill_bank.RECEIPT_ENV: str(path),
            init_fill_bank.RECEIPT_SHA256_ENV: hashlib.sha256(raw).hexdigest(),
        }
    )
    assert loaded["bank"]["allocation_semantic_sha256"] == bank["allocation_semantic_sha256"]
    with __import__("pytest").raises(ValueError, match="SHA-256"):
        init_fill_bank.load_receipt(
            env={init_fill_bank.RECEIPT_ENV: str(path), init_fill_bank.RECEIPT_SHA256_ENV: "0" * 64}
        )


def test_a2_resets_only_unmarked_legacy_open_rows():
    rows = [
        _row(open_=True),
        dict(_row(open_=True), open_definition=audit_b8_init.OPEN_DEFINITION),
    ]
    reset = audit_b8_init._reset_stale_open(rows, open_definition=None)
    assert [row["open"] for row in reset] == [False, True]


def test_fill_refuses_the_registered_hard_time_cap_before_loading_models(tmp_path):
    with __import__("pytest").raises(ValueError, match="1..900"):
        audit_b8_init._run_supplemental_fill(
            tmp_path / "missing-b8.pth",
            audit_b8_init.CoverageReservoir(),
            fill_receipt=tmp_path / "missing-receipt.json",
            fill_receipt_sha256="0" * 64,
            episodes=1,
            workers=1,
            partials_dir=tmp_path / "partials",
            max_seconds=901,
        )
