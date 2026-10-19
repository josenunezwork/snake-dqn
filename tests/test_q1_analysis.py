"""Receipt-bound analysis and prospective decision boundaries; no gameplay."""

import copy
import json

import pytest

from research.q0q1_20261017 import q1_analyze as a
from research.q0q1_20261017 import q1_bank as b
from research.q0q1_20261017 import q1_evaluate as e


def _manifest(delta=25.0):
    bank = b.load_world_bank()
    source = e.source_manifest()
    bank_sha = a._digest(bank)
    parity = {
        "schema": "q1-runtime-parity-receipt/v1",
        "pass": True,
        "bank_sha256": bank_sha,
        "b8_sha256": a.B8_SHA256,
        "source_manifest": source,
        "environment": {"sha256": "e" * 64},
    }
    shards = []
    for seed in b.TRAINING_SEEDS:
        worlds = e.shard_worlds(b.worlds_for_seed(bank, seed), 0, 1)
        for arm in (*a.ARMS, a.B8):
            ckpt = a.B8_SHA256 if arm == a.B8 else a._digest([arm, seed])
            provenance = (
                None
                if arm == a.B8
                else {
                    "schema": "q1-training-provenance/v1",
                    "arm": arm,
                    "training_seed": seed,
                    "endpoint": 8_000_000,
                    "clean_exit": True,
                    "checkpoint_sha256": ckpt,
                    "parity_pass": True,
                    "tripwire_triggered": False,
                    "environment_sha256": "e" * 64,
                    "runtime_parity_receipt_sha256": "f" * 64,
                    "counters": {
                        "actual_learner_decisions": 8_000_000,
                        "transition_budget_overshoot": 0,
                        "registered_overshoot_bound": 2303,
                        "emitted_replay_rows": 8_000_000,
                        "completed_updates": 31250,
                        "pending_update_debt": 0,
                        "truncation_rows_dropped": 0,
                        "pending_nstep_rows": 0,
                    },
                }
            )
            if provenance is not None:
                seats = 3 if arm in ("A", "B") else 1
                config = {
                    "seed": str(seed),
                    "transitions": "8000000",
                    "worlds": "128",
                    "learner_seats": str(seats),
                    "actors": "6",
                    "reward": "mass",
                    "kill_scale": "0.0",
                    "batch_scale": "2",
                    "obs_version": "bv8",
                    "anchor_mode": "v8margin",
                    "namespace": b.training_namespace(arm),
                    "schedule_package": "mature" if arm in ("B", "D") else "restart",
                    "collection_profile": "legacy" if arm in ("A", "B") else "task-aligned-n1",
                }
                provenance.update(
                    environment={"runtime": "fixture"},
                    config=config,
                    environment_sha256=a._digest({"runtime": "fixture"}),
                    config_sha256=a._digest(config),
                    source_sha256=e.sha256(b.ROOT / "research/redesign_m3c_20261008/train_m3c.py"),
                )
                provenance["counters"].update(
                    actors=6,
                    worlds=128,
                    learner_seats=seats,
                    registered_overshoot_bound=6 * 128 * seats - 1,
                )
            identity = {
                "training_seed": seed,
                "arm": arm,
                "checkpoint_sha256": ckpt,
                "b8_sha256": a.B8_SHA256,
                "endpoint": 8_000_000,
                "shard": 0,
                "shards": 1,
                "world_count": 128,
                "world_hashes": [w["world_semantic_sha256"] for w in worlds],
                "bank_sha256": bank_sha,
                "profile_digest": bank["profile"]["base_profile_digest"],
                "evaluator_source_sha256": source["research/q0q1_20261017/q1_evaluate.py"],
                "evaluator_source_manifest": source,
                "environment_sha256": "e" * 64,
                "runtime_parity_receipt_sha256": a._digest(parity),
                "training_provenance_sha256": None if provenance is None else a._digest(provenance),
            }
            rows = []
            for w in worlds:
                rows.append(
                    {
                        "schema": e.ROW_SCHEMA,
                        "training_seed": seed,
                        "arm": arm,
                        "checkpoint_step": 8_000_000,
                        "checkpoint_sha256": ckpt,
                        "b8_checkpoint_sha256": a.B8_SHA256,
                        "world_bank_sha256": bank_sha,
                        "world_profile_sha256": identity["profile_digest"],
                        **{
                            k: w[k]
                            for k in (
                                "initial_world_sha256",
                                "world_semantic_sha256",
                                "roster_semantic_sha256",
                                "world_seed",
                                "mix",
                            )
                        },
                        "horizon": 5000,
                        "frames_stepped": 5000,
                        "clean_exit": True,
                        "evaluator_source_sha256": identity["evaluator_source_sha256"],
                        "alive_frame_fraction": 1.0,
                        "no_death_episode": True,
                        "mass_integral": 100.0 + (delta if arm != a.B8 else 0.0),
                    }
                )
            shards.append(
                {
                    "schema": e.SCHEMA,
                    "status": "completed",
                    "clean_exit": True,
                    "identity": identity,
                    "runtime_parity": parity,
                    "training_provenance": provenance,
                    "rows": rows,
                    "rows_sha256": a._digest(rows),
                }
            )
    return {
        "schema": "q1-analysis-input/v1",
        "bank": bank,
        "bank_audit": json.loads(b.AUDIT_OUTPUT.read_text()),
        "shards": shards,
    }


@pytest.fixture(autouse=True)
def isolate_parity_comparator(monkeypatch):
    # Actual four-world action/frame checks have their own negative tests.
    # This suite isolates the aggregator's binding and decision algebra.
    def validate(value, **expected):
        if value.get("pass") is not True or any(value.get(k) != v for k, v in expected.items()):
            raise ValueError("parity failed")

    monkeypatch.setattr(a.q1_parity, "validate_runtime_receipt", validate)


def test_go_selects_d_and_reports_contrasts():
    result = a.analyze(_manifest(), 10)
    assert result["validation_errors"] == []
    assert result["verdict"] == "GO_TO_MAC_PHASE_R" and result["selected_recipe"] == "D"
    assert set(result["contrasts"]) == {"B-A", "D-C", "C-A", "D-B", "interaction"}


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "duplicate",
        "checkpoint",
        "early",
        "source",
        "bank",
        "parity",
        "provenance",
        "nonfinite",
    ],
)
def test_receipt_corruption_is_invalid(corruption):
    data = _manifest()
    shard = data["shards"][0]
    if corruption == "missing":
        data["shards"].pop()
    elif corruption == "duplicate":
        data["shards"].append(copy.deepcopy(shard))
    elif corruption == "checkpoint":
        shard["rows"][0]["checkpoint_sha256"] = "0" * 64
    elif corruption == "early":
        shard["rows"][0]["checkpoint_step"] = 500000
    elif corruption == "source":
        shard["identity"]["evaluator_source_manifest"] = {}
    elif corruption == "bank":
        shard["identity"]["bank_sha256"] = "0" * 64
    elif corruption == "parity":
        shard["runtime_parity"]["pass"] = False
    elif corruption == "provenance":
        shard.pop("training_provenance")
    elif corruption == "nonfinite":
        shard["rows"][0]["mass_integral"] = -1
    shard["rows_sha256"] = a._digest(shard["rows"])
    result = a.analyze(data, 10)
    invalid = [
        arm for arm, value in result["arms"].items() if value["verdict"] == "INVALID/INCOMPLETE"
    ]
    assert invalid
    assert result.get("selected_recipe") not in invalid


def test_unbound_rows_cannot_advance():
    data = _manifest()
    assert (
        a.analyze(
            {"bank": data["bank"], "rows": [r for s in data["shards"] for r in s["rows"]]}, 10
        )["verdict"]
        == "INVALID"
    )


def test_small_loss_can_preserve_but_efficacy_kill():
    result = a.analyze(_manifest(-5), 10)
    assert result["verdict"] == "Q2_PARENT"
    assert result["arms"]["D"]["verdict"] == "KILL"
    assert result["arms"]["D"]["preservation_certificate"]


def test_guards_positive_seed_rule_and_strict_boundaries():
    g = {"mean": 25, "lower90": 1, "upper90": 49, "seed_effects": [50, 50, 50, -1, -1]}
    alive = dict.fromkeys(b.MIX_COUNTS, 0)
    nodeath = dict.fromkeys(b.MIX_COUNTS, 0.5)
    assert a._decision(g, 0, alive, nodeath, -0.15, False)[0] == "INCONCLUSIVE"
    g["seed_effects"] = [30] * 5
    assert a._decision(g, 0, alive, nodeath, -0.15, False)[0] == "GO"
    alive["mixed"] = -0.030001
    verdict, _, preserve = a._decision(g, 0, alive, nodeath, -0.15, False)
    assert verdict == "KILL" and not preserve


def test_serialized_evaluator_shards_are_accepted_without_rehashing_file_whitespace(
    tmp_path, monkeypatch
):
    """Run the real writer and analyzer with only world execution replaced."""
    from types import SimpleNamespace

    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    template = _manifest()
    parity = template["shards"][0]["runtime_parity"]
    parity["runtime_artifact"] = {"runtime": a.q1_parity.runtime_environment()}
    parity_path = tmp_path / "parity.json"
    parity_path.write_text(json.dumps(parity, indent=3) + "\n")
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"synthetic B8 carrier")
    original_sha = e.sha256
    monkeypatch.setattr(e, "sha256", lambda path: a.B8_SHA256 if path == b8 else original_sha(path))
    bank = template["bank"]
    monkeypatch.setattr(
        e.t23,
        "_context",
        lambda _: {
            "profile": SimpleNamespace(digest=bank["profile"]["base_profile_digest"]),
            "lookup": {gid.FRP3_S12[1]: ("carrier", "sha")},
        },
    )

    def execute(world, checkpoint, *_):
        return {
            "mix": world["mix"],
            "world_seed": world["world_seed"],
            "world_semantic_sha256": world["world_semantic_sha256"],
            "roster_semantic_sha256": world["roster_semantic_sha256"],
            "mass_integral": 100 if checkpoint == b8 else 125,
            "alive_frame_fraction": 1,
            "no_death_episode": True,
            "horizon": 5000,
            "frames_stepped": 5000,
        }

    monkeypatch.setattr(e.t23, "evaluate_world", execute)
    output = []
    for synthetic in template["shards"]:
        identity = synthetic["identity"]
        arm, seed = identity["arm"], identity["training_seed"]
        provenance_path = None
        checkpoint = b8
        if arm != "B8":
            checkpoint = tmp_path / f"{arm}-{seed}.pth"
            checkpoint.write_bytes(f"synthetic {arm} {seed}".encode())
            provenance = copy.deepcopy(synthetic["training_provenance"])
            provenance["checkpoint_sha256"] = original_sha(checkpoint)
            provenance_path = tmp_path / f"{arm}-{seed}-provenance.json"
            provenance_path.write_text(json.dumps(provenance, indent=4) + "\n")
        path = e.run(
            root=tmp_path,
            bank_path=b.OUTPUT,
            b8=b8,
            checkpoint=checkpoint,
            arm=arm,
            seed=seed,
            shard="0/1",
            # This test substitutes world execution in-process; real spawn
            # identity is covered by the reserved-world A4 capacity probe.
            workers=1,
            provenance_path=provenance_path,
            parity_path=parity_path,
        )
        output.append(json.loads(path.read_text()))
    template["shards"] = output
    result = a.analyze(template, 10)
    assert result["validation_errors"] == []
    assert result["selected_recipe"] == "D"


def test_training_tripwire_kills_only_its_arm():
    data = _manifest()
    data["shards"] = [s for s in data["shards"] if s["identity"]["arm"] != "A"]
    seed = b.TRAINING_SEEDS[0]
    summary = {
        "halt": "action collapse [100000, 0, 0, 0, 0, 0]",
        "args": {"seed": str(seed), "namespace": b.training_namespace("A")},
    }
    outcome = {
        "schema": "q1-training-outcome/v1",
        "arm": "A",
        "training_seed": seed,
        "status": "KILL",
        "reason": summary["halt"],
        "summary": summary,
        "summary_sha256": a._digest(summary),
        "source_sha256": e.sha256(b.ROOT / "research/redesign_m3c_20261008/train_m3c.py"),
    }
    data["training_outcomes"] = [outcome]
    result = a.analyze(data, 10)
    assert result["arms"]["A"]["verdict"] == "KILL"
    assert result["arms"]["D"]["verdict"] == "GO"
    assert result["selected_recipe"] == "D"
    assert result["contrasts"]["B-A"]["status"] == "UNAVAILABLE"
    assert result["contrasts"]["D-C"]["mean"] == 0


def test_incomplete_arm_cannot_replace_missing_seed_with_remaining_four():
    data = _manifest()
    data["shards"] = [
        s
        for s in data["shards"]
        if not (
            s["identity"]["arm"] == "D" and s["identity"]["training_seed"] == b.TRAINING_SEEDS[0]
        )
    ]
    result = a.analyze(data, 10)
    assert result["arms"]["D"]["verdict"] == "INVALID/INCOMPLETE"
    assert result["selected_recipe"] == "C"


def test_operational_incomplete_precedes_tripwire_within_same_arm():
    data = _manifest()
    data["shards"] = [s for s in data["shards"] if s["identity"]["arm"] != "A"]
    outcomes = []
    for seed, halt, status in (
        (b.TRAINING_SEEDS[0], "action collapse [100000, 0]", "KILL"),
        (b.TRAINING_SEEDS[1], "RuntimeError: Drive disconnected", "INCOMPLETE"),
    ):
        summary = {
            "halt": halt,
            "args": {"seed": str(seed), "namespace": b.training_namespace("A")},
        }
        outcomes.append(
            {
                "schema": "q1-training-outcome/v1",
                "arm": "A",
                "training_seed": seed,
                "status": status,
                "reason": halt,
                "summary": summary,
                "summary_sha256": a._digest(summary),
                "source_sha256": e.sha256(b.ROOT / "research/redesign_m3c_20261008/train_m3c.py"),
            }
        )
    data["training_outcomes"] = outcomes
    result = a.analyze(data, 10)
    assert result["arms"]["A"]["verdict"] == "INVALID/INCOMPLETE"
    assert result["selected_recipe"] == "D"


def test_pending_update_debt_cannot_hide_absent_optimizer_work():
    provenance = _manifest()["shards"][0]["training_provenance"]
    provenance["counters"].update(completed_updates=1, pending_update_debt=8_000_000 - 256)
    with pytest.raises(ValueError, match="accounting"):
        e.validate_provenance(provenance, "A", b.TRAINING_SEEDS[0], provenance["checkpoint_sha256"])
