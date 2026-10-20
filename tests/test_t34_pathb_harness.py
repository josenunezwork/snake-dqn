from __future__ import annotations

import json

from research.t34_pathb.analyze import analyze


def test_incomplete_bank_fails_closed(tmp_path):
    (tmp_path / "records").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"expected_records": 1}))
    report = analyze(tmp_path)
    assert report["decision"] == "INCOMPLETE"
    assert report["client_latency"] == "NOT_MEASURED"


def test_smoke_config_is_thread_bounded():
    from research.t34_pathb import run

    config = json.loads((run.HERE / "smoke.json").read_text())
    assert config["workers"] == 1
    assert config["horizon"] < 5000
    assert config["warmup"] == 0


def test_bank_seed_recipe_is_fixed_and_unique():
    from research.t34_pathb.bank import _candidate_seeds

    assert _candidate_seeds("t34-test", 3) == _candidate_seeds("t34-test", 3)
    assert len(set(_candidate_seeds("t34-test", 100))) == 100
