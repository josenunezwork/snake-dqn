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
