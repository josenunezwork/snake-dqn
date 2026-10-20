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
    assert config["warmup"] == 1


def test_bank_seed_recipe_is_fixed_and_unique():
    from research.t34_pathb.bank import _candidate_seeds

    assert _candidate_seeds("t34-test", 3) == _candidate_seeds("t34-test", 3)
    assert len(set(_candidate_seeds("t34-test", 100))) == 100


def test_large_death_uses_fatal_length_not_previous_length(monkeypatch):
    import io
    from types import SimpleNamespace

    from research.t34_pathb import run

    hero = SimpleNamespace(id=0, length=799, is_alive=True, _pre_collision_action=1)
    game = SimpleNamespace(frame=0, snakes=[hero], frame_death_causes={})

    def update():
        hero.length = 800
        hero.is_alive = False

    def step():
        game.frame += 1
        hero.update()
        game.frame_death_causes = {0: "collision"}

    hero.update = update
    session = SimpleNamespace(game=game, step=step, snapshot=lambda: {})
    monkeypatch.setattr(run.time, "sleep", lambda _: None)
    metric = run._metrics(session, 1, 12, None, io.StringIO(), float("inf"))
    assert metric["deaths_length_ge_800"] == 1
    assert metric["deaths"] == metric["decisions"] == 1
