"""Protect engineering-only exports, root observers and narrow admission."""

import json
import pickle

import pytest

from research.watch_lead_p_successor import followup
from research.watch_lead_p_successor.host import FOLLOWUP_SCRIPT, require_profile
from research.watch_lead_p_successor.prepare import ROOT


class World:
    """Tiny session with a serializable root, suitable for observer tests."""

    def __init__(self, seed: int) -> None:
        self.seed, self.frame = seed, 0
        self.roster = {"fixture": True}

    @property
    def hero(self):
        from types import SimpleNamespace

        return SimpleNamespace(is_alive=True, length=1000 + self.frame)

    def step(self, forced=None) -> dict:
        self.frame += 1
        return {"frame": self.frame, "mass": 123456, "death": False}

    def snapshot(self) -> bytes:
        assert "step" not in self.__dict__  # observer must not cross root boundary
        return pickle.dumps(self)

    def digest(self) -> str:
        return str((self.seed, self.frame))


@pytest.mark.parametrize("frames", [2, 500])
def test_scan_preserves_real_bare_path_and_strips_observer(monkeypatch, tmp_path, frames):
    monkeypatch.setattr(followup, "LiveWorld", World)
    value = followup.scan(7, frames, scratch=tmp_path)
    assert value["metadata"]["frames"] == frames
    assert value["metadata"]["selected"]["peak"]["frame"] == frames
    assert value["metadata"]["selected"]["late"]["frame"] == frames
    for path in value["paths"].values():
        from pathlib import Path

        restored = pickle.loads(Path(path).read_bytes())
        assert "step" not in restored.__dict__
        restored.step()
        assert restored.frame == frames + 1
    assert not {"mass", "deaths", "roots", "counts"} & value["metadata"].keys()
    assert "123456" not in json.dumps(value["metadata"])


def test_followup_foreign_bank_refuses_before_mkdir(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text("{}")
    with pytest.raises(ValueError, match="foreign"):
        followup.execute(bank, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_followup_adapter_only_admits_exact_family():
    manifest = {
        "p_profile_source_delivery": "background-sparse-v1",
        "script": FOLLOWUP_SCRIPT,
        "device": "cpu",
        "timeout_minutes": 30,
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"]},
        "canary": {"schema": "colab-family-canary/v1", "mode": "probe", "family": FOLLOWUP_SCRIPT},
        "resume_exemption": {"kind": "short-canary"},
        "env": {"SNAKE_COLAB_CANARY": "1"},
    }
    require_profile(manifest, ROOT)
    for change in (
        {"script": "foreign.sh"},
        {"timeout_minutes": 31},
        {
            "canary": {
                "schema": "colab-family-canary/v1",
                "mode": "probe",
                "family": "research/watch_lead_p_successor/profile_job.sh",
            }
        },
    ):
        with pytest.raises(ValueError):
            require_profile({**manifest, **change}, ROOT)
