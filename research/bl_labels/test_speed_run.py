"""Light isolated bootstrap and pre-data evidence contracts; no gameplay."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from research.bl_labels import speed_equiv as eq
from research.bl_labels import speed_run
from research.bl_labels.exports import _verify, publish_json


def test_worker_isolates_science_and_keeps_callback_exports_distinct(tmp_path: Path) -> None:
    root = tmp_path / "frozen"
    files = {
        "research/bl_labels/factory.py": """
from research.bl_served_search.b2_engine import WatchDriver
make_driver=lambda *args: WatchDriver()
def collect_unit(item,b8,frp3,namespace):
 d=make_driver(); d.prepared_decision(); d.step(packet=object())
 base={'namespace':namespace,'allocation_id':item['allocation_id']}
 base['allocation']={k:v for k,v in item.items() if not callable(v)}
 item['publish_life'](dict(base,schema='bl-labels-f0-life/v1',life=0,rows=[]))
 d=WatchDriver(); d.life=1; d.prepared_decision(); d.step(0)
 item['publish_life'](dict(base,schema='bl-labels-f0-session/v1',rows=[]))
 return dict(base,schema='bl-labels-f0-unit/v1',
             rows=[],timing_seconds={'play':1.})
""",
        "research/bl_served_search/b2_engine.py": """
from src.simd_env import ego2s_policy,ego_raster
class Mask:
 def __getitem__(self,key): return self
 def astype(self,kind): return self
 def tolist(self): return [True]*6
class Sim:
 def get_advisory_action_mask(self): return Mask()
class WatchDriver:
 def __init__(self): self.sim=Sim(); self.life=0; self.decision=0
 def prepared_decision(self):
  ego_raster.build_ego_raster(self.sim); ego2s_policy.forward_q(None); return object()
 def step(self,forced=None,packet=None):
  if forced is None and packet is None: ego2s_policy.forward_q(None)
  self.decision+=1
""",
        "src/simd_env/ego2s_policy.py": """from .ego_raster import build_ego_raster
def forward_q(*args): return None
""",
        "src/simd_env/ego_raster.py": "def build_ego_raster(*args,**kwargs): return {}\n",
    }
    # Use the real immutable export code, copied by bytes into the frozen tree.
    files["research/bl_labels/exports.py"] = Path(eq.__file__).with_name("exports.py").read_text()
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        parent = path.parent
        while parent != root:
            (parent / "__init__.py").touch()
            parent = parent.parent
    manifest = tmp_path / "bank.json"
    publish_json(
        manifest, {"namespace": eq.NAMESPACE, "allocation": [{"allocation_id": "p", "seed": 1}]}
    )
    output = tmp_path / "output"
    command = [
        sys.executable,
        str(speed_run.WORKER),
        "--worktree",
        str(root),
        "--pin",
        eq.OLD_PIN,
        "--output",
        str(output),
        "--b8",
        str(tmp_path / "b8"),
        "--frp3",
        str(tmp_path / "frp3"),
        "--manifest",
        str(manifest),
        "--index",
        "0",
        "--deadline",
        str(time.monotonic() + 30),
        "--launched",
        str(time.monotonic()),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, env=dict(os.environ), check=False
    )
    assert result.returncode == 0, result.stderr
    diagnostic = json.loads((output / "diagnostic.json").read_text())
    assert diagnostic["counts"] == {
        "living_decisions": 2,
        "natural_rasters": 2,
        "natural_forwards": 2,
        "forced_forwards": 0,
        "forced_steps": 1,
    }
    assert set(diagnostic["advisory"]) == {"0:0", "1:0"}
    paths = list((output / "payloads").glob("*.json"))
    assert len(paths) == 3
    assert {json.loads(p.read_text())["schema"] for p in paths} == {
        "bl-labels-f0-life/v1",
        "bl-labels-f0-session/v1",
        "bl-labels-f0-unit/v1",
    }
    for path in paths:
        _verify(path)


def test_publication_gate_precedes_any_science_read(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    publish_json(output / "phase-ledger.json", {"opaque": "metadata"})
    expected = {
        "ledger": str((output / "phase-ledger.json").resolve()),
        "sha256": _verify(output / "phase-ledger.json"),
    }
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: json.dumps(expected).encode())
    assert speed_run.verify_publication(output, tmp_path, "ledger.json") == {"opaque": "metadata"}
    expected["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="published unchanged"):
        speed_run.verify_publication(output, tmp_path, "ledger.json")


def test_f1_requires_all_schedules_unique_roots_and_pins(tmp_path: Path, monkeypatch) -> None:
    from research.bl_labels import f1_ruler

    roots = [{"root_id": f"r{i}"} for i in range(64)]
    monkeypatch.setattr(f1_ruler, "bank_digest", lambda p: eq.F1_BANK_SHA256)
    monkeypatch.setattr(f1_ruler, "load_bank", lambda p: {"roots": roots})

    def result(pin):
        rows = [
            dict(
                root_id=r["root_id"],
                label_sha256="a" * 64,
                digest="b" * 64,
                state_digest="c" * 64,
                digest_checks=1,
                bare_audit_mismatches=0,
            )
            for r in roots
        ]
        return dict(
            status="COMPLETE",
            source_commit=pin,
            bank_sha256=eq.F1_BANK_SHA256,
            configs=[
                dict(
                    schedule_label=s,
                    records=list(rows),
                    source_commit=pin,
                    bank_sha256=eq.F1_BANK_SHA256,
                )
                for s in ("one", "half", "all")
            ],
        )

    old, new = tmp_path / "old.json", tmp_path / "new.json"
    publish_json(old, result(eq.OLD_PIN))
    publish_json(new, result("d" * 40))
    assert eq.f1_identity(old, new, expected_new_sha="d" * 40)["status"] == "PASS"
    malformed = result("d" * 40)
    malformed["configs"][1]["records"][1] = malformed["configs"][1]["records"][0]
    duplicate = tmp_path / "duplicate.json"
    publish_json(duplicate, malformed)
    with pytest.raises(ValueError, match="duplicate/missing"):
        eq.f1_identity(old, duplicate, expected_new_sha="d" * 40)


def test_phase_with_diagnostics_but_no_science_cannot_pass(tmp_path: Path) -> None:
    output = tmp_path / "run"
    cell = output / "new-w1"
    bank = tmp_path / "bank.json"
    allocation = [{"allocation_id": f"p{i}"} for i in range(12)]
    publish_json(bank, {"allocation": allocation})
    paths = []
    for i, row in enumerate(allocation):
        diagnostic = cell / f"parent-{i:02d}/diagnostic.json"
        publish_json(diagnostic, dict(allocation_id=row["allocation_id"], pin="d" * 40))
        paths.append(diagnostic)
    phase = cell / "phase-ledger.json"
    publish_json(
        phase,
        dict(
            pin="d" * 40,
            workers=1,
            parent_indexes=list(range(12)),
            files=speed_run.metadata(paths, cell),
        ),
    )
    ledger = dict(
        old_pin=eq.OLD_PIN,
        new_pin="d" * 40,
        bank_path=str(bank),
        phases={"new-w1": {"path": str(phase), "sha256": _verify(phase)}},
    )
    with pytest.raises(ValueError, match="missing science unit/session"):
        speed_run.load_phase(output, "new-w1", ledger)


def test_header_only_pass_receipt_is_rejected(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    publish_json(
        receipt,
        dict(
            schema=eq.SCHEMA,
            status="PASS",
            old={"commit": eq.OLD_PIN},
            new={"commit": "d" * 40},
            f1_bank_sha256=eq.F1_BANK_SHA256,
        ),
    )
    with pytest.raises(ValueError, match="ruling/call-counter"):
        eq.verify_receipt(receipt, eq.OLD_PIN, "d" * 40, eq.F1_BANK_SHA256)
