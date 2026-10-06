"""Audit rules of the frp3-s12 checkpoint-swap web-serving run, tested on REAL record shapes.

Positive tests use the producer's dry-run smoke (fixtures/smoke-v1, unmodified: served under
the in-process placeholder pin). Every negative test mutates a copy. The real-run strict-pin
binding is tested against a constructed strict root and a temporary pins file. Nothing here
plays an episode.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from research.frp3_checkpoint_serving_20261005 import serving_audit as audit_mod

HERE = Path(__file__).resolve().parents[1] / "research" / "frp3_checkpoint_serving_20261005"
SMOKE = HERE / "fixtures" / "smoke-v1"
WATCH = "records/watch_hero-000.json"
PLAY = "records/play-000.json"
PARITY = "parity/parity-000.json"
DEFAULT = "default_check.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def run_copy(tmp_path):
    root = tmp_path / "run"
    shutil.copytree(SMOKE, root)
    return root


def mutate(root: Path, rel: str, change, rehash: bool = True) -> None:
    """Apply ``change`` to one JSON file and (by default) re-hash it in the receipt."""
    path = root / rel
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    if rehash and rel != "receipt.json":
        receipt = json.loads((root / "receipt.json").read_text())
        receipt["files"][rel] = _sha(path)
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))


def rebind_intent(root: Path, change) -> None:
    """Mutate intent.json and re-bind the receipt (records keep the old intent sha)."""
    mutate(root, "intent.json", change, rehash=False)
    sha = _sha(root / "intent.json")
    mutate(root, "receipt.json", lambda d: d.update(intent_sha256=sha))


def failing(result) -> set:
    return {name for name, c in result["criteria"].items() if not c["pass"]}


def failures(result, criterion) -> list:
    return result["criteria"][criterion]["failures"]


class TestPositive:
    def test_fixture_provenance(self):
        provenance = json.loads((HERE / "fixtures" / "provenance.json").read_text())
        assert "smoke-v1" in provenance

    def test_smoke_passes_every_criterion_and_never_qualifies(self):
        result = audit_mod.audit(SMOKE)
        assert result["verdict"] == "SERVING_PASS", result["criteria"]
        assert result["serving_path_qualified"] is False
        assert result["reported"]["parity"]["label"] in (
            "head-exercised",
            "rerank-exercised",
            "exercised",
            "non-exercising",
        )
        assert result["reported"]["served_checkpoint_paths"]

    def test_smoke_records_are_the_swap(self):
        intent = json.loads((SMOKE / "intent.json").read_text())
        assert intent["smoke"] is True and intent["strict_pin_placeholder_for_smoke"] is True
        watch = json.loads((SMOKE / WATCH).read_text())
        assert watch["served_checkpoint"]["name"] == "frp3-s12"
        assert watch["checkpoint"]["session_sha256"] == audit_mod.CHECKPOINT_SHA256
        assert watch["veto"]["variant"] == "v8" and watch["veto"]["active"] is True
        parity = json.loads((SMOKE / PARITY).read_text())
        assert parity["simd"]["provenance"] == audit_mod.SIMD_PROVENANCE

    def test_isolated_cli_is_stdlib_only(self, tmp_path):
        out = tmp_path / "audit"
        script = HERE / "serving_audit.py"
        proc = subprocess.run(
            [sys.executable, "-I", str(script), "--root", str(SMOKE), "--out", str(out)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert json.loads((out / "audit.json").read_text())["verdict"] == "SERVING_PASS"


class TestInvalidAndS1:
    def test_missing_intent_is_invalid(self, run_copy):
        (run_copy / "intent.json").unlink()
        assert audit_mod.audit(run_copy)["verdict"] == "INVALID"

    def test_hash_mismatch_fails_s1(self, run_copy):
        mutate(run_copy, WATCH, lambda d: d.update(frames_stepped=499), rehash=False)
        assert "S1" in failing(audit_mod.audit(run_copy))

    def test_wrong_seed_fails_s1(self, run_copy):
        mutate(run_copy, WATCH, lambda d: d.update(world_seed=1))
        assert "S1" in failing(audit_mod.audit(run_copy))

    def test_relabelled_real_run_fails_preregistration(self, run_copy):
        rebind_intent(run_copy, lambda d: d.update(smoke=False))
        result = audit_mod.audit(run_copy)
        assert "S1" in failing(result) and result["serving_path_qualified"] is False
        assert any("strict run's seeds" in f for f in failures(result, "S1"))


class TestS2Identity:
    @pytest.mark.parametrize(
        "change,needle",
        [
            (lambda d: d["served_checkpoint"].update(name="champion"), "served_checkpoint.name"),
            (lambda d: d["served_checkpoint"].update(reason="refused"), "served_checkpoint.reason"),
            (lambda d: d["served_checkpoint"].update(path="/tmp/x.pth"), "served_checkpoint.path"),
            (
                lambda d: d["checkpoint"].update(session_sha256="0" * 64),
                "checkpoint sha (file/session)",
            ),
            (lambda d: d["env"].update(SNAKE_SERVE_CHECKPOINT=None), "release env"),
            (lambda d: d["wrapper"].update(method="free-space-veto/v7"), "wrapper.method"),
            (lambda d: d["served"].update(epsilon=0.1), "served.epsilon"),
        ],
    )
    def test_watch_record_mutations(self, run_copy, change, needle):
        mutate(run_copy, WATCH, change)
        result = audit_mod.audit(run_copy)
        assert "S2" in failing(result)
        assert any(needle in f for f in failures(result, "S2")), failures(result, "S2")

    def test_play_with_a_wrapper_fails(self, run_copy):
        watch = json.loads((run_copy / WATCH).read_text())
        mutate(run_copy, PLAY, lambda d: d.update(wrapper=watch["wrapper"]))
        assert "S2" in failing(audit_mod.audit(run_copy))

    def test_smoke_without_the_placeholder_flag_fails(self, run_copy):
        rebind_intent(run_copy, lambda d: d.update(strict_pin_placeholder_for_smoke=False))
        result = audit_mod.audit(run_copy)
        assert any("placeholder pin" in f for f in failures(result, "S2"))


class TestStrictPinBinding:
    """Real-run S2: the intent's pin is the repo's filled pin and the strict files verify."""

    def strict_root(self, tmp_path, candidate_sha=None, status="PASS"):
        root = tmp_path / "strict"
        (root / "output" / "audit").mkdir(parents=True)
        intent = root / "intent.json"
        intent.write_text(
            json.dumps(
                {
                    "arms": {
                        "candidate": {
                            "checkpoint_sha256": candidate_sha or audit_mod.CHECKPOINT_SHA256,
                            "method": audit_mod.V8_METHOD,
                        }
                    }
                }
            )
        )
        report = root / "output" / "audit" / "audit.json"
        report.write_text(
            json.dumps({"status": status, "recomputed": {"expected_outcome": "STRICT_PASS"}})
        )
        (root / "output" / "receipt.json").write_text(
            json.dumps({"intent_sha256": _sha(intent), "audit_report_sha256": _sha(report)})
        )
        return root

    def pin(self, root: Path) -> dict:
        files = {
            "receipt": "output/receipt.json",
            "intent": "intent.json",
            "audit_report": "output/audit/audit.json",
        }
        receipt = {"verdict": "STRICT_PASS", "root": str(root)}
        for key, rel in files.items():
            receipt[key] = {"path": rel, "sha256": _sha(root / rel)}
        return {
            "checkpoint_sha256": audit_mod.CHECKPOINT_SHA256,
            "veto_variant": "v8",
            "veto_method": audit_mod.V8_METHOD,
            "strict_receipt": receipt,
        }

    def pins_file(self, tmp_path, entry) -> Path:
        doc = json.loads(audit_mod.PINS_FILE.read_text())
        doc["checkpoints"]["frp3-s12"] = entry
        path = tmp_path / "pins.json"
        path.write_text(json.dumps(doc))
        return path

    def intent(self, pin) -> dict:
        intent = json.loads((SMOKE / "intent.json").read_text())
        intent.update(smoke=False, strict_pin=pin, strict_pin_placeholder_for_smoke=False)
        return intent

    def test_filled_and_verified_pin_passes(self, tmp_path):
        root = self.strict_root(tmp_path)
        pin = self.pin(root)
        pins = self.pins_file(tmp_path, pin)
        assert audit_mod.strict_pin_failures(self.intent(pin), False, pins) == []

    def test_unfilled_pins_file_fails_a_real_run(self, tmp_path):
        pin = self.pin(self.strict_root(tmp_path))
        placeholder = copy.deepcopy(pin)
        placeholder["strict_receipt"] = None
        pins = self.pins_file(tmp_path, placeholder)
        out = audit_mod.strict_pin_failures(self.intent(pin), False, pins)
        assert "intent.strict_pin differs from the repo pins file" in out

    def test_tampered_strict_file_fails(self, tmp_path):
        root = self.strict_root(tmp_path)
        pin = self.pin(root)
        pins = self.pins_file(tmp_path, pin)
        (root / "output" / "receipt.json").write_text('{"x": 1}')
        out = audit_mod.strict_pin_failures(self.intent(pin), False, pins)
        assert "strict receipt sha256 differs from the pin" in out

    def test_strict_run_for_another_checkpoint_fails(self, tmp_path):
        root = self.strict_root(tmp_path, candidate_sha=audit_mod.CHAMPION_SHA256)
        pin = self.pin(root)
        pins = self.pins_file(tmp_path, pin)
        out = audit_mod.strict_pin_failures(self.intent(pin), False, pins)
        assert any("candidate.checkpoint_sha256" in f for f in out), out

    def test_failed_strict_audit_fails(self, tmp_path):
        root = self.strict_root(tmp_path, status="FAIL")
        pin = self.pin(root)
        pins = self.pins_file(tmp_path, pin)
        out = audit_mod.strict_pin_failures(self.intent(pin), False, pins)
        assert any("status is not PASS" in f for f in out), out

    def test_missing_strict_file_and_placeholder_root_fail(self, tmp_path):
        root = self.strict_root(tmp_path)
        pin = self.pin(root)
        pins = self.pins_file(tmp_path, pin)
        (root / "output" / "audit" / "audit.json").unlink()
        out = audit_mod.strict_pin_failures(self.intent(pin), False, pins)
        assert any("strict audit_report missing" in f for f in out)
        smoke_like = copy.deepcopy(pin)
        smoke_like["strict_receipt"]["root"] = audit_mod.SMOKE_PIN_ROOT
        pins = self.pins_file(tmp_path, smoke_like)
        out = audit_mod.strict_pin_failures(self.intent(smoke_like), False, pins)
        assert "strict_receipt.root is not a real absolute path" in out

    def test_real_run_under_the_smoke_placeholder_fails(self, tmp_path):
        root = self.strict_root(tmp_path)
        pin = self.pin(root)
        pins = self.pins_file(tmp_path, pin)
        intent = self.intent(pin)
        intent["strict_pin_placeholder_for_smoke"] = True
        out = audit_mod.strict_pin_failures(intent, False, pins)
        assert "a real run served under the smoke placeholder pin" in out


class TestS3S4:
    def test_inactive_watch_fails_s3(self, run_copy):
        mutate(run_copy, WATCH, lambda d: d["veto"].update(active=False))
        assert "S3" in failing(audit_mod.audit(run_copy))

    def test_unmatched_checkpoint_fails_s3(self, run_copy):
        mutate(run_copy, WATCH, lambda d: d["veto"].update(strict_receipt_checkpoint_match=False))
        assert "S3" in failing(audit_mod.audit(run_copy))

    def test_wrapped_play_fails_s3(self, run_copy):
        mutate(run_copy, PLAY, lambda d: d.update(snakes_with_veto=[1]))
        assert "S3" in failing(audit_mod.audit(run_copy))

    def test_counter_partition_break_fails_s4(self, run_copy):
        def change(d):
            sid = str(d["served"]["hero_id"])
            d["veto"]["per_snake"][sid]["kept_base"] += 1

        mutate(run_copy, WATCH, change)
        assert "S4" in failing(audit_mod.audit(run_copy))


class TestS5Parity:
    @pytest.mark.parametrize(
        "change,needle",
        [
            (lambda d: d.update(first_divergence_frame=3), "first_divergence_frame"),
            (lambda d: d.update(trace_sha256_equal=False), "trace_sha256"),
            (lambda d: d["simd"]["provenance"].update(forward="batched"), "simd_provenance"),
            (lambda d: d["simd"].update(wrapper_method="x"), "simd_method"),
            (lambda d: d["session"].update(checkpoint_sha256="0" * 64), "session_checkpoint"),
            (lambda d: d["session"]["served_checkpoint"].update(name="champion"), "session_served"),
            (lambda d: d["simd"].update(record_complete=False), "record_complete"),
            (lambda d: d["session"].update(wrapped_snake_ids=[0, 1]), "wrapped_hero_only"),
            (lambda d: d.update(divergence={"x": 1}), "first_divergence_frame"),
        ],
    )
    def test_probe_mutations(self, run_copy, change, needle):
        mutate(run_copy, PARITY, change)
        result = audit_mod.audit(run_copy)
        assert "S5" in failing(result)
        assert any(needle in f for f in failures(result, "S5")), failures(result, "S5")

    def test_unequal_diagnostics_fail(self, run_copy):
        def change(d):
            d["simd"]["diagnostics"]["head_checks"] += 1

        mutate(run_copy, PARITY, change)
        assert "S5" in failing(audit_mod.audit(run_copy))

    def test_real_run_needs_the_pinned_profile(self, run_copy):
        ev = audit_mod.load_evidence(run_copy)
        probe = ev["docs"][PARITY]
        probe["simd"]["profile_scored_horizon"] = 200
        design = dict(audit_mod.DESIGN[True])
        out = audit_mod.s5_parity(ev, design, smoke=False)
        assert any("simd_profile" in f for f in out)


class TestS6Defaults:
    @pytest.mark.parametrize(
        "change,needle",
        [
            (lambda d: d.update(variant_from_empty_env="v7"), "released v8"),
            (lambda d: d.update(checkpoint_released_default="frp3-s12"), "champion"),
            (
                lambda d: d["builds"]["frp3_pin_unfilled"]["served_checkpoint"].update(
                    name="frp3-s12"
                ),
                "frp3_pin_unfilled: served",
            ),
            (
                lambda d: d["builds"]["checkpoint_unknown"]["served_checkpoint"].update(
                    reason=None
                ),
                "checkpoint_unknown: served_checkpoint.reason",
            ),
            (
                lambda d: d["builds"]["frp3_rollback_variant_v7"]["modes"]["watch"].update(
                    active=True
                ),
                "frp3_rollback_variant_v7: Watch is wrapped",
            ),
            (
                lambda d: d["builds"]["empty_env"]["modes"]["play"].update(active=True),
                "empty_env: Play is wrapped",
            ),
            (
                lambda d: d["builds"]["frp3_watch_off"].update(pins="unfilled"),
                "frp3_watch_off: pins",
            ),
            (lambda d: d["builds"].pop("frp3_watch_off"), "default_check builds"),
        ],
    )
    def test_default_check_mutations(self, run_copy, change, needle):
        mutate(run_copy, DEFAULT, change)
        result = audit_mod.audit(run_copy)
        assert "S6" in failing(result)
        assert any(needle in f for f in failures(result, "S6")), failures(result, "S6")
