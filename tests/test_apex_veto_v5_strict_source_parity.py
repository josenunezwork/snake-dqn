"""Source parity between the v5 screen pilot and the v5 strict arms.

The screen's paired B-A variance sizes N, so its A/B arms must have run the same v2/v5/v3
veto bytes the strict intent freezes. ``strict_run.screen_source_parity`` (prepare) and the
independent ``pilot.screen_source_parity`` audit rule compare ``git show <screen
commit>:<module>`` against ``wrapper_identity.source_sha256s``. No test plays an episode
(``block_episode_runners`` is autouse here too).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from research.apex_veto_v5_strict_20261001 import strict_run as sr
from tests import test_apex_veto_v5_strict_audit as helpers
from tests import test_apex_veto_v5_strict_run as run_tests

block_episode_runners = helpers.block_episode_runners  # autouse fixture (re-exported)
small_pilot = helpers.small_pilot  # module fixtures (re-exported)
pass_run = helpers.pass_run
tier2_root = run_tests.tier2_root  # autouse: tmp Tier-2 root
SA = helpers.SA


def git(*args: str) -> str:
    command = ["git", "-C", str(sr.REPO), *args]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


def pre_v5_commit() -> str:
    """Parent of the last commit that touched safety_veto_v5.py (other v5 bytes, or none)."""
    return git("rev-parse", git("rev-list", "-n1", "HEAD", "--", sr.V5_SOURCE) + "^")


def set_screen_git(pilot: Path, commit: str, dirty: str = "", rebind: bool = True) -> None:
    """Rewrite the screen intent's git state; ``rebind`` keeps the receipt bound to it."""
    helpers.write_json(pilot / "intent.json", {"git": {"commit": commit, "dirty_paths": dirty}})
    if rebind:
        helpers.edit_json(
            pilot / "receipt.json",
            lambda r: r.update(intent_sha256=helpers.sha(pilot / "intent.json")),
        )


def copy_of(pilot: Path, tmp_path: Path) -> Path:
    target = tmp_path / "screen"
    shutil.copytree(pilot, target)
    return target


def test_parity_passes_at_the_checked_out_commit(small_pilot: Path) -> None:
    parity = sr.screen_source_parity(small_pilot, sr.arm_identities())
    assert parity["passes"] is True, parity["problems"]
    expected = {rel for arm in sr.ARMS for rel in sr.ARM_SOURCES[arm]}
    assert set(parity["sources"]) == expected
    for row in parity["sources"].values():
        assert row["screen_sha256"] == row["strict_sha256"]


def test_parity_refuses_a_candidate_edited_after_the_screen(small_pilot: Path) -> None:
    identities = json.loads(json.dumps(sr.arm_identities()))
    identities["candidate"]["source_sha256s"][sr.V3_SOURCE] = "0" * 64
    parity = sr.screen_source_parity(small_pilot, identities)
    assert parity["passes"] is False
    assert any(sr.V3_SOURCE in problem for problem in parity["problems"])


@pytest.mark.parametrize(
    "commit, dirty, rebind, fragment",
    [
        (None, "", True, "other v5"),
        ("HEAD", " M src/evaluation/safety_veto_v5.py", True, "dirty tree"),
        ("HEAD", None, True, "dirty tree"),
        ("abc123", "", True, "full commit"),
        ("HEAD", "", False, "intent_sha256"),
    ],
)
def test_parity_refuses_a_screen_from_other_sources(
    small_pilot: Path, tmp_path: Path, commit, dirty, rebind, fragment
) -> None:
    pilot = copy_of(small_pilot, tmp_path)
    resolved = pre_v5_commit() if commit is None else commit
    resolved = git("rev-parse", "HEAD") if resolved == "HEAD" else resolved
    set_screen_git(pilot, resolved, dirty, rebind)
    if not rebind:
        helpers.write_json(pilot / "intent.json", {"git": {"commit": resolved, "x": 1}})
    parity = sr.screen_source_parity(pilot, sr.arm_identities())
    assert parity["passes"] is False
    if fragment == "other v5":
        assert any(sr.V5_SOURCE in p for p in parity["problems"]), parity["problems"]
    else:
        assert any(fragment in p for p in parity["problems"]), parity["problems"]


def test_parity_refuses_a_missing_screen_intent(small_pilot: Path, tmp_path: Path) -> None:
    pilot = copy_of(small_pilot, tmp_path)
    (pilot / "intent.json").unlink()
    parity = sr.screen_source_parity(pilot, sr.arm_identities())
    assert parity["passes"] is False
    assert "screen intent.json missing" in parity["problems"]


# ---------------------------------------------------------------- prepare and validate_intent


@run_tests.NEEDS_REAL
def test_intent_records_parity_and_prepare_refuses_other_sources(
    small_pilot: Path, tmp_path: Path
) -> None:
    intent = run_tests.real_intent(tmp_path, small_pilot)
    parity = intent["pilot"]["screen_source_parity"]
    assert parity["passes"] is True and parity["screen_commit"] == git("rev-parse", "HEAD")
    sr.validate_intent(intent)
    pilot = copy_of(small_pilot, tmp_path)
    set_screen_git(pilot, pre_v5_commit())
    with pytest.raises(sr.StrictRunError, match="v5 screen ran other veto sources"):
        run_tests.real_intent(tmp_path, pilot)


@run_tests.NEEDS_REAL
@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(passes=False),
        lambda p: p["sources"].pop(sr.V3_SOURCE),
        lambda p: p["sources"][sr.V5_SOURCE].update(screen_sha256="0" * 64),
        lambda p: p.clear(),
    ],
)
def test_validate_intent_rejects_a_tampered_parity_record(
    small_pilot: Path, tmp_path: Path, mutate
) -> None:
    intent = run_tests.real_intent(tmp_path, small_pilot)
    mutate(intent["pilot"]["screen_source_parity"])
    with pytest.raises(sr.StrictRunError, match="screen_source_parity"):
        sr.validate_intent(intent)


# ---------------------------------------------------------------- independent audit


def parity_rule(audit: dict) -> dict:
    return next(row for row in audit["rules"] if row["rule"] == "pilot.screen_source_parity")


def test_audit_parity_rule_passes_on_a_bound_screen(pass_run: Path, small_pilot: Path) -> None:
    audit = helpers.run_inline(pass_run, small_pilot)
    assert parity_rule(audit)["ok"] is True, parity_rule(audit)
    assert "pilot.screen_source_parity" not in helpers.failed_rules(audit)


@pytest.mark.parametrize("tamper", ["pre_v5_commit", "dirty", "unbound_intent"])
def test_audit_detects_a_screen_from_other_sources(
    pass_run: Path, small_pilot: Path, tmp_path: Path, tamper: str
) -> None:
    pilot = copy_of(small_pilot, tmp_path)
    head = git("rev-parse", "HEAD")
    if tamper == "pre_v5_commit":
        set_screen_git(pilot, pre_v5_commit())
    elif tamper == "dirty":
        set_screen_git(pilot, head, " M src/evaluation/safety_veto_v3.py")
    else:
        set_screen_git(pilot, head, rebind=False)
        helpers.write_json(
            pilot / "intent.json", {"git": {"commit": head, "dirty_paths": ""}, "x": 1}
        )
    audit = helpers.run_inline(helpers.clone(pass_run, tmp_path), pilot)
    assert "pilot.screen_source_parity" in helpers.failed_rules(audit)
    assert audit["status"] == "FAIL"


def test_audit_detects_a_recorded_parity_that_disagrees(
    pass_run: Path, small_pilot: Path, tmp_path: Path
) -> None:
    run = helpers.clone(pass_run, tmp_path)

    def forge(intent: dict) -> None:
        intent["candidate"]["wrapper_identity"]["source_sha256s"][sr.V5_SOURCE] = "0" * 64
        intent["pilot"]["screen_source_parity"]["sources"][sr.V5_SOURCE]["strict_sha256"] = "0" * 64

    helpers.edit_json(run / "intent.json", forge)
    problems = SA.screen_source_problems(small_pilot, json.loads((run / "intent.json").read_text()))
    assert any(sr.V5_SOURCE in p and "strict" in p for p in problems), problems
    assert "pilot.screen_source_parity" in helpers.failed_rules(
        helpers.run_inline(run, small_pilot)
    )


def test_audit_parity_without_a_screen_intent(small_pilot: Path, tmp_path: Path) -> None:
    pilot = copy_of(small_pilot, tmp_path)
    (pilot / "intent.json").unlink()
    assert SA.screen_source_problems(pilot, {}) == ["screen intent.json or receipt.json missing"]
