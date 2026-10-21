"""Round 0 boost-floor amendment: rule, unchanged clauses, and one-look mechanics.

All records are synthetic. Git tests use a temporary bare remote only; nothing
here reads the real study root.
"""

from __future__ import annotations

import copy
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from research.t34_pathb.retention_history_v2 import _registration_sha
from research.t34_round0 import history
from research.t34_round0.analyze import COUNTS, summarize
from research.t34_round0.common import ROOT, read_json, sha256
from research.t34_round0_amend import decide
from research.t34_round0_amend.rule import (
    CLAUSE,
    REGISTERED_CLAUSES,
    amend,
    amended_boost_clause,
    pooled_boosts,
)

D = 5000  # decisions per surviving world
SEEDS = 120


def _arm(alive: bool, survival: float, mass: float, decisions: int, boosts: int) -> dict:
    counts = [decisions - boosts, 0, 0, boosts, 0, 0]
    return dict(
        mass_integral=mass,
        survival_fraction=1.0 if alive else survival,
        no_death=int(alive),
        deaths=int(not alive),
        decisions=decisions,
        boost_decisions=boosts,
        action_counts=counts,
        unconditional_faults=[],
    )


def passing_records(teacher_boosts, bare_boosts) -> list[dict]:
    """Every non-boost bar passes; boosts are callables of the world index."""
    rows = []
    for mix, n in COUNTS.items():
        for i in range(n):
            index = len(rows)
            arms = {
                "bare": _arm(i < int(n * 0.5), 0.5, 100.0 + i % 7, D, bare_boosts(index)),
                "teacher": _arm(i < int(n * 0.8), 0.6, 150.0 + i % 5, D, teacher_boosts(index)),
            }
            rows.append(
                dict(
                    world=dict(
                        world_id=f"world-{index}", world_seed=1000 + index, mix=mix, horizon=5000
                    ),
                    arms=arms,
                )
            )
    return rows


def _boosts_for(share_per_million: int):
    """Integer boosts per world whose pooled share over 240 x D is exact."""
    total = share_per_million * 240 * D // 1_000_000
    return lambda index: total // 240 + (1 if index < total % 240 else 0)


def _amended(rows: list[dict]) -> dict:
    registered = json.loads(json.dumps(summarize(rows)))
    return amend(registered, pooled_boosts(rows, "teacher"), pooled_boosts(rows, "bare"))


# ----------------------------------------------------------------- the rule ---


def test_bare_012_teacher_013_passes_where_registered_floor_stops() -> None:
    rows = passing_records(_boosts_for(1300), _boosts_for(1200))
    registered = summarize(rows)
    assert registered["clauses"][CLAUSE] is False and registered["decision"] == "STOP"
    result = _amended(rows)
    assert result["clauses"][CLAUSE] is True
    assert result["decision"] == "GO"


def test_teacher_005_below_half_of_bare_012_fails() -> None:
    rows = passing_records(_boosts_for(500), _boosts_for(1200))
    result = _amended(rows)
    assert result["clauses"][CLAUSE] is False
    assert result["decision"] == "STOP"


def test_teacher_11_percent_fails_unchanged_ceiling() -> None:
    rows = passing_records(_boosts_for(110_000), _boosts_for(1200))
    assert summarize(rows)["clauses"][CLAUSE] is False
    result = _amended(rows)
    assert result["clauses"][CLAUSE] is False
    assert result["decision"] == "STOP"


def test_exact_half_is_inclusive_and_one_fewer_fails() -> None:
    assert amended_boost_clause(600, 1_000_000, 1200, 1_000_000)
    assert not amended_boost_clause(599, 1_000_000, 1200, 1_000_000)
    # Different denominators: floor compares shares, not counts.
    assert amended_boost_clause(3, 5000, 12, 10000)
    assert not amended_boost_clause(2, 5000, 12, 10000)


def test_ceiling_is_inclusive_and_bare_zero_means_zero_floor() -> None:
    assert amended_boost_clause(100, 1000, 0, 1000)
    assert not amended_boost_clause(101, 1000, 0, 1000)
    assert amended_boost_clause(0, 1000, 0, 1000)
    rows = passing_records(lambda _i: 0, lambda _i: 0)
    assert summarize(rows)["clauses"][CLAUSE] is False
    assert _amended(rows)["decision"] == "GO"


def test_teacher_counts_must_match_registered_share() -> None:
    rows = passing_records(_boosts_for(1300), _boosts_for(1200))
    registered = json.loads(json.dumps(summarize(rows)))
    with pytest.raises(ValueError, match="registered share"):
        amend(registered, (1, 10), pooled_boosts(rows, "bare"))


def _random_records(rng: random.Random) -> list[dict]:
    mass_shift = {m: rng.uniform(20, 110) for m in COUNTS}
    noise = rng.choice((40, 300, 600))
    bare_alive = rng.uniform(0.45, 0.65)
    alive_rate = {"bare": bare_alive, "teacher": min(0.98, bare_alive + rng.uniform(0.0, 0.45))}
    dead_survival = {"bare": rng.uniform(0.2, 0.8), "teacher": rng.uniform(0.2, 0.8)}
    rates = (0.0, 0.0005, 0.0012, 0.003, 0.006, 0.05, 0.15)
    boost_rate = {"bare": rng.choice(rates), "teacher": rng.choice(rates)}
    fault_world = rng.randrange(240) if rng.random() < 0.1 else None
    rows = []
    for mix, n in COUNTS.items():
        for _ in range(n):
            index = len(rows)
            arms = {}
            for name in ("bare", "teacher"):
                alive = rng.random() < alive_rate[name]
                survival = (
                    1.0 if alive else min(0.99, max(0.01, rng.gauss(dead_survival[name], 0.2)))
                )
                decisions = D if alive else max(1, int(survival * D))
                boosts = min(decisions, round(decisions * boost_rate[name] * rng.uniform(0.5, 1.5)))
                mass = max(
                    0.0, rng.gauss(120 + (mass_shift[mix] if name == "teacher" else 0), noise)
                )
                arms[name] = _arm(alive, survival, mass, decisions, boosts)
            if index == fault_world:
                arms["teacher"]["unconditional_faults"] = ["nonfinite_q"]
            rows.append(
                dict(
                    world=dict(
                        world_id=f"world-{index}", world_seed=50 + index, mix=mix, horizon=5000
                    ),
                    arms=arms,
                )
            )
    return rows


def _with_registered_boost_clause(rows: list[dict], value: bool) -> list[dict]:
    """Change only teacher boost counts so the REGISTERED clause takes ``value``.

    Boost counts enter no other clause, so the registered summarize on these rows
    is an independent oracle for the amended decision order.
    """
    rows = copy.deepcopy(rows)
    for row in rows:
        arm = row["arms"]["teacher"]
        boosts = arm["decisions"] // 50 if value else 0  # 2% (in band) or 0% (below 0.5%)
        arm["boost_decisions"] = boosts
        arm["action_counts"] = [arm["decisions"] - boosts, 0, 0, boosts, 0, 0]
    return rows


@pytest.mark.parametrize("seed", range(SEEDS))
def test_every_other_clause_and_the_order_match_registered_summarize(seed: int) -> None:
    rng = random.Random(seed)
    rows = _random_records(rng)
    registered = json.loads(json.dumps(summarize(rows)))
    result = amend(registered, pooled_boosts(rows, "teacher"), pooled_boosts(rows, "bare"))
    assert set(result["clauses"]) == set(REGISTERED_CLAUSES)
    for name in REGISTERED_CLAUSES:
        if name != CLAUSE:
            assert result["clauses"][name] == registered["clauses"][name]
    for key in registered:
        if key not in ("clauses", "decision", "reason"):
            assert result[key] == registered[key]
    if result["clauses"][CLAUSE] == registered["clauses"][CLAUSE]:
        assert (result["decision"], result["reason"]) == (
            registered["decision"],
            registered["reason"],
        )
    oracle = summarize(_with_registered_boost_clause(rows, result["clauses"][CLAUSE]))
    assert oracle["clauses"] == result["clauses"]
    assert (oracle["decision"], oracle["reason"]) == (result["decision"], result["reason"])


def test_randomized_inputs_cover_every_decision_and_both_clause_flips() -> None:
    """Guards the coverage of the equivalence test above (same seeds)."""
    decisions, flips = set(), set()
    for seed in range(SEEDS):
        rows = _random_records(random.Random(seed))
        registered, result = summarize(rows), _amended(rows)
        decisions.add(result["decision"])
        flips.add((registered["clauses"][CLAUSE], result["clauses"][CLAUSE]))
    assert decisions == {"GO", "STOP", "INCONCLUSIVE"}
    assert flips == {(True, True), (True, False), (False, True), (False, False)}


# ------------------------------------------------------------ source binding ---


def test_real_amendment_binds_the_ratified_round0_sources() -> None:
    amendment = read_json(decide.AMENDMENT)
    ratification = ROOT / "research/t34_round0/REGISTRATION_ROUND0.json"
    registration = read_json(ratification)
    decide.verify_amendment(amendment, registration, ratification)
    decide.verify_sources(ROOT, amendment, registration["commit"])
    tampered = dict(amendment, registered_sources={"research/t34_round0/analyze.py": "0" * 64})
    with pytest.raises(decide.AmendmentRefused, match="differs"):
        decide.verify_sources(ROOT, tampered, registration["commit"])


# --------------------------------------------------------- one-look mechanics ---


def _git(repo: Path, *args: str, date: str | None = None) -> str:
    env = dict(os.environ)
    if date:
        env.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True, env=env
    ).stdout.strip()


class Study:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        remote, self.repo = tmp_path / "origin.git", tmp_path / "author"
        subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
        _git(tmp_path, "init", str(self.repo))
        _git(self.repo, "config", "user.email", "round0@example.test")
        _git(self.repo, "config", "user.name", "Round 0 amend test")
        _git(self.repo, "checkout", "-b", "main")
        _git(self.repo, "remote", "add", "origin", str(remote))
        (self.repo / "README").write_text("fixture\n")
        _git(self.repo, "add", "README")
        _git(self.repo, "commit", "-m", "ratified source", date="2020-01-01T00:00:00")
        _git(self.repo, "push", "-u", "origin", "main")
        self.root = tmp_path / "round0-study"
        self.registration = {
            "status": "RATIFIED",
            "study_root": str(self.root),
            "commit": _git(self.repo, "rev-parse", "HEAD"),
            "history_ledger": {
                "repo": str(self.repo.resolve()),
                "path": f"research/t34_round0/ledger/{self.root.name}.jsonl",
            },
        }
        self.ratification = tmp_path / "ratification.json"
        self.ratification.write_text(json.dumps(self.registration, sort_keys=True) + "\n")
        self.amendment = self.repo / decide.AMENDMENT_NAME
        self.out = self.root / decide.PHASE / "decision.json"

    def publish_amendment(self, date: str = "2020-01-02T00:00:00") -> None:
        self.amendment.parent.mkdir(parents=True, exist_ok=True)
        value = dict(
            schema="t34-round0-analysis-amendment/v1",
            status="DECLARED_PRE_LOOK",
            ratification_canonical_sha256=_registration_sha(self.registration),
            ratification_file_sha256=sha256(self.ratification),
            ratified_commit=self.registration["commit"],
            study_root=self.registration["study_root"],
            registered_sources={},
        )
        self.amendment.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        _git(self.repo, "add", decide.AMENDMENT_NAME)
        _git(self.repo, "commit", "-m", "declare amendment", date=date)
        _git(self.repo, "push", "origin", "main")

    def write_chunk(self, rows: list[dict]) -> dict[str, str]:
        chunk = self.tmp / "exports/round0"
        (chunk / "pairs").mkdir(parents=True)
        entries = []
        for row in rows:
            path = chunk / "pairs" / f"{row['world']['world_id']}.json"
            pair = dict(row, schema="t34-round0-pair/v1", status="complete", preflight=False)
            path.write_text(json.dumps(pair, sort_keys=True))
            entries.append({"path": f"pairs/{path.name}", "sha256": sha256(path)})
        receipt = chunk / "chunk.json"
        receipt.write_text(json.dumps({"pairs": entries}, sort_keys=True))
        return {str(receipt): sha256(receipt)}

    def registered_analysis(self, decision: dict, *, publish: bool = True) -> None:
        analysis = self.root / "analysis"
        history.reserve(analysis, self.registration)
        (analysis / "decision.json").write_text(json.dumps(decision, sort_keys=True) + "\n")
        history.finalize(analysis, self.registration, "COMPLETE", "source-fenced phase complete")
        if publish:
            self.publish_ledger(self.registration["history_ledger"]["path"])

    def publish_ledger(self, name: str) -> None:
        _git(self.repo, "add", name)
        _git(self.repo, "commit", "-m", "publish ledger")
        _git(self.repo, "push", "origin", "main")

    def run(self) -> dict:
        """Run from a separate clean checkout of origin/main, as in production."""
        checkout = self.tmp / "amend-checkout"
        if not checkout.exists():
            _git(self.tmp, "clone", "--quiet", str(self.tmp / "origin.git"), str(checkout))
        return decide.run(
            self.ratification,
            self.out,
            checkout=checkout,
            amendment_path=checkout / decide.AMENDMENT_NAME,
        )

    def remote_tag(self) -> str:
        return _git(self.repo, "ls-remote", "--refs", "origin", decide.tag_ref(self.root))


@pytest.fixture
def study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Study:
    monkeypatch.setattr(decide, "verify_sources", lambda *_args: None)
    return Study(tmp_path)


def _registered_from(study: Study, rows: list[dict]) -> dict:
    chunk_hashes = study.write_chunk(rows)
    return json.loads(
        json.dumps(
            dict(
                summarize(rows),
                evidence_authority="TIER_1_DEVELOPMENT",
                release_authorized=False,
                chunk_sha256=chunk_hashes,
            )
        )
    )


def test_refuses_before_the_registered_analysis_exists(study: Study) -> None:
    study.publish_amendment()
    with pytest.raises(decide.AmendmentRefused, match="not been sealed"):
        study.run()
    assert not (study.root / decide.PHASE).exists()
    assert not study.remote_tag()


def test_refuses_unpublished_registered_ledger(study: Study) -> None:
    study.publish_amendment()
    rows = passing_records(_boosts_for(1300), _boosts_for(1200))
    study.registered_analysis(_registered_from(study, rows), publish=False)
    with pytest.raises(decide.AmendmentRefused, match="analysis ledger entry is not published"):
        study.run()
    assert not study.remote_tag()


def test_refuses_amendment_published_after_the_analysis_claim(study: Study) -> None:
    rows = passing_records(_boosts_for(1300), _boosts_for(1200))
    study.registered_analysis(_registered_from(study, rows))
    study.publish_amendment(date="2090-01-01T00:00:00")
    with pytest.raises(decide.AmendmentRefused, match="before the analysis"):
        study.run()
    assert not study.remote_tag()


def test_one_governing_look_then_every_second_look_is_refused(study: Study) -> None:
    study.publish_amendment()
    rows = passing_records(_boosts_for(1300), _boosts_for(1200))
    registered = _registered_from(study, rows)
    assert registered["decision"] == "STOP"
    study.registered_analysis(registered)

    result = study.run()
    assert result["decision"] == "GO" and result["governing"] is True
    assert result["registered_decision"] == "STOP"
    detail = result["boost_floor_amendment"]
    assert detail["applied"] and detail["registered_clause"] is False
    assert detail["bare_boost_decisions"] == 1440 and detail["teacher_boost_decisions"] == 1560
    assert read_json(study.out) == result
    terminal = read_json(study.out.parent / "phase_terminal.json")
    assert terminal["status"] == "COMPLETE"
    ledger = study.repo / decide.ledger_name(study.root)
    entry = json.loads(ledger.read_text())
    assert entry == {
        "phase": decide.PHASE,
        "terminal_sha256": sha256(study.out.parent / "phase_terminal.json"),
    }
    assert study.remote_tag()
    # The registered phase and Round 0's own claim inventory are untouched.
    assert set(history._claims(study.registration)) == {"analysis"}

    study.publish_ledger(decide.ledger_name(study.root))
    with pytest.raises(decide.AmendmentRefused, match="create-only"):
        study.run()
    shutil.rmtree(study.out.parent)  # even deleting local evidence cannot re-open the look
    with pytest.raises(decide.AmendmentRefused, match="already records a look"):
        study.run()


def test_remote_claim_alone_blocks_a_second_look(study: Study) -> None:
    study.publish_amendment()
    rows = passing_records(_boosts_for(1300), _boosts_for(1200))
    study.registered_analysis(_registered_from(study, rows))
    study.run()
    shutil.rmtree(study.out.parent)
    shutil.rmtree(study.repo / "research/t34_round0_amend/ledger")
    with pytest.raises(decide.AmendmentRefused, match="remote claim already exists"):
        study.run()


@pytest.mark.parametrize(
    "registered",
    [
        dict(
            schema="t34-round0-decision/v1", decision="KILL", reason="x", release_authorized=False
        ),
        dict(
            schema="t34-round0-decision/v1",
            decision="INCONCLUSIVE",
            reason="cap",
            evidence_authority="INVALID_OR_INCOMPLETE",
            release_authorized=False,
        ),
        dict(
            schema="t34-round0-decision/v1",
            decision="STOP",
            reason="unconditional fault; this teacher/controller failed its registered bar",
            evidence_authority="TIER_1_DEVELOPMENT",
            release_authorized=False,
            unconditional_faults=[{"fault": "illegal_action"}],
        ),
    ],
)
def test_verdicts_that_precede_clauses_pass_through(study: Study, registered: dict) -> None:
    study.publish_amendment()
    study.registered_analysis(registered)
    result = study.run()
    assert result["decision"] == registered["decision"]
    assert result["reason"] == registered["reason"]
    assert result["boost_floor_amendment"]["applied"] is False


def test_altered_bound_record_is_a_sealed_kill(study: Study) -> None:
    study.publish_amendment()
    rows = passing_records(_boosts_for(1300), _boosts_for(1200))
    study.registered_analysis(_registered_from(study, rows))
    pair = next((study.tmp / "exports/round0/pairs").iterdir())
    pair.write_text(pair.read_text().replace('"boost_decisions": 6', '"boost_decisions": 5', 1))
    result = study.run()
    assert result["decision"] == "KILL"
    assert result["evidence_authority"] == "INTEGRITY_FAILURE"
    assert read_json(study.out.parent / "phase_terminal.json")["status"] == "KILL"
    assert study.remote_tag()


def test_non_reproducing_registered_summary_is_a_sealed_kill(study: Study) -> None:
    study.publish_amendment()
    rows = passing_records(_boosts_for(1300), _boosts_for(1200))
    registered = _registered_from(study, rows)
    registered["mass"]["mean"] += 1.0
    study.registered_analysis(registered)
    result = study.run()
    assert result["decision"] == "KILL"
    assert "reproduce registered mass" in result["reason"]
