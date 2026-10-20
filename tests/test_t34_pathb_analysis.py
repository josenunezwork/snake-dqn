"""Synthetic fixed-bank tests; no efficacy gameplay or timing probes run here."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import subprocess

import pytest

analysis = importlib.import_module("research.t34_pathb.analyze")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def metrics(*, candidate=False, kind="terminal", mass=1000, no_death=1):
    result = {
        "mass_integral": mass + (50 if candidate else 0),
        "decisions": 1000,
        "boost_decisions": 10,
        "boost_share": 0.01,
        "survival_fraction": 1,
        "no_death": no_death,
        "deaths": 1 if candidate else 2,
        "deaths_per_1k": 1 if candidate else 2,
        "median_length": 800,
        "p95_length": 1000,
        "deaths_length_ge_800": 1,
        "late_deaths": 1,
        "late_large_overlap": 1,
    }
    if kind == "terminal" and not no_death:
        result["survival_fraction"] = 0.9
    return result


@pytest.fixture
def bank(tmp_path, monkeypatch):
    # Identity validation is tested independently below. Statistics fixtures use
    # fictional checkpoint bytes, never production worlds or results.
    monkeypatch.setattr(analysis, "_verify_identity", lambda intent: None)
    blocks = []
    for block in range(3):
        terminal = []
        for index, mix in enumerate(analysis.MIXES):
            for repeat in range(2):
                terminal.append(
                    {"seed": block * 100 + index * 10 + repeat, "mix": mix, "roster": ["fixed"]}
                )
        product = [{"seed": block * 100 + 90, "mix": "mixed", "roster": ["fixed"]}]
        blocks.append({"block": block, "terminal": terminal, "product": product})
        for kind, worlds in (("terminal", terminal), ("product", product)):
            for row in worlds:
                for fps in (12, 24):
                    # Scripted baseline one death in two, assisted all survive.
                    no_death = 0 if row["mix"] == "scripted" and row["seed"] % 2 == 0 else 1
                    record = {
                        "schema": f"t34-pathb-{kind}/v1",
                        "status": "complete",
                        "kind": kind,
                        "block": block,
                        "seed": row["seed"],
                        "mix": row["mix"],
                        "fps": fps,
                        "arms": {
                            "bare": metrics(kind=kind, no_death=no_death),
                            "async": metrics(candidate=True, kind=kind),
                            "sync": metrics(candidate=True, kind=kind),
                        },
                    }
                    write(tmp_path / "records" / f"{kind}-{block}-{row['seed']}-{fps}.json", record)
    manifest = {
        "schema": "t34-pathb-manifest/v1",
        "expected_records": 42,
        "settings": {
            "horizon": 5000,
            "fps": [12, 24],
            "workers": 1,
            "smoke": False,
            "timing_samples": 3,
        },
        "bank": {"blocks": blocks},
    }
    write(tmp_path / "manifest.json", manifest)
    for mode in ("bare", "async"):
        for fps in (12, 24):
            write(
                tmp_path / "timing" / f"{mode}-{fps}.json",
                {
                    "status": "complete",
                    "mode": mode,
                    "fps": fps,
                    "actual_speed": fps,
                    "speed_ceiling": 24,
                    "elapsed_seconds": 3 / fps + 0.009,
                    "samples": [
                        {
                            "update_ms": 1.0,
                            "tick_ms": 2.0,
                            "hero_length": 700,
                            "worker_busy": mode == "async",
                        }
                        for _ in range(3)
                    ],
                },
            )
    return tmp_path


def record_path(root, kind="terminal", block=0, seed=0, fps=12):
    return root / "records" / f"{kind}-{block}-{seed}-{fps}.json"


def mutate(path, edit):
    value = json.loads(path.read_text())
    edit(value)
    write(path, value)


def test_complete_development_pass_never_claims_client_or_release(bank):
    report = analysis.analyze(bank)
    assert report["decision"] == "DEVELOPMENT_BARS_PASS", report
    assert report["serving_qualification"] == "NOT_MEASURED"
    assert report["client_latency"] == "NOT_MEASURED"
    assert report["release_authorized"] is False
    assert report["efficacy_by_fps"]["12"]["async"]["skill"]["mass_gain_lb90"]["n"] == 3


def test_ci_uses_student_t_and_independent_blocks():
    result = analysis._ci([1, 2, 3])
    assert result["df"] == 2
    assert result["se"] == pytest.approx(1 / math.sqrt(3))
    assert result["lower"] == pytest.approx(2 - 1.885618083 * result["se"])
    assert analysis._ci([500])["status"] == "NOT_ESTIMABLE"


def test_equal_mix_weighting_and_frame_rates_not_pooled(bank):
    # Six terminal rows/block; only scripted gains vary. Four extra scripted
    # replicas below leave equal-mix weighting unchanged despite world imbalance.
    for path in (bank / "records").glob("terminal*.json"):

        def edit(row):
            gain = {"scripted": 120, "frozen": 0, "mixed": 0}[row["mix"]]
            row["arms"]["async"]["mass_integral"] = 1000 + (gain if row["fps"] == 12 else 0)

        mutate(path, edit)
    rows = [json.loads(path.read_text()) for path in (bank / "records").glob("terminal*.json")]
    first = [row for row in rows if row["fps"] == 12]
    extra = [dict(row, seed=row["seed"] + 10000) for row in first if row["mix"] == "scripted"]
    assert analysis._skill(first + extra, "async")["mass_gain_lb90"]["mean"] == 40
    report = analysis.analyze(bank)
    assert report["efficacy_by_fps"]["12"]["async"]["skill"]["mass_gain_lb90"]["mean"] == 40
    assert report["efficacy_by_fps"]["24"]["async"]["skill"]["mass_gain_lb90"]["mean"] == 0
    assert report["decision"] == "DEVELOPMENT_BARS_FAIL"


@pytest.mark.parametrize(
    "problem",
    [
        "missing_row",
        "missing_fps",
        "duplicate",
        "missing_mix",
        "missing_arm",
        "missing_metric",
        "nonfinite",
        "bad_rate",
    ],
)
def test_missing_and_malformed_data_fail_closed(bank, problem):
    path = record_path(bank)
    if problem == "missing_row":
        path.unlink()
    elif problem == "missing_fps":
        mutate(path, lambda row: row.update(fps=30))
    elif problem == "duplicate":
        other = record_path(bank, seed=1)
        other.write_text(path.read_text())
    elif problem == "missing_mix":
        mutate(
            bank / "manifest.json",
            lambda m: m["bank"]["blocks"][0].update(
                terminal=[r for r in m["bank"]["blocks"][0]["terminal"] if r["mix"] != "mixed"]
            ),
        )
    elif problem == "missing_arm":
        mutate(path, lambda row: row["arms"].pop("sync"))
    elif problem == "missing_metric":
        mutate(path, lambda row: row["arms"]["async"].pop("mass_integral"))
    elif problem == "nonfinite":
        mutate(path, lambda row: row["arms"]["async"].update(mass_integral=float("nan")))
    else:
        mutate(path, lambda row: row["arms"]["async"].update(boost_share=0.09))
    assert analysis.analyze(bank)["decision"] == "INCOMPLETE"


def test_product_relative_effect_is_paired_before_block_average(bank):
    # Paired 10% and -10% average to 0%; ratio of arm means is -9.8%.
    rows = []
    for block in range(3):
        for seed, base, assisted in ((1, 10, 11), (2, 1000, 900)):
            rows.append(
                {
                    "block": block,
                    "seed": seed,
                    "arms": {
                        "bare": metrics(kind="product", mass=base),
                        "async": metrics(kind="product", mass=assisted),
                    },
                }
            )
    product = analysis._product(rows, "async")
    assert product["relative_mass_delta95"]["mean"] == pytest.approx(0)
    assert product["relative_mass_delta95"]["n"] == 3


def test_product_counters_and_zero_denominator_fail_closed(bank):
    path = record_path(bank, kind="product", seed=90)
    mutate(path, lambda row: row["arms"]["bare"].update(mass_integral=0))
    assert analysis.analyze(bank)["decision"] == "INCOMPLETE"
    mutate(path, lambda row: row["arms"]["bare"].update(mass_integral=1000, deaths_per_1k=0))
    assert analysis.analyze(bank)["decision"] == "INCOMPLETE"


def test_sync_gain_cannot_rescue_deadline_arm(bank):
    for path in (bank / "records").glob("terminal*.json"):
        mutate(path, lambda row: row["arms"]["async"].update(mass_integral=1000))
    report = analysis.analyze(bank)
    assert report["decision"] == "DEVELOPMENT_BARS_FAIL"
    assert report["efficacy_by_fps"]["12"]["sync_unlimited_descriptive"]["skill"]["pass"] is True


def test_timing_missing_row_and_unpaced_elapsed_rejected(bank):
    path = bank / "timing" / "async-12.json"
    original = path.read_text()
    mutate(path, lambda row: row.update(elapsed_seconds=0.01))
    assert analysis.analyze(bank)["decision"] == "INCOMPLETE"
    path.write_text(original)
    path.unlink()
    assert analysis.analyze(bank)["decision"] == "INCOMPLETE"


def test_timing_thresholds_and_unobserved_tails_reported(bank):
    path = bank / "timing" / "async-24.json"
    mutate(
        path,
        lambda row: row.update(
            samples=[{"update_ms": 12, "tick_ms": 14, "hero_length": 30, "worker_busy": False}] * 3,
            elapsed_seconds=3 / 24 + 0.042,
        ),
    )
    report = analysis.analyze(bank)
    assert report["decision"] == "DEVELOPMENT_BARS_FAIL"
    timing = report["timing_by_fps"]["24"]
    assert timing["clauses"]["update_p95_le_11688ms"] is False
    assert timing["async"]["natural_long_body_tick_ms"]["status"] == "NOT_MEASURED"
    assert timing["async"]["worker_contention_tick_ms"]["status"] == "NOT_MEASURED"


def test_smoke_omits_skill_and_allows_product_only_plumbing(bank):
    def edit(manifest):
        manifest["settings"]["smoke"] = True
        for block in manifest["bank"]["blocks"]:
            block["terminal"] = []
        manifest["expected_records"] = 6

    mutate(bank / "manifest.json", edit)
    for path in (bank / "records").glob("terminal*.json"):
        path.unlink()
    report = analysis.analyze(bank)
    assert report["decision"] == "SMOKE_ONLY"
    assert report["skill"] == "NOT_ESTIMABLE"
    assert "efficacy_by_fps" not in report


def test_identity_hashes_and_source_commit_are_verified(tmp_path, monkeypatch):
    protocol = tmp_path / "research/t34_pathb/PROTOCOL.md"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("frozen protocol\n")
    producer = tmp_path / "producer.py"
    producer.write_text("original producer\n")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"synthetic fixture only")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "producer.py", str(protocol)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    cp = {"path": str(checkpoint), "sha256": digest(checkpoint)}
    intent = {
        "source_commit": commit,
        "protocol_sha256": digest(protocol),
        "source_files": {"producer.py": digest(producer)},
        "checkpoint": cp,
        "checkpoints": [cp],
    }
    monkeypatch.setattr(analysis, "REPO", tmp_path)
    monkeypatch.setattr(analysis, "B8_SHA256", digest(checkpoint))
    analysis._verify_identity(intent)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(analysis.InvalidBank, match="checkpoint hash"):
        analysis._verify_identity(intent)
    checkpoint.write_bytes(b"synthetic fixture only")
    producer.write_text("uncommitted altered producer\n")
    with pytest.raises(analysis.InvalidBank, match="current source hash"):
        analysis._verify_identity(intent)
    intent["source_files"]["producer.py"] = digest(producer)
    with pytest.raises(analysis.InvalidBank, match="source commit hash"):
        analysis._verify_identity(intent)
