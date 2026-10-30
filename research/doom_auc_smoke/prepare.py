"""Prepare source-bound packets offline; genuine canary receipts admit the main."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path

from research.colab_cli.canary import SCHEMA, family
from research.compute.gpu_preference import prepare_preference, variants
from research.doom_auc_smoke.data import load_gzip, save_gzip
from research.doom_auc_smoke.evidence import digest, empty_ledger, json_once

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/doom_auc_smoke"
SCRIPT = PACKAGE + "/job.sh"
REQUIREMENTS = PACKAGE + "/requirements.txt"
THREADS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OMP_THREAD_LIMIT",
    "TORCHINDUCTOR_COMPILE_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def binding(path: Path, dest: str | None = None) -> dict[str, str]:
    row = dict(path=str(path.resolve()), sha256=digest(path))
    if dest is not None:
        row["dest"] = dest
    return row


def sources(commit: str) -> list[str]:
    """Compute conservative static closure from the pin, excluding outcomes/tests."""
    from tools.ops.import_preflight import import_closure, module_path

    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, text=True
    ).splitlines()
    archive = subprocess.check_output(
        ["git", "archive", commit, "--", *[n for n in names if n.endswith(".py")]], cwd=ROOT
    )
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(base, filter="data")
        modules, _ = import_closure(base, ["research.doom_auc_smoke.run"])
        paths = [str(module_path(base, m).relative_to(base)) for m in modules]
    paths.extend(
        n
        for n in names
        if n.startswith("research/colab_cli/")
        and n.endswith((".py", ".sh"))
        and not Path(n).name.startswith("test_")
    )
    return sorted(
        set(
            paths
            + [
                SCRIPT,
                REQUIREMENTS,
                "pyproject.toml",
                PACKAGE + "/REGISTRATION.md",
                PACKAGE + "/REGISTRATION_OPERATIONAL_CLARIFICATION_01.md",
            ]
        )
    )


def manifest(
    config: dict, inputs: list[dict], source_paths: list[str], output: Path, job: str, mode: str
) -> dict:
    if mode not in {"canary", "fit"} or not output.is_absolute() or output.exists():
        raise ValueError("mode and fresh absolute output required")
    timeout = 30 if mode == "canary" else 120
    result = dict(
        job_id=job,
        commit=config["commit"],
        script=SCRIPT,
        device="cuda",
        source_paths=source_paths,
        inputs=inputs,
        requirements=dict(min_vcpu=2, est_minutes=timeout // 2),
        timeout_minutes=timeout,
        output_dir=str(output),
        max_export_bytes=256 * 1024**2,
        required_exports=["doom/summary.json"],
        registration_sha256=config["registration_sha256"],
        env=dict(
            DOOM_MODE=mode,
            DOOM_SOURCE_COMMIT=config["commit"],
            COLAB_CLI_REQUIREMENTS=REQUIREMENTS,
            SNAKE_DQN_DEVICE="cuda",
            **{k: "1" for k in THREADS},
        ),
        budget=dict(reserve_units=timeout * 12 / 60 / 6, other_assignments={}),
        resume=dict(
            unit_kind="doom-full-state-model-step",
            ledger_path="inputs/verified-ledger.json",
            skip_verified=True,
            max_outcome_blind_retries=2,
        ),
        technical_envelope=dict(
            useful_worker_seconds=600 if mode == "canary" else 2700,
            series_max_original_plus_completions=3,
        ),
        hardware_envelopes={
            gpu: dict(
                base_timeout_minutes=timeout / 2,
                effective_timeout_minutes=timeout,
                base_unit_cap=timeout / 2 * rate / 60,
                effective_unit_cap=timeout * rate / 60,
                base_reserve_units=timeout / 2 * 12 / 60 / 6,
                effective_reserve_units=timeout * 12 / 60 / 6,
            )
            for gpu, rate in (("H100", 12), ("A100", 8))
        },
    )
    prepare_preference(result)
    if mode == "canary":
        result["canary"] = dict(schema=SCHEMA, mode="probe", family=family(result))
        result["env"].update(SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE=PACKAGE)
        # Synthetic transport/fit canary; no real model trajectory to resume.
        result.pop("resume")
        result["resume_exemption"] = dict(
            kind="short-canary", reason="Synthetic full-board doom fit; no scoring look"
        )
    return result


def commands(packet: Path, job: str, filename: str) -> list[str]:
    argv = [
        str(ROOT / "tools/ops/dispatch.sh"),
        job,
        str(packet / filename),
        "--broker",
        str(ROOT),
        "--cwd",
        str(ROOT),
        "--state-dir",
        str(packet / (job + "-state")),
        "--dispatch-txt",
        str(packet / "DISPATCH.txt"),
    ]
    return [shlex.join([argv[0], "--dry-run", *argv[1:]]), shlex.join(argv)]


def prepare(packet: Path, commit: str, bank: Path, labels: Path, checkpoint: Path) -> dict:
    import torch

    from research.doom_auc_smoke.data_mac import convert
    from tools.ops.import_preflight import check

    if not packet.is_absolute() or packet.exists():
        raise ValueError("fresh absolute packet required")
    commit = subprocess.check_output(
        ["git", "rev-parse", commit + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    # No ambient source may quietly differ from the pin used by qualification.
    paths = sources(commit)
    for path in paths:
        pinned = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
        if (ROOT / path).read_bytes() != pinned:
            raise ValueError(f"runtime source differs from commit: {path}")
    imports = check(ROOT, commit, paths, REQUIREMENTS, ["research.doom_auc_smoke.run"])
    runtime, heldout, audit = convert(bank, labels, checkpoint)
    packet.mkdir(parents=True)
    save_gzip(packet / "dataset.pt.gz", runtime)
    # Byte-exact tensor roundtrip, including every body/food cell, no lossy observation archive.
    decoded = load_gzip(packet / "dataset.pt.gz")
    for original, restored in zip(runtime["compact"], decoded["compact"]):
        for key in original:
            if isinstance(original[key], torch.Tensor):
                if not torch.equal(original[key], restored[key]):
                    raise ValueError("lossless compact roundtrip failed")
            elif original[key] != restored[key]:
                raise ValueError("lossless compact metadata roundtrip failed")
    save_gzip(packet / "assessment-MAC-ONLY.pt.gz", heldout)
    registration = subprocess.check_output(
        ["git", "show", f"{commit}:{PACKAGE}/REGISTRATION.md"], cwd=ROOT
    )
    (packet / "REGISTRATION.md").write_bytes(registration)
    clarification = "REGISTRATION_OPERATIONAL_CLARIFICATION_01.md"
    (packet / clarification).write_bytes(
        subprocess.check_output(["git", "show", f"{commit}:{PACKAGE}/{clarification}"], cwd=ROOT)
    )
    config = dict(
        schema="doom-fixed-config/v1",
        commit=commit,
        mac_look_file=str(packet.parent / "SCORING_LOOK.json"),
        dataset_sha256=digest(packet / "dataset.pt.gz"),
        assessment_sha256=digest(packet / "assessment-MAC-ONLY.pt.gz"),
        registration_sha256=digest(packet / "REGISTRATION.md"),
        operational_clarification_sha256=digest(packet / clarification),
    )
    config["identity"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    json_once(packet / "config.json", config)
    json_once(packet / "verified-ledger.json", empty_ledger(config["identity"]))
    json_once(packet / "split-audit.json", audit)
    json_once(packet / "import-preflight.json", imports)
    inputs = [
        binding(packet / name, name)
        for name in (
            "config.json",
            "dataset.pt.gz",
            "REGISTRATION.md",
            clarification,
            "verified-ledger.json",
        )
    ]
    for mode in ("canary", "fit"):
        job = "t34-doom-" + mode + "-01"
        item = manifest(config, inputs, paths, packet / (mode + "-output"), job, mode)
        json_once(packet / (mode + ".json" if mode == "canary" else "fit.pending.json"), item)
        if mode == "canary":
            for variant in variants(item):
                gpu = variant["requirements"]["gpu"].lower()
                variant["requirements"].pop("prefer", None)
                json_once(packet / f"canary-{gpu}.json", variant)
    lines = [
        "Prepared offline. Operator runs real same-pin H100/A100 family canaries,",
        "then the admit command seals fit.json with both verified receipts.",
        *commands(packet, "t34-doom-canary-01", "canary.json"),
        *commands(packet, "t34-doom-canary-a100-01", "canary-a100.json"),
        *commands(packet, "t34-doom-fit-01", "fit.json"),
    ]
    (packet / "DISPATCH.txt").write_text("\n".join(lines) + "\n")
    uploaded = sum(Path(i["path"]).stat().st_size for i in inputs)
    # Conservative many-medium-file staging floor; <=150s still needs live canary proof.
    projection = dict(
        private_upload_bytes=uploaded,
        private_files=len(inputs),
        nominal_seconds=uploaded / 1635817 + 2.193 * (len(inputs) + 3) + 10,
        conservative_seconds=uploaded / 160000 + 3.67 * (len(inputs) + 3) + 10,
        measured_gpu_rate=False,
        source_paths=len(paths),
        assessment_uploaded=False,
        lossless_compact_roundtrip=True,
        max_export_bytes=256 * 1024**2,
        source_commit=commit,
    )
    if projection["conservative_seconds"] > 150:
        raise ValueError("projected staged inputs exceed 150s family canary bar")
    json_once(packet / "PREPARATION.json", projection)
    return projection


def fit_forecast(timings: list[float]) -> float:
    """Charge cold initialization once, then extrapolate the slower measured warm step."""
    if len(timings) != 3 or any(not math.isfinite(t) or t <= 0 for t in timings):
        raise ValueError("canary requires three finite positive step timings")
    return timings[0] + 2999 * max(timings[1:])


def admit(packet: Path, h100: Path, a100: Path) -> None:
    """No allocation: unchanged supervisor validates genuine same-pin family proofs."""
    from research.colab_cli.supervisor import validate

    item = json.loads((packet / "fit.pending.json").read_text())
    item["requirements"]["hardware_canaries"] = dict(H100=binding(h100), A100=binding(a100))
    for variant in variants(item):
        validate(variant, ROOT)
    rates = {}
    for gpu, receipt_path in (("H100", h100), ("A100", a100)):
        config = json.loads((packet / "config.json").read_text())
        from research.doom_auc_smoke.evidence import build_ledger, load_unit

        identity = "canary-" + config["identity"]
        ledger = build_ledger([receipt_path.parent], identity)
        arms = {}
        for arm in ("spatial", "b8"):
            row = next(r for r in ledger["units"] if r["unit"] == arm + "-prediction")
            prediction = load_unit(receipt_path.parent / "exports/doom", row, identity)
            timings = prediction["measured_step_seconds"]
            if prediction["steps"] != 3:
                raise ValueError("canary lacks three real optimizer step timings")
            arms[arm] = fit_forecast(timings)
        rates[gpu] = dict(
            projected_fit_seconds=arms, projected_useful_seconds=sum(arms.values()) + 120
        )
        if rates[gpu]["projected_useful_seconds"] > 2700:
            raise ValueError("same-pin measured GPU fit forecast exceeds 45-minute worker envelope")
    json_once(packet / "CANARY_RATES.json", rates)
    json_once(packet / "fit.json", item)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    initial = sub.add_parser("build")
    initial.add_argument("--packet", type=Path, required=True)
    initial.add_argument("--commit", required=True)
    initial.add_argument("--bank", type=Path, required=True)
    initial.add_argument("--labels", type=Path, required=True)
    initial.add_argument("--checkpoint", type=Path, required=True)
    accepted = sub.add_parser("admit")
    accepted.add_argument("--packet", type=Path, required=True)
    accepted.add_argument("--h100-canary", type=Path, required=True)
    accepted.add_argument("--a100-canary", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "build":
        print(
            json.dumps(
                prepare(args.packet.resolve(), args.commit, args.bank, args.labels, args.checkpoint)
            )
        )
    else:
        admit(args.packet.resolve(), args.h100_canary.resolve(), args.a100_canary.resolve())


if __name__ == "__main__":
    main()
