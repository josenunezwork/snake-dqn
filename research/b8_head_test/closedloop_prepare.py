"""Fit one all-root head, audit fresh seeds, and seal/queue-ready Colab packets offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

import torch

from research.b8_head_test.closedloop import NAMESPACE, ledger, seed
from research.doom_auc_smoke.evidence import atomic, digest, json_once

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ROOT.parent / "snake-dqn"
ARTIFACTS = ROOT.parent / "snake-dqn-artifacts"
PACKAGE = "research/b8_head_test"
SCRIPT = PACKAGE + "/closedloop_job.sh"
REQUIREMENTS = "research/spatial_student_pilot/requirements.txt"


def inputs(output: Path) -> None:
    from research.b8_head_test.run import (
        B8,
        BANK,
        LABELS,
        fit_fold,
        outcomes,
        save_tensor,
        training_bank,
    )
    from research.t34_pathb.bank import _audit_candidates

    # Scan before writing any raw new keys into artifact history.
    keys = [
        seed(mode, i, opp)
        for mode, count in (("main", 192), ("canary", 3), ("preflight", 3))
        for i in range(count)
        for opp in (False, True)
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("internal seed collision; reject entire allocation")
    audit = _audit_candidates(PRIMARY, keys)
    # Also scan this branch, whose unmerged inventories may differ from main.
    branch_audit = _audit_candidates(ROOT, keys)
    output.mkdir(parents=True, exist_ok=False)
    json_once(
        output / "seed-audit.json",
        dict(
            namespace=NAMESPACE,
            allocation_index=0,
            status="PASS",
            primary=audit,
            branch=branch_audit,
            key_digest=hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
            domains=dict(main=192, canary=3, preflight=3),
        ),
    )
    data, provenance = training_bank(BANK, B8)
    selection, _ = outcomes(data, LABELS, "sel")
    means = selection.mean(-1)
    target = means - means.gather(1, data["b8"][:, None])
    target = torch.where(data["legal"], target, 0)
    ids = torch.arange(2048)
    _, state = fit_fold(data["phi"], target, data["legal"], ids, torch.empty(0, dtype=torch.long))
    save_tensor(
        output / "head.pt",
        dict(schema="b8-allroot-advantage-head/v1", **state, provenance=provenance),
    )
    from research.spatial_student_pilot.prepare import B8_SHA, CHECKPOINT_NAMES, OPPONENT_SHAS

    hashes = {
        "head.pt": digest(output / "head.pt"),
        "seed-audit.json": digest(output / "seed-audit.json"),
    }
    for name, source in CHECKPOINT_NAMES.items():
        path = PRIMARY / "saved_snakes" / source
        expected = B8_SHA if name == "b8-s3.pth" else OPPONENT_SHAS[name]
        if digest(path) != expected:
            raise ValueError("foreign checkpoint")
        atomic(output / name, path.read_bytes())
        hashes[name] = expected
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    json_once(
        output / "config.json",
        dict(
            namespace=NAMESPACE,
            allocation_index=0,
            identity=identity,
            hashes=hashes,
            margins=[0, 10, 25, 50],
            worlds=192,
            horizon=5000,
            batch_worlds=64,
            recipe=dict(
                roots=2048,
                worlds=524,
                updates=3000,
                batch=32,
                init_seed=343010,
                sampler_seed=343011,
                learning_rate=0.0003,
                weight_decay=0.0001,
            ),
        ),
    )
    json_once(output / "verified-ledger.json", ledger(identity + "/main"))
    json_once(
        output / "FIT.json",
        dict(
            head_sha256=hashes["head.pt"],
            updates=state["updates"],
            seconds=state["seconds"],
            train_roots=len(state["train"]),
            final_loss=state["final_loss"],
            provenance=provenance,
        ),
    )


def binding(path: Path, dest: str | None = None) -> dict:
    return dict(path=str(path.resolve()), sha256=digest(path), **({"dest": dest} if dest else {}))


def source_paths(commit: str) -> list[str]:
    """Full code subtrees retain dynamic native sources; no prior banks or outcomes."""
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, text=True
    ).splitlines()
    trees = (
        "src/",
        "research/gpu_sim_spike/",
        "research/sandbox_a_resident/",
        "research/spatial_student_pilot/",
        "research/colab_cli/",
        PACKAGE + "/",
    )
    paths = [
        n
        for n in names
        if n == "pyproject.toml"
        or n == "research/__init__.py"
        or (
            n.startswith(trees)
            and Path(n).suffix in {".py", ".sh", ".cu", ".cpp", ".h", ".txt"}
            and not Path(n).name.startswith("test_")
        )
    ]
    # Static import closure includes the adapters used by snapshot/vector helpers.
    from tools.ops.import_preflight import import_closure, module_path

    closure, _ = import_closure(ROOT, ["research.b8_head_test.closedloop"])
    paths += [str(module_path(ROOT, m).relative_to(ROOT)) for m in closure]
    for name in sorted(set(paths)):
        if (
            subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
            != (ROOT / name).read_bytes()
        ):
            raise ValueError(f"source differs from pin: {name}")
    return sorted(set(paths))


def packet(output: Path, input_dir: Path, commit: str, receipt: Path | None = None) -> None:
    from research.colab_cli.canary import SCHEMA, family
    from tools.ops.import_preflight import check

    commit = subprocess.check_output(
        ["git", "rev-parse", commit + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    paths = source_paths(commit)
    imports = check(ROOT, commit, paths, REQUIREMENTS, ["research.b8_head_test.closedloop"])
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((input_dir / "config.json").read_text())
    uploads = [
        binding(input_dir / name, name)
        for name in ["config.json", "verified-ledger.json", *config["hashes"]]
    ]
    mode = "main" if receipt else "canary"
    timeout, estimate = (180, 90) if receipt else (30, 10)
    job = "t34-b8-head-closedloop-" + mode + "-" + output.name
    manifest = dict(
        job_id=job,
        commit=commit,
        script=SCRIPT,
        device="cuda",
        source_paths=paths,
        inputs=uploads,
        requirements=dict(gpu="A100", allowed_gpus=["A100"], min_vcpu=2, est_minutes=estimate),
        timeout_minutes=timeout,
        output_dir=str(output / "output"),
        max_export_bytes=64 * 1024**2,
        required_exports=["closedloop/summary.json"],
        env=dict(
            B8_HEAD_MODE=mode,
            COLAB_CLI_REQUIREMENTS=REQUIREMENTS,
            SNAKE_DQN_DEVICE="cuda",
            **{
                n: "1"
                for n in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "OMP_THREAD_LIMIT",
                    "NUMEXPR_NUM_THREADS",
                    "TORCHINDUCTOR_COMPILE_THREADS",
                )
            },
        ),
        budget=dict(
            unit_cap=timeout * 8 / 60,
            rate_units_per_hour=8,
            reserve_units=timeout * 8 / 360,
            other_assignments={},
        ),
        technical_envelope=dict(
            base_timeout_minutes=timeout / 2,
            effective_timeout_minutes=timeout,
            base_unit_cap=timeout * 8 / 120,
            effective_unit_cap=timeout * 8 / 60,
            base_reserve_units=timeout * 8 / 720,
            effective_reserve_units=timeout * 8 / 360,
            series_max_original_plus_completions=3,
            workload_estimate_status="Mac tiny preflight; A100 forecast unmeasured",
            mechanical_completion_triggers=[
                "verified mechanical failure",
                "verified stop",
                "missing immutable batch units",
            ],
            namespace=NAMESPACE,
            allocation_index=0,
        ),
    )
    if receipt:
        manifest["canary"] = dict(receipt=binding(receipt))
        manifest["resume"] = dict(
            unit_kind="paired-parent-world-batch",
            ledger_path="inputs/verified-ledger.json",
            skip_verified=True,
            max_outcome_blind_retries=2,
        )
    else:
        manifest["canary"] = dict(schema=SCHEMA, mode="probe", family=family(manifest))
        manifest["env"].update(SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE=PACKAGE)
        manifest["resume_exemption"] = dict(
            kind="short-canary", reason="Excluded three-world H64 real-controller smoke"
        )
    from research.colab_cli.supervisor import validate

    validate(manifest, ROOT)
    json_once(output / "manifest.json", manifest)
    json_once(output / "IMPORTS.json", imports)
    argv = [
        str(PRIMARY / "tools/ops/dispatch.sh"),
        job,
        str(output / "manifest.json"),
        "--broker",
        str(ROOT),
        "--cwd",
        str(ROOT),
        "--state-dir",
        str(output / "state"),
        "--dispatch-txt",
        str(output / "DISPATCH.txt"),
    ]
    atomic(
        output / "DISPATCH.txt",
        (
            "\n".join([shlex.join([argv[0], "--dry-run", *argv[1:]]), shlex.join(argv)]) + "\n"
        ).encode(),
    )
    print(shlex.join(argv))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("inputs", "packet"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--commit", default="HEAD")
    p.add_argument("--receipt", type=Path)
    a = p.parse_args()
    if a.mode == "inputs":
        inputs(a.output)
    else:
        packet(a.output, a.input_dir, a.commit, a.receipt)


if __name__ == "__main__":
    main()
