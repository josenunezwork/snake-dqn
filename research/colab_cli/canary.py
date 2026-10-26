"""Source-bound target-platform canaries; preparation never allocates a runtime.

A canary runs the *same* Bash entrypoint and ships the full input inventory.
The operator supplies a small, excluded workload via that entrypoint's normal
controls. A successful run alone is insufficient: broker transfer measurements,
the runtime scanner, CUDA observations and verified release must all agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from research.colab_cli import transfer_policy

SCHEMA = "colab-family-canary/v1"
ACCEPTANCE_SCRIPT = "research/colab_cli/acceptance_probe.sh"
MAX_UPLOAD_SECONDS = 150.0
MAX_STAGING_SECONDS = 300.0  # independent of every transfer exception
MAX_CANARY_MINUTES = 30.0
FAMILIES = {
    "gpu_sim_spike": ("research/gpu_sim_spike/colab.sh",),
    "t34_claim_a": ("research/t34_claim_a/label.sh", "research/t34_claim_a/pipeline.sh"),
    "bl_prefix": ("research/bl_prefix/run.sh",),
    "bl_info_value": ("research/bl_info_value/job.sh",),
    "bl_ppo": ("research/bl_ppo/colab.sh", "research/bl_ppo/p1_colab.sh"),
    "bl_e16": ("research/bl_e16/colab.sh",),
    "bl_served_search": (
        "research/bl_served_search/run_l2.sh",
        "research/bl_served_search/run_ruler.sh",
    ),
}


class CanaryError(ValueError):
    """A dispatch lacks target-platform transport/execution proof."""


class StagingCapError(CanaryError):
    """Measured input staging exceeded its prospective transport allowance."""


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def family(manifest: dict[str, Any]) -> str:
    """A family includes the real entrypoint and operational mode, not sizing."""
    script = str(manifest["script"])
    env = manifest.get("env", {})
    modes = {key: env[key] for key in ("BL_PPO_MODE", "BL_P1_MODE", "BL_E16_ARM") if key in env}
    return script + (
        ":" + json.dumps(modes, sort_keys=True, separators=(",", ":")) if modes else ""
    )


def executable_identity(repo: Path, commit: str | None = None) -> dict[str, str]:
    """Bind the entire transport inventory, including newly added modules."""
    if commit:
        names = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", commit, "research/colab_cli"],
            cwd=repo,
            text=True,
            timeout=10,
        ).splitlines()
        return {
            name: hashlib.sha256(
                subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repo, timeout=10)
            ).hexdigest()
            for name in names
            if name.endswith((".py", ".sh"))
        }
    return {
        str(path.relative_to(repo)): digest(path)
        for path in sorted((repo / "research/colab_cli").iterdir())
        if path.suffix in {".py", ".sh"} and path.is_file()
    }


def input_inventory(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe real inputs; public fetches cannot count as uploaded files."""
    return sorted(
        [
            {
                "dest": item["dest"],
                "size": (
                    Path(item["path"]).stat().st_size if "path" in item else item["size_bytes"]
                ),
                "transport": "upload" if "path" in item else "https",
                "sha256": item["sha256"],
            }
            for item in manifest.get("inputs", [])
        ],
        key=lambda row: row["dest"],
    )


def _exports(manifest: dict[str, Any]) -> set[str]:
    rows = manifest.get("required_exports")
    if not isinstance(rows, list) or not rows:
        raise CanaryError("family dispatch requires a nonempty required_exports producer inventory")
    return {row if isinstance(row, str) else row["path"] for row in rows}


def is_canary(manifest: dict[str, Any]) -> bool:
    return isinstance(manifest.get("canary"), dict) and manifest["canary"].get("mode") == "probe"


def _entrypoint(manifest: dict[str, Any], repo: Path) -> str:
    """Read device arguments from the pinned entrypoint, never a mutable script."""
    try:
        return subprocess.check_output(
            ["git", "show", f"{manifest['commit']}:{manifest['script']}"],
            cwd=repo,
            text=True,
            timeout=10,
            stderr=subprocess.PIPE,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise CanaryError("ambiguous job device: cannot read pinned entrypoint") from exc


def job_device(manifest: dict[str, Any], repo: Path | None = None) -> str:
    """Resolve explicit science devices, independently of rented hardware.

    A job can use CPU workers alongside CUDA inference, so any CUDA declaration
    requires CUDA proof. Missing, automatic, or unresolved device choices are
    refused rather than inferred from the allocation target. Entrypoints must
    declare their devices in ``device``, ``DEVICE``/``*_DEVICE`` environment
    variables, or ``--device`` arguments in environment values and the pinned
    entrypoint. Device choices hidden inside Python remain the author's explicit
    manifest contract; this function never executes shell code to guess them.
    """
    declared = []
    env = manifest.get("env", {})
    if "device" in manifest:
        declared.append(manifest["device"])
    arguments = []
    for key, value in env.items():
        if key == "DEVICE" or key.endswith("_DEVICE"):
            declared.append(value)
        # Some Bash entrypoints receive their CLI options through an env var.
        # Do not interpret shell expansions or evaluate the command.
        if isinstance(value, str) and "--device" in value:
            arguments.append((value, False))
    if repo is not None:
        arguments.append((_entrypoint(manifest, repo), True))
    for value, entrypoint in arguments:
        try:
            tokens = shlex.split(value.replace("\\\n", ""), comments=True)
        except ValueError as exc:
            raise CanaryError("ambiguous job device arguments") from exc
        assignments = {}
        if entrypoint:
            for token in tokens:
                assignment = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*)=(.*)", token)
                if assignment:
                    key, choice = assignment.groups()
                    assignments[key] = choice
                    if key == "DEVICE" or key.endswith("_DEVICE"):
                        declared.append(choice)
        for index, token in enumerate(tokens):
            if token == "--device":
                choice = tokens[index + 1] if index + 1 < len(tokens) else None
            elif token.startswith("--device="):
                choice = token.split("=", 1)[1]
            else:
                continue
            # Resolve only manifest-bound variables. Local shell variables,
            # defaults and computed branches are ambiguous, never executed.
            variable = re.fullmatch(
                r"\$(?:\{([A-Za-z_][A-Za-z_0-9]*)\}|([A-Za-z_][A-Za-z_0-9]*))", choice or ""
            )
            if variable:
                name = variable[1] or variable[2]
                if name in assignments:
                    # An entrypoint-local assignment is not the manifest's
                    # value. Without evaluating control flow it is ambiguous.
                    raise CanaryError("ambiguous job device: entrypoint reassigns device argument")
                choice = env.get(name)
            declared.append(choice)
    if not declared or any(
        not isinstance(value, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", value)
        for value in declared
    ):
        raise CanaryError("ambiguous job device: explicitly declare cpu or cuda[:index]")
    return "cuda" if any(value.startswith("cuda") for value in declared) else "cpu"


def check(manifest: dict[str, Any], repo: Path) -> None:
    """Admission gate called by validate, and consequently again before run."""
    try:
        transfer_policy.transfer_limit(manifest)
    except ValueError as exc:
        raise CanaryError(str(exc)) from exc
    f5_inventory = manifest.get("f5_timing_synthetic_inventory")
    if isinstance(f5_inventory, dict):
        try:
            tree = subprocess.check_output(
                ["git", "rev-parse", f"{manifest['commit']}^{{tree}}"],
                cwd=repo,
                text=True,
                timeout=10,
            ).strip()
        except (subprocess.SubprocessError, OSError) as exc:
            raise CanaryError("cannot resolve F5 timing transport source tree") from exc
        if tree != f5_inventory.get("source_tree"):
            raise CanaryError("F5 timing transfer authority source tree differs")
    if manifest["script"] == ACCEPTANCE_SCRIPT:
        # The source qualification probe must be able to bootstrap qualification.
        if manifest.get("canary"):
            raise CanaryError("acceptance probe cannot qualify a science family")
        return
    device = job_device(manifest, repo)
    expected_exports = _exports(manifest)
    request = manifest.get("canary", {})
    if not isinstance(request, dict):
        raise CanaryError("canary must be an object")
    if is_canary(manifest):
        if request.get("schema") != SCHEMA or request.get("family") != family(manifest):
            raise CanaryError("canary family/entrypoint/mode identity mismatch")
        if manifest["timeout_minutes"] > MAX_CANARY_MINUTES:
            raise CanaryError("canary whole-session cap must be <=30 minutes")
        if manifest.get("env", {}).get("SNAKE_COLAB_CANARY") != "1":
            raise CanaryError("canary instrumentation must be enabled")
        package = str(Path(manifest["script"]).parent)
        if manifest["env"].get("SNAKE_COLAB_CANARY_PACKAGE") != package:
            raise CanaryError("canary instrumentation package differs from real entrypoint")
        if executable_identity(repo, manifest["commit"]) != executable_identity(repo):
            raise CanaryError("canary commit does not contain the current transport inventory")
        return
    if manifest.get("env", {}).get("SNAKE_COLAB_CANARY"):
        raise CanaryError("science manifest cannot enable canary instrumentation")
    binding = request.get("receipt")
    if not isinstance(binding, dict) or not re.fullmatch(
        r"[0-9a-f]{64}", str(binding.get("sha256", ""))
    ):
        raise CanaryError("dispatch requires a SHA-bound family canary receipt")
    path = Path(binding.get("path", "")).expanduser()
    if not path.is_absolute() or not path.is_file() or digest(path) != binding["sha256"]:
        raise CanaryError("canary receipt is missing or its SHA256 differs")
    receipt = json.loads(path.read_text())
    saved = path.parent / "manifest.json"
    if not saved.is_file() or digest(saved) != receipt.get("manifest_sha256"):
        raise CanaryError("canary receipt manifest binding differs")
    probe = json.loads(saved.read_text())
    if not is_canary(probe):
        raise CanaryError("receipt is not from a family canary run")
    if job_device(probe, repo) != device:
        raise CanaryError("family canary science device differs from dispatch")
    proof = receipt.get("canary_evidence")
    if (
        receipt.get("status") != "PASS"
        or receipt.get("stop_verified") is not True
        or receipt.get("error")
        or receipt.get("cleanup_error")
        or receipt.get("failure_class") is not None
        or not isinstance(proof, dict)
    ):
        raise CanaryError("family canary did not complete with verified release")
    if (
        proof.get("schema") != SCHEMA
        or proof.get("family") != family(manifest)
        or proof.get("commit") != manifest["commit"]
        or probe["commit"] != manifest["commit"]
        or family(probe) != family(manifest)
        or probe.get("source_paths") != manifest.get("source_paths")
        or proof.get("transport_identity") != executable_identity(repo)
        or proof.get("transport_identity") != executable_identity(repo, manifest["commit"])
    ):
        raise CanaryError("family canary source commit or transport identity differs")
    if probe.get("transfer_exception") != manifest.get("transfer_exception"):
        raise CanaryError("family canary transfer exception authority differs from dispatch")
    if "f5_timing_transfer_exception" in manifest or "f5_timing_transfer_exception" in probe:
        # Only the context differs: a tiny family probe never becomes a full
        # timing operation. Inventory, ruling, source and bounds stay exact.
        main_authority = dict(manifest.get("f5_timing_transfer_exception", {}))
        probe_authority = dict(probe.get("f5_timing_transfer_exception", {}))
        if (
            main_authority.pop("context", None) != "timing"
            or probe_authority.pop("context", None) != "canary"
            or main_authority != probe_authority
        ):
            raise CanaryError("F5 timing family canary scoped authority differs")
        if probe.get("f5_timing_synthetic_inventory") != manifest.get(
            "f5_timing_synthetic_inventory"
        ):
            raise CanaryError("F5 timing family canary inventory authority differs")
    # Re-derive proof from the immutable broker receipt rather than trusting a
    # hand-written summary. Recovered exports are re-hashed on the Mac as well.
    actual = evidence(probe, receipt, path.parent, repo)
    if actual != proof:
        raise CanaryError("family canary evidence differs from broker observations")
    if manifest["requirements"]["gpu"] != proof["hardware"]:
        raise CanaryError("requested allocation target lacks a family canary")
    if proof["hardware"] not in manifest["requirements"].get(
        "allowed_gpus", [manifest["requirements"]["gpu"]]
    ):
        raise CanaryError("family canary target hardware differs")
    if manifest["requirements"].get("allowed_gpus", [manifest["requirements"]["gpu"]]) != [
        proof["hardware"]
    ]:
        raise CanaryError("every allowed target platform needs its own family canary")
    observed = {row["dest"]: row for row in proof["inputs"]}
    requested = input_inventory(manifest)
    if len(requested) != len(observed) or any(
        row["dest"] not in observed
        or observed[row["dest"]]["size"] < row["size"]
        or observed[row["dest"]]["transport"] != row["transport"]
        or observed[row["dest"]]["sha256"] != row["sha256"]
        for row in requested
    ):
        raise CanaryError("canary does not cover real upload bytes x file count/destinations")
    if not expected_exports <= set(proof["exports"]):
        raise CanaryError("canary scanner did not recover every required producer export")


def staging_measurement(manifest: dict[str, Any], measured: dict[str, Any]) -> dict[str, float]:
    """Validate persisted staging measurements before bootstrap or qualification.

    The broker calls this for probe canaries immediately after it records the
    measurements.  ``evidence`` calls the same function later when binding a
    completed receipt, so the timing contract has one implementation.
    """

    def duration(name: str) -> float:
        value = measured.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise CanaryError(f"invalid {name} staging measurement")
        return float(value)

    elapsed = duration("elapsed_seconds")
    transfer_seconds = duration("transfer_seconds")
    source_delivery_seconds = duration("source_delivery_seconds")
    if elapsed <= 0:
        raise CanaryError("staging elapsed_seconds must be positive")
    if elapsed > MAX_STAGING_SECONDS:
        raise StagingCapError("staging must complete within 300 seconds")
    if transfer_seconds <= 0:
        raise CanaryError("transfer_seconds must be positive")
    try:
        transfer_cap = transfer_policy.transfer_limit(manifest)
    except ValueError as exc:
        raise CanaryError(str(exc)) from exc
    if transfer_seconds > transfer_cap:
        raise StagingCapError(f"transfers must complete within {transfer_cap:g} seconds")
    if measured.get("staging_completed") is False:
        raise CanaryError("partial staging measurements cannot qualify transport")
    if source_delivery_seconds < 0:
        raise CanaryError("source_delivery_seconds must be nonnegative")
    # The total includes setup/renewals as well as both measured portions.
    # Permit only rounding noise from independent monotonic-clock reads.
    if (
        transfer_seconds > elapsed
        or source_delivery_seconds > elapsed
        or transfer_seconds + source_delivery_seconds > elapsed + 1e-6
    ):
        raise CanaryError("staging timing components exceed elapsed_seconds")
    return {
        "elapsed_seconds": elapsed,
        "transfer_seconds": transfer_seconds,
        "source_delivery_seconds": source_delivery_seconds,
    }


def evidence(
    manifest: dict[str, Any], receipt: dict[str, Any], output: Path, repo: Path
) -> dict[str, Any]:
    """Construct proof only from measured transfers, scanner output and CUDA calls."""
    measured = staging_measurement(
        manifest, json.loads((output / "upload-measurement.json").read_text())
    )
    elapsed = measured["elapsed_seconds"]
    transfer_seconds = measured["transfer_seconds"]
    source_delivery_seconds = measured["source_delivery_seconds"]
    terminal = receipt.get("terminal") or {}
    if (
        receipt.get("commit") != manifest["commit"]
        or receipt.get("job_id") != manifest["job_id"]
        or terminal.get("job_id") != manifest["job_id"]
        or terminal.get("generation") != receipt.get("generation")
        or not receipt.get("generation")
        or terminal.get("returncode") != 0
        or terminal.get("status") not in {"completed", "success", "succeeded", "PASS"}
    ):
        raise CanaryError("canary terminal job/source/generation identity differs")
    scanner = terminal.get("export_scanner") or {}
    exports = {row["path"]: row for row in receipt.get("exports", [])}
    scanned = scanner.get("inventory", [])
    if scanner.get("name") != "runtime._exports" or sorted(
        scanned, key=lambda row: row["path"]
    ) != sorted(receipt.get("exports", []), key=lambda row: row["path"]):
        raise CanaryError("canary lacks the actual runtime export scanner inventory")
    if not _exports(manifest) <= set(exports):
        raise CanaryError("canary is missing required producer sidecars/exports")
    for name, row in exports.items():
        path = output / "exports" / name
        if (
            not path.is_file()
            or path.stat().st_size != row["size"]
            or digest(path) != row["sha256"]
        ):
            raise CanaryError("canary recovered export differs")
    inputs = input_inventory(manifest)
    uploads = {
        row["remote"].split("/inputs/", 1)[1]: row
        for row in receipt.get("transfers", [])
        if row.get("direction") == "upload"
        and "/inputs/" in row.get("remote", "")
        and row.get("status") == "verified"
    }
    for row in inputs:
        if row["transport"] == "upload":
            observed = uploads.get(row["dest"], {})
            if observed.get("size_bytes") != row["size"] or observed.get("sha256") != row["sha256"]:
                raise CanaryError("canary lacks a real verified upload for every input")
    hardware = json.loads((output / "hardware.json").read_text())["hardware"]["kind"]
    decisions = terminal.get("canary_observations", [])
    package = str(Path(manifest["script"]).parent)
    valid_decisions = [
        row
        for row in decisions
        if (
            type(row.get("cuda_decisions")) is int
            and row["cuda_decisions"] > 0
            and str(row.get("device", "")).startswith("cuda")
            and row.get("package") == package
            and str(row.get("caller", "")).startswith(package + "/")
            and row.get("operation") in {"Tensor.argmax", "torch.argmax", "torch.multinomial"}
            and isinstance(row.get("shape"), list)
            and row["shape"]
            and row["shape"][-1] == 6
            and row.get("root") == "/content/snake-cli/" + str(receipt.get("generation", ""))
        )
    ]
    device = job_device(manifest, repo)
    observed_cuda = any(str(row.get("device", "")).startswith("cuda") for row in decisions)
    if (device == "cuda" or observed_cuda) and not valid_decisions:
        raise CanaryError("GPU family canary lacks a real CUDA action decision from its entrypoint")
    return {
        "schema": SCHEMA,
        "family": family(manifest),
        "commit": manifest["commit"],
        "transport_identity": executable_identity(repo),
        "inputs": inputs,
        "upload_elapsed_seconds": elapsed,
        "transfer_seconds": transfer_seconds,
        "source_delivery_seconds": source_delivery_seconds,
        "exports": sorted(exports),
        "hardware": hardware,
        "device": device,
        "cuda_observations": decisions,
        **(
            {"transfer_exception": manifest["transfer_exception"]}
            if "transfer_exception" in manifest
            else {}
        ),
    }


def prepare(
    template: dict[str, Any],
    *,
    package: str,
    output: Path,
    job_id: str,
    env: dict[str, str] | None = None,
    timeout_minutes: float = 30,
) -> dict[str, Any]:
    """Keep real transport and entrypoint; reduce only normal package workload knobs."""
    probe = json.loads(json.dumps(template))
    if probe["script"] not in FAMILIES.get(package, ()):
        raise CanaryError(f"entrypoint is not part of package family {package}")
    _exports(probe)
    timeout_minutes = min(timeout_minutes, probe["timeout_minutes"])
    probe.update(job_id=job_id, output_dir=str(output), timeout_minutes=timeout_minutes)
    probe.pop("canary", None)
    probe["env"] = {
        **probe.get("env", {}),
        **(env or {}),
        "SNAKE_COLAB_CANARY": "1",
        "SNAKE_COLAB_CANARY_PACKAGE": str(Path(probe["script"]).parent),
    }
    probe["requirements"]["est_minutes"] = min(
        probe["requirements"]["est_minutes"], timeout_minutes / 2
    )
    probe["canary"] = {"schema": SCHEMA, "mode": "probe", "family": family(probe)}
    return probe


def main(package: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    if package is None:
        parser.add_argument("--package", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--env-json", type=Path, help="normal entrypoint knobs for a small excluded workload"
    )
    parser.add_argument("--timeout-minutes", type=float, default=30)
    args = parser.parse_args()
    probe = prepare(
        json.loads(args.template.read_text()),
        package=package or args.package,
        output=args.output,
        job_id=args.job_id,
        env=json.loads(args.env_json.read_text()) if args.env_json else None,
        timeout_minutes=args.timeout_minutes,
    )
    from research.colab_cli.supervisor import validate, write_once

    validated = validate(probe)
    write_once(args.manifest, validated)
    write_once(
        args.manifest.with_suffix(args.manifest.suffix + ".validation.json"),
        {"manifest_sha256": digest(args.manifest), "status": "VALIDATED_NOT_DISPATCHED"},
    )


if __name__ == "__main__":
    main()
