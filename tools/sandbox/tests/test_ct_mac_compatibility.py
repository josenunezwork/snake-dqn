"""Exact reseal-03 broker/runtime compatibility with committed policy and fault twins."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools.sandbox import ct_mac_compatibility as compat
from tools.sandbox import preflight


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.fixture
def compatible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "Compatibility Test")
    git("config", "user.email", "compat@example.invalid")
    monkeypatch.setattr(preflight, "REPO_ROOT", root)
    real_run = preflight._run
    monkeypatch.setattr(preflight, "_run", lambda command: real_run(command, cwd=root))
    policy_path = root / "policy.json"
    monkeypatch.setattr(compat, "POLICY", policy_path)
    packet = root / "initial"
    manifest = {
        "commit": "a" * 40,
        "sandbox_admission": {"manifest_path": str(packet / "sandbox.json")},
    }
    dump(packet / "dispatch-manifest.json", manifest)

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    dump(
        packet / "sandbox.json",
        {
            "outputs": {
                "broker_manifest": {
                    "path": str(packet / "dispatch-manifest.json"),
                    "sha256": sha(packet / "dispatch-manifest.json"),
                }
            }
        },
    )
    dump(packet / "SEAL.json", {"manifest_sha256": sha(packet / "dispatch-manifest.json")})
    policy = {
        "schema": "t34-cf-mac-source-compatibility/v1",
        "runtime_commit": "a" * 40,
        "root_manifest_sha256": sha(packet / "dispatch-manifest.json"),
        "root_seal_sha256": sha(packet / "SEAL.json"),
        "mac_only_sources": {
            name: {
                "runtime_sha256": hashlib.sha256(b"old").hexdigest(),
                "broker_sha256": hashlib.sha256(b"new").hexdigest(),
            }
            for name in compat.MAC_ONLY
        },
    }
    dump(policy_path, policy)
    git("add", ".")
    git("commit", "-qm", "freeze compatibility")
    return dict(
        root=root,
        packet=packet,
        manifest=manifest,
        policy=policy,
        policy_path=policy_path,
        git=git,
        sha=sha,
    )


@pytest.mark.parametrize("index", [0, 1, 2])
def test_exact_compatibility_across_two_hops(compatible: dict, index: int) -> None:
    f = compatible
    manifest = f["manifest"]
    prior = f["packet"] / "dispatch-manifest.json"
    for i in range(1, index + 1):
        manifest = {
            "commit": "a" * 40,
            "completion_attempt": {
                "index": i,
                "prior_manifest_path": str(prior),
                "prior_manifest_sha256": f["sha"](prior),
            },
        }
        prior = f["root"] / f"c{i}/dispatch-manifest.json"
        dump(prior, manifest)
    for name in compat.MAC_ONLY:
        assert compat.permits_mac_source(manifest, name, b"old", b"new")
        assert not compat.permits_mac_source(manifest, name, b"old", b"foreign")
        assert not compat.permits_mac_source(manifest, name, b"foreign", b"new")
    assert not compat.permits_mac_source(
        manifest, "research/sandbox_a_resident/ct_run.py", b"old", b"new"
    )


@pytest.mark.parametrize("fault", [None, "manifest", "seal", "uncommitted"])
def test_additional_root_requires_exact_committed_pair(compatible: dict, fault: str | None) -> None:
    f = compatible
    pair = {key: f["policy"][key] for key in ("root_manifest_sha256", "root_seal_sha256")}
    f["policy"].update(root_manifest_sha256="b" * 64, root_seal_sha256="c" * 64)
    if fault in ("manifest", "seal"):
        pair[f"root_{fault}_sha256"] = "d" * 64
    f["policy"]["additional_roots"] = [pair]
    dump(f["policy_path"], f["policy"])
    if fault != "uncommitted":
        f["git"]("add", ".")
        f["git"]("commit", "-qm", "bind additional exact root")
    if fault == "uncommitted":
        with pytest.raises(preflight.PreflightError):
            compat.permits_mac_source(f["manifest"], next(iter(compat.MAC_ONLY)), b"old", b"new")
    else:
        assert compat.permits_mac_source(
            f["manifest"], next(iter(compat.MAC_ONLY)), b"old", b"new"
        ) == (fault is None)


@pytest.mark.parametrize(
    "fault", ["runtime", "policy-hash", "seal", "root", "uncommitted", "c3", "cycle", "symlink"]
)
def test_compatibility_fault_twins(compatible: dict, fault: str) -> None:
    f = compatible
    manifest = f["manifest"]
    if fault == "runtime":
        manifest = {**manifest, "commit": "b" * 40}
    elif fault == "policy-hash":
        f["policy"]["root_manifest_sha256"] = "b" * 64
        dump(f["policy_path"], f["policy"])
        f["git"]("add", ".")
        f["git"]("commit", "-qm", "fault policy")
    elif fault == "seal":
        (f["packet"] / "SEAL.json").write_text("{}")
    elif fault == "root":
        (f["packet"] / "dispatch-manifest.json").write_text("{}")
    elif fault == "uncommitted":
        f["policy_path"].write_text("{}")
    elif fault == "symlink":
        raw = f["policy_path"].read_bytes()
        f["policy_path"].unlink()
        target = f["root"] / "foreign.json"
        target.write_bytes(raw)
        f["policy_path"].symlink_to(target)
    else:
        path = f["packet"] / "dispatch-manifest.json"
        manifest = {
            "commit": "a" * 40,
            "completion_attempt": {
                "index": 3 if fault == "c3" else 1,
                "prior_manifest_path": str(path),
                "prior_manifest_sha256": "b" * 64,
            },
        }
    try:
        assert not compat.permits_mac_source(manifest, next(iter(compat.MAC_ONLY)), b"old", b"new")
    except preflight.PreflightError:
        assert fault in ("uncommitted", "symlink")


@pytest.mark.parametrize(
    "fault", [None, "foreign-path", "foreign-head", "seal", "component", "symlink"]
)
def test_external_root_commit_and_component_binding(compatible: dict, fault: str | None) -> None:
    f = compatible
    external = f["root"].parent / "original-broker"
    packet = external / "initial"
    packet.mkdir(parents=True)
    for name in ("dispatch-manifest.json", "SEAL.json"):
        (packet / name).write_bytes((f["packet"] / name).read_bytes())
    dump(packet / "control.json", {"frozen": True})

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(external), *args])

    git("init", "-q")
    git("config", "user.name", "Original Broker")
    git("config", "user.email", "original@example.invalid")
    git("add", ".")
    git("commit", "-qm", "original packet")
    f["policy"].update(
        root_broker_checkout=str(external),
        root_packet_directory=str(packet),
        root_broker_commit=git("rev-parse", "HEAD").decode().strip(),
    )
    dump(f["policy_path"], f["policy"])
    f["git"]("add", ".")
    f["git"]("commit", "-qm", "bind original external packet")
    path = packet / "control.json"
    if fault == "foreign-path":
        path = f["packet"] / "control.json"
        dump(path, {"frozen": True})
    elif fault == "foreign-head":
        git("commit", "--allow-empty", "-qm", "foreign commit")
    elif fault == "seal":
        (packet / "SEAL.json").write_text("{}")
    elif fault == "component":
        path.write_text("{}")
    elif fault == "symlink":
        path.unlink()
        path.symlink_to(packet / "dispatch-manifest.json")
    if fault is None:
        compat.verify_sealed_root_component(path)
    else:
        with pytest.raises(preflight.PreflightError):
            compat.verify_sealed_root_component(path)


@pytest.mark.parametrize("fault", [None, "root", "uncommitted"])
def test_transport_successor_has_separate_exact_policy(compatible: dict, fault: str | None) -> None:
    f = compatible
    manifest = {**f["manifest"], "commit": "b" * 40}
    dump(f["packet"] / "dispatch-manifest.json", manifest)
    dump(
        f["packet"] / "sandbox.json",
        {
            "outputs": {
                "broker_manifest": {
                    "path": str(f["packet"] / "dispatch-manifest.json"),
                    "sha256": f["sha"](f["packet"] / "dispatch-manifest.json"),
                }
            }
        },
    )
    dump(
        f["packet"] / "SEAL.json",
        {"manifest_sha256": f["sha"](f["packet"] / "dispatch-manifest.json")},
    )
    policy = {
        **f["policy"],
        "runtime_commit": "b" * 40,
        "root_manifest_sha256": f["sha"](f["packet"] / "dispatch-manifest.json"),
        "root_seal_sha256": f["sha"](f["packet"] / "SEAL.json"),
    }
    if fault == "root":
        policy["root_manifest_sha256"] = "c" * 64
    path = f["policy_path"].with_name("ct_mac_compatibility_transport01.json")
    dump(path, policy)
    if fault != "uncommitted":
        f["git"]("add", ".")
        f["git"]("commit", "-qm", "exact transport successor compatibility")
    if fault == "uncommitted":
        with pytest.raises(preflight.PreflightError):
            compat.permits_mac_source(manifest, next(iter(compat.MAC_ONLY)), b"old", b"new")
    else:
        assert compat.permits_mac_source(manifest, next(iter(compat.MAC_ONLY)), b"old", b"new") == (
            fault is None
        )
