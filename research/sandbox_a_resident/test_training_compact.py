"""Compact lossless transport for the train01-11 iteration-0000 full-state snapshot."""

from __future__ import annotations

import copy
import io
import json
import random
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

from research.sandbox_a_resident import training_units
from research.sandbox_a_resident.training_compact import (
    CompactResumeRefused,
    decode_stream,
    decode_to_bytes,
    describe_state,
    encode_snapshot,
    load_manifest,
    materialize_compact_resume,
    verify_decoded_state,
)
from research.sandbox_a_resident.training_preflight import validate_uploaded_ledger
from research.sandbox_a_resident.training_resume import (
    ResumeRefused,
    attach_compact_resume,
    build_ledger,
)
from research.sandbox_a_resident.training_units import (
    COMPACT_ALLOWLIST_SCHEMA,
    COMPACT_RESUME_ALLOWLIST,
    TRAIN01_15_SOURCE_COMMIT,
    TRAIN01_16_SOURCE_COMMIT,
    allocation,
    load_ledger,
    register_compact_allowlist_entry,
    unit_id,
    validate_compact_resume,
)
from research.sandbox_a_resident.units import digest, publish

REAL_EXPORTS = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/"
    "train01-attempt-11/output/exports/training"
)
REAL_ORIGINAL = REAL_EXPORTS / "iteration-0000.pt"
REAL_DERIVATIVE = REAL_EXPORTS / "iteration-0000.pt.xz"
REAL_MANIFEST = REAL_EXPORTS / "iteration-0000.pt.xz.manifest"


def control() -> dict:
    return {"study": "sandbox-a-train01", "source_commit": "a" * 40, "outer_iterations": 4}


def sample_state() -> dict:
    """A state exercising every container/scalar kind a TRAIN-01 snapshot uses."""
    generator = torch.Generator().manual_seed(3)
    bodies = torch.zeros((4, 3, 64, 2), dtype=torch.int64)
    bodies[0, 0, :5] = torch.arange(10).reshape(5, 2)
    return {
        "schema": "sandbox-a-train-state/v1",
        "iteration": 0,
        "actor": OrderedDict(
            [
                ("weight", torch.randn(8, 4, generator=generator)),
                ("_temperature", torch.tensor(0.0084397)),
            ]
        ),
        "trainer": {"actor_rows": 0, "popart": {"mu": torch.zeros(1), "beta": 0.999}},
        "task": {
            "sim": {"bodies": bodies, "food_present": torch.zeros(4, 9, dtype=torch.bool)},
            "groups": {
                2: {"random_slots": [torch.randint(0, 3, (3, 5, 3), generator=generator), None]}
            },
        },
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": [],
            "numpy": ("MT19937", np.arange(624, dtype=np.uint32), 624, 0, 0.0),
            "python": random.Random(5).getstate(),
            "collection": generator.get_state(),
        },
        "metrics": [{"iteration": 0, "loss": -0.0, "ok": True, "name": "prep"}],
        "preparation": {"temperature": 0.008439731784164906, "stage": "ready"},
        "action_window": torch.empty(0, dtype=torch.long),
    }


def write_snapshot(root: Path, state: dict | None = None) -> Path:
    """Publish one canonical completion-last snapshot triple with a real torch payload."""
    root.mkdir(parents=True, exist_ok=True)
    state = sample_state() if state is None else state
    payload = publish(root / "iteration-0000.pt", lambda path: torch.save(state, path))
    info = {
        "schema": "sandbox-a-train-snapshot/v1",
        "id": unit_id(0),
        "iteration": 0,
        "snapshot": {"id": unit_id(0), **payload},
        "metadata": {
            "allocation": allocation(control()),
            "source_commit": control()["source_commit"],
            "phase": "pre-actor",
            "prepare_phase": "ready",
        },
    }
    publish(root / "iteration-0000.json", lambda path: path.write_text(json.dumps(info)))
    publish(root / "iteration-0000-actor.pth", lambda path: path.write_bytes(b"actor"))
    return root / "iteration-0000.pt"


def descriptor(path: Path, runtime: str) -> dict:
    return {"path": runtime, "size": path.stat().st_size, "sha256": digest(path)}


def allowlist_for(original: Path) -> dict:
    return {
        unit_id(0): {
            "original": descriptor(original, "resume/iteration-0000.pt"),
            "derivative": descriptor(Path(f"{original}.xz"), "resume/iteration-0000.pt.xz"),
            "manifest": descriptor(
                Path(f"{original}.xz.manifest"), "resume/iteration-0000.pt.xz.manifest"
            ),
        }
    }


@pytest.fixture
def compact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An encoded synthetic snapshot, its allowlist, packet ledger and runtime inputs."""
    original = write_snapshot(tmp_path / "exports")
    encode_snapshot(original, unit=unit_id(0))
    allowlist = allowlist_for(original)
    monkeypatch.setattr(training_units, "COMPACT_RESUME_ALLOWLIST", allowlist)
    ledger, uploads, latest = build_ledger(original.parent, control())
    assert latest == 0
    ledger, uploads = attach_compact_resume(ledger, uploads, original.parent)
    inputs = tmp_path / "inputs"
    for row in uploads:
        (inputs / row["dest"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["path"], inputs / row["dest"])
    return original, ledger, uploads, inputs


def test_encode_is_lossless_create_only_and_deterministic(tmp_path: Path) -> None:
    original = write_snapshot(tmp_path / "a")
    result = encode_snapshot(original, unit=unit_id(0))
    derivative = Path(result["derivative"]["path"])
    manifest_path = Path(result["manifest"]["path"])
    for path in (derivative, manifest_path):
        assert Path(f"{path}.sha256").read_text().strip() == digest(path)
        assert Path(f"{path}.complete").read_text() == "complete\n"
    decoded = decode_to_bytes(
        derivative, expected_size=original.stat().st_size, expected_sha256=digest(original)
    )
    assert decoded == original.read_bytes()
    with pytest.raises(FileExistsError):
        encode_snapshot(original, unit=unit_id(0))
    # A byte-identical original re-encodes to byte-identical derivative and manifest.
    twin = tmp_path / "b"
    twin.mkdir()
    for suffix in ("", ".sha256", ".complete"):
        shutil.copy2(f"{original}{suffix}", twin / f"iteration-0000.pt{suffix}")
    again = encode_snapshot(twin / "iteration-0000.pt", unit=unit_id(0))
    assert again["derivative"]["sha256"] == result["derivative"]["sha256"]
    assert again["manifest"]["sha256"] == result["manifest"]["sha256"]
    manifest = json.loads(manifest_path.read_text())
    state = torch.load(original, map_location="cpu", weights_only=False)
    assert {key: manifest[key] for key in ("structure_sha256", "tensor_count", "tensors")} == {
        key: describe_state(state)[key] for key in ("structure_sha256", "tensor_count", "tensors")
    }
    verify_decoded_state(
        torch.load(io.BytesIO(decoded), map_location="cpu", weights_only=False), manifest
    )


def test_encode_refuses_corrupt_original(tmp_path: Path) -> None:
    original = write_snapshot(tmp_path)
    Path(f"{original}.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(CompactResumeRefused, match="corrupt original"):
        encode_snapshot(original, unit=unit_id(0))
    assert not Path(f"{original}.xz").exists()


def _mutate_weight_one_ulp(state: dict) -> None:
    """Change a finite weight's bytes even when a fixed epsilon would round away."""
    weight = state["actor"]["weight"].view(-1)[3]
    before = weight.clone()
    assert torch.isfinite(weight)
    weight.copy_(torch.nextafter(weight, torch.full_like(weight, float("inf"))))
    assert not torch.equal(before.reshape(1).view(torch.uint8), weight.reshape(1).view(torch.uint8))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: _mutate_weight_one_ulp(s),
        lambda s: s["task"]["sim"]["bodies"].view(-1)[-1].fill_(1),
        lambda s: s["task"]["sim"].__setitem__(
            "food_present", s["task"]["sim"]["food_present"].to(torch.uint8)
        ),
        lambda s: s["task"]["sim"].__setitem__("bodies", s["task"]["sim"]["bodies"][:3]),
        lambda s: s["task"]["sim"].__setitem__(
            "bodies", s["task"]["sim"]["bodies"].transpose(0, 1)
        ),
        lambda s: s["trainer"].__setitem__("extra", torch.zeros(1)),
        lambda s: s["trainer"]["popart"].__setitem__("beta", 0.9990000000000001),
        lambda s: s["metrics"][0].__setitem__("loss", 0.0),
        lambda s: s["rng"]["numpy"][1].__setitem__(0, 7),
        lambda s: s["rng"].__setitem__("python", s["rng"]["python"][:2] + (0.5,)),
        lambda s: s.__setitem__("actor", dict(s["actor"])),
        lambda s: s["preparation"].__setitem__("stage", "skipped"),
        lambda s: s["actor"]["weight"].requires_grad_(True),
    ],
)
def test_verify_decoded_state_refuses_any_tensor_or_structure_difference(mutate) -> None:
    state = sample_state()
    manifest = describe_state(state)
    verify_decoded_state(copy.deepcopy(state), manifest)
    changed = copy.deepcopy(state)
    mutate(changed)
    with pytest.raises(CompactResumeRefused):
        verify_decoded_state(changed, manifest)


@pytest.mark.parametrize("value", [2.0, 3.0, -3.0])
def test_verify_decoded_state_refuses_one_ulp_when_fixed_epsilon_rounds_away(value: float) -> None:
    """Reproduce the admission flake and require rejection of an actual tiny change."""
    state = sample_state()
    state["actor"]["weight"].view(-1)[3] = value
    manifest = describe_state(state)
    unchanged = copy.deepcopy(state)
    unchanged["actor"]["weight"].view(-1)[3].add_(1e-7)
    assert torch.equal(unchanged["actor"]["weight"], state["actor"]["weight"])
    verify_decoded_state(unchanged, manifest)
    changed = copy.deepcopy(state)
    _mutate_weight_one_ulp(changed)
    # Deepcopy must isolate the mutation from the original and its frozen manifest.
    assert state["actor"]["weight"].view(-1)[3].item() == value
    assert describe_state(state) == manifest
    with pytest.raises(CompactResumeRefused, match="decoded tensor differs"):
        verify_decoded_state(changed, manifest)


@pytest.mark.parametrize("bits", [0, -2147483648, 0x7FC00000, 0x7FC00001, 0x7F800000])
def test_verify_decoded_state_compares_float_bits_including_zero_and_nan(bits: int) -> None:
    """Numeric equality and NaN behavior must not hide changed tensor bytes."""
    original = {"weight": torch.tensor([bits], dtype=torch.int32).view(torch.float32)}
    manifest = describe_state(original)
    changed = copy.deepcopy(original)
    changed["weight"].view(torch.int32).bitwise_xor_(1)
    assert not torch.equal(
        original["weight"].view(torch.uint8), changed["weight"].view(torch.uint8)
    )
    with pytest.raises(CompactResumeRefused, match="decoded tensor differs"):
        verify_decoded_state(changed, manifest)


def test_describe_state_refuses_unsupported_types() -> None:
    with pytest.raises(CompactResumeRefused, match="unsupported state type"):
        describe_state({"bad": {1, 2}})
    with pytest.raises(CompactResumeRefused, match="key type"):
        describe_state({(1, 2): 3})
    with pytest.raises(CompactResumeRefused, match="unsupported state type"):
        describe_state({"param": torch.nn.Parameter(torch.zeros(1))})


def test_decoder_refuses_tampered_truncated_trailing_and_oversized_streams(tmp_path: Path) -> None:
    original = write_snapshot(tmp_path / "x")
    encode_snapshot(original, unit=unit_id(0))
    derivative = Path(f"{original}.xz")
    size, sha = original.stat().st_size, digest(original)
    raw = derivative.read_bytes()
    cases = {
        "flip": raw[:40] + bytes([raw[40] ^ 0x01]) + raw[41:],
        "truncated": raw[:-8],
        "trailing": raw + b"\x00",
    }
    for name, payload in cases.items():
        bad = tmp_path / f"{name}.xz"
        bad.write_bytes(payload)
        with pytest.raises(CompactResumeRefused):
            decode_stream(bad, lambda block: None, expected_size=size, expected_sha256=sha)
    with pytest.raises(CompactResumeRefused, match="exceeds its original size"):
        decode_stream(derivative, lambda block: None, expected_size=size - 1, expected_sha256=sha)
    with pytest.raises(CompactResumeRefused, match="differs from the original"):
        decode_stream(derivative, lambda block: None, expected_size=size, expected_sha256="0" * 64)


def test_materialize_decodes_byte_identical_original_create_only(compact) -> None:
    original, ledger, _uploads, inputs = compact
    target = inputs / "resume/iteration-0000.pt"
    assert not target.exists()
    manifest = materialize_compact_resume(ledger, inputs)
    assert target.read_bytes() == original.read_bytes()
    assert not Path(f"{target}.partial").exists()
    assert manifest["original"]["sha256"] == digest(original)
    # The ordinary runtime ledger verification now accepts the decoded original.
    path = inputs / "verified-ledger.json"
    path.write_text(json.dumps(ledger))
    verified = load_ledger(path, control(), input_root=inputs)
    assert Path(verified[0]["exports"][0]["path"]) == target.resolve()
    state = torch.load(target, map_location="cpu", weights_only=False)
    verify_decoded_state(state, manifest)
    with pytest.raises(CompactResumeRefused, match="already present"):
        materialize_compact_resume(ledger, inputs)


def test_materialize_refuses_tampered_derivative_without_publishing(compact) -> None:
    _original, ledger, _uploads, inputs = compact
    derivative = inputs / "resume/iteration-0000.pt.xz"
    raw = bytearray(derivative.read_bytes())
    raw[len(raw) // 2] ^= 0x10
    derivative.write_bytes(bytes(raw))
    with pytest.raises(CompactResumeRefused, match="differs from its binding"):
        materialize_compact_resume(ledger, inputs)
    assert not (inputs / "resume/iteration-0000.pt").exists()
    assert not (inputs / "resume/iteration-0000.pt.partial").exists()


def test_materialize_refuses_tampered_manifest_or_preexisting_original(compact) -> None:
    _original, ledger, _uploads, inputs = compact
    manifest = inputs / "resume/iteration-0000.pt.xz.manifest"
    saved = manifest.read_bytes()
    manifest.write_bytes(saved.replace(b'"tensor_count": ', b'"tensor_count":  '))
    with pytest.raises(CompactResumeRefused, match="differs from its binding"):
        materialize_compact_resume(ledger, inputs)
    manifest.write_bytes(saved)
    (inputs / "resume/iteration-0000.pt").write_bytes(b"planted")
    with pytest.raises(CompactResumeRefused, match="already present"):
        materialize_compact_resume(ledger, inputs)


def test_manifest_declarations_must_match_binding(compact) -> None:
    original, ledger, _uploads, inputs = compact
    manifest = json.loads(Path(f"{original}.xz.manifest").read_text())
    manifest["original"]["sha256"] = "1" * 64
    forged = inputs / "resume/forged.manifest"
    forged.write_text(json.dumps(manifest))
    entry = copy.deepcopy(ledger["compact_resume"])
    entry["manifest"] = descriptor(forged, entry["manifest"]["path"])
    with pytest.raises(CompactResumeRefused, match="does not match its binding"):
        load_manifest(forged, entry)


def test_ledger_compact_binding_is_allowlist_exact(compact, monkeypatch) -> None:
    _original, ledger, _uploads, _inputs = compact
    assert validate_compact_resume(ledger) == ledger["compact_resume"]
    for role in ("original", "derivative", "manifest"):
        forged = copy.deepcopy(ledger)
        forged["compact_resume"][role]["sha256"] = "f" * 64
        with pytest.raises(ValueError, match="frozen allowlist"):
            validate_compact_resume(forged)
    foreign = copy.deepcopy(ledger)
    foreign["compact_resume"]["unit"] = unit_id(16)
    with pytest.raises(ValueError, match="not allowlisted"):
        validate_compact_resume(foreign)
    extra = copy.deepcopy(ledger)
    extra["units"].append(copy.deepcopy(extra["units"][0]))
    with pytest.raises(ValueError, match="sole full-state export"):
        validate_compact_resume(extra)
    # The production allowlist names only train01-11 iteration-0000 and refuses the fixture.
    monkeypatch.setattr(training_units, "COMPACT_RESUME_ALLOWLIST", COMPACT_RESUME_ALLOWLIST)
    assert set(COMPACT_RESUME_ALLOWLIST) == {unit_id(0)}
    with pytest.raises(ValueError, match="frozen allowlist"):
        validate_compact_resume(ledger)


def test_attach_replaces_full_state_upload_and_mac_preflight_binds_it(
    compact, tmp_path: Path
) -> None:
    original, ledger, uploads, _inputs = compact
    destinations = [row["dest"] for row in uploads]
    assert "resume/iteration-0000.pt" not in destinations
    assert destinations[:4] == [
        "resume/iteration-0000.pt.xz",
        "resume/iteration-0000.pt.xz.manifest",
        "resume/iteration-0000.pt.sha256",
        "resume/iteration-0000.pt.complete",
    ]
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "verified-ledger.json").write_text(json.dumps(ledger))
    broker = {"inputs": [{"dest": "b8-s3.pth", "path": "/x", "sha256": "0" * 64}, *uploads]}
    validate_uploaded_ledger(packet, broker, control())

    with_full = copy.deepcopy(broker)
    with_full["inputs"].append(
        {"dest": "resume/iteration-0000.pt", "path": str(original), "sha256": digest(original)}
    )
    with pytest.raises(ValueError, match="must not also upload"):
        validate_uploaded_ledger(packet, with_full, control())
    missing = copy.deepcopy(broker)
    missing["inputs"] = [r for r in missing["inputs"] if not r["dest"].endswith(".xz.manifest")]
    with pytest.raises(ValueError, match="manifest is not a hash-bound upload"):
        validate_uploaded_ledger(packet, missing, control())
    foreign = copy.deepcopy(broker)
    for row in foreign["inputs"]:
        if row["dest"].endswith(".pt.xz"):
            row["sha256"] = "e" * 64
    with pytest.raises(ValueError, match="derivative is not a hash-bound upload"):
        validate_uploaded_ledger(packet, foreign, control())
    no_sidecar = copy.deepcopy(broker)
    no_sidecar["inputs"] = [
        r for r in no_sidecar["inputs"] if r["dest"] != "resume/iteration-0000.pt.complete"
    ]
    with pytest.raises(ValueError, match="exact uploaded siblings"):
        validate_uploaded_ledger(packet, no_sidecar, control())


def test_attach_refuses_unallowlisted_or_incomplete_derivative(tmp_path: Path, monkeypatch) -> None:
    original = write_snapshot(tmp_path / "exports")
    ledger, uploads, _latest = build_ledger(original.parent, control())
    monkeypatch.setattr(training_units, "COMPACT_RESUME_ALLOWLIST", {})
    with pytest.raises(ResumeRefused, match="not allowlisted"):
        attach_compact_resume(ledger, uploads, original.parent)
    encode_snapshot(original, unit=unit_id(0))
    monkeypatch.setattr(training_units, "COMPACT_RESUME_ALLOWLIST", allowlist_for(original))
    Path(f"{original}.xz.complete").unlink()
    with pytest.raises(ResumeRefused, match="incomplete export"):
        attach_compact_resume(ledger, uploads, original.parent)


@pytest.mark.parametrize(
    ("source_packet_id", "source_commit", "iteration"),
    [
        ("packet-train01-15", TRAIN01_15_SOURCE_COMMIT, 192),
        ("packet-train01-16", TRAIN01_16_SOURCE_COMMIT, 176),
        ("packet-train01-17", "e" * 40, 176),
        ("packet-train01-s2-01", "1" * 40, 64),
        ("packet-train01-s2-01-c1-i0064", "2" * 40, 176),
    ],
)
def test_v2_periodic_checkpoint_compact_encode_attach_preflight_and_materialize(
    tmp_path: Path, source_packet_id: str, source_commit: str, iteration: int
) -> None:
    src_control = {
        "study": "sandbox-a-train01",
        "source_commit": source_commit,
        "outer_iterations": 384,
        "snapshot_interval": 16,
    }
    dst_control = {**src_control, "source_commit": "f" * 40}
    export_dir = tmp_path / "exports"
    export_dir.mkdir(parents=True)
    state = sample_state()
    state["iteration"] = iteration
    # Exercise non-contiguous transposed/sliced tensors in periodic checkpoints.
    state["non_contiguous_view"] = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()[1:5]
    assert not state["non_contiguous_view"].is_contiguous()

    stem = f"iteration-{iteration:04d}"
    payload = publish(export_dir / f"{stem}.pt", lambda path: torch.save(state, path))
    info = {
        "schema": "sandbox-a-train-snapshot/v1",
        "id": unit_id(iteration),
        "iteration": iteration,
        "snapshot": {"id": unit_id(iteration), **payload},
        "metadata": {
            "allocation": allocation(src_control),
            "source_commit": src_control["source_commit"],
            "prepare_phase": "ready",
        },
    }
    publish(export_dir / f"{stem}.json", lambda path: path.write_text(json.dumps(info)))
    publish(export_dir / f"{stem}-actor.pth", lambda path: path.write_bytes(b"periodic-actor"))

    encoded = encode_snapshot(
        export_dir / f"{stem}.pt",
        source_packet_id=source_packet_id,
        iteration=iteration,
        source_commit=source_commit,
    )
    assert "allowlist_entry" in encoded
    allowlist_path = tmp_path / "compact_allowlist.json"
    allowlist_path.write_text(json.dumps({"schema": COMPACT_ALLOWLIST_SCHEMA, "entries": []}))
    register_compact_allowlist_entry(encoded["allowlist_entry"], path=allowlist_path)

    ledger, uploads, latest = build_ledger(
        export_dir, dst_control, source_control=src_control, allowlist_path=allowlist_path
    )
    assert latest == iteration
    ledger, uploads = attach_compact_resume(
        ledger,
        uploads,
        export_dir,
        source_packet_id=source_packet_id,
        allowlist_path=allowlist_path,
    )
    assert ledger["compact_resume"]["schema"] == "sandbox-a-train-compact-resume/v2"
    assert ledger["compact_resume"]["source_packet_id"] == source_packet_id
    assert ledger["compact_resume"]["iteration"] == iteration
    assert f"resume/{stem}.pt" not in {row["dest"] for row in uploads}

    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "verified-ledger.json").write_text(json.dumps(ledger))
    broker = {"inputs": [{"dest": "b8-s3.pth", "path": "/x", "sha256": "0" * 64}, *uploads]}
    validate_uploaded_ledger(packet, broker, dst_control, allowlist_path=allowlist_path)

    # Forging source_packet_id or iteration in the manifest fails closed.
    manifest_file = export_dir / f"{stem}.pt.xz.manifest"
    forged_payload = json.loads(manifest_file.read_text())
    forged_payload["iteration"] = iteration + 16
    forged_path = tmp_path / "forged.xz.manifest"
    forged_path.write_text(json.dumps(forged_payload))
    forged_entry = copy.deepcopy(ledger["compact_resume"])
    forged_entry["manifest"] = descriptor(forged_path, forged_entry["manifest"]["path"])
    with pytest.raises(CompactResumeRefused, match="source_packet_id/iteration differs"):
        load_manifest(forged_path, forged_entry)

    # Runtime materialization decodes byte-identical periodic checkpoint and verifies tensors.
    inputs = tmp_path / "inputs"
    for row in uploads:
        (inputs / row["dest"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["path"], inputs / row["dest"])
    materialized = materialize_compact_resume(ledger, inputs, allowlist_path=allowlist_path)
    decoded_target = inputs / f"resume/{stem}.pt"
    assert decoded_target.read_bytes() == (export_dir / f"{stem}.pt").read_bytes()
    loaded_state = torch.load(decoded_target, map_location="cpu", weights_only=False)
    verify_decoded_state(loaded_state, materialized)
    ledger_on_vm = inputs / "verified-ledger.json"
    ledger_on_vm.write_text(json.dumps(ledger))
    verified = load_ledger(
        ledger_on_vm, dst_control, input_root=inputs, allowlist_path=allowlist_path
    )
    assert list(verified) == [iteration]


real = pytest.mark.skipif(
    not (REAL_ORIGINAL.is_file() and REAL_DERIVATIVE.is_file() and REAL_MANIFEST.is_file()),
    reason="train01-11 iteration-0000 artifacts are only present on the operator Mac",
)


@real
def test_real_artifacts_are_the_frozen_allowlisted_identities() -> None:
    frozen = COMPACT_RESUME_ALLOWLIST[unit_id(0)]
    for role, path in (
        ("original", REAL_ORIGINAL),
        ("derivative", REAL_DERIVATIVE),
        ("manifest", REAL_MANIFEST),
    ):
        assert path.stat().st_size == frozen[role]["size"]
        assert Path(f"{path}.sha256").read_text().strip() == frozen[role]["sha256"]
        assert Path(f"{path}.complete").is_file()
    assert digest(REAL_DERIVATIVE) == frozen["derivative"]["sha256"]
    assert digest(REAL_MANIFEST) == frozen["manifest"]["sha256"]
    entry = {"schema": "sandbox-a-train-compact-resume/v1", "unit": unit_id(0), **frozen}
    manifest = load_manifest(REAL_MANIFEST, entry)
    assert manifest["tensor_count"] == len(manifest["tensors"]) == 209


@real
def test_real_snapshot_round_trip_is_exact_for_every_tensor() -> None:
    """Decode == original: file bytes, and every tensor's dtype/shape/stride/bytes."""
    torch.set_num_threads(1)
    frozen = COMPACT_RESUME_ALLOWLIST[unit_id(0)]
    decoded = decode_to_bytes(
        REAL_DERIVATIVE,
        expected_size=frozen["original"]["size"],
        expected_sha256=frozen["original"]["sha256"],
    )
    assert len(decoded) == 1_327_620_931
    manifest = json.loads(REAL_MANIFEST.read_text())
    from_decoded = torch.load(io.BytesIO(decoded), map_location="cpu", weights_only=False)
    del decoded
    original = torch.load(REAL_ORIGINAL, map_location="cpu", weights_only=False)
    verify_decoded_state(from_decoded, manifest)
    assert describe_state(original) == {
        key: manifest[key]
        for key in ("structure_sha256", "tensor_count", "tensor_bytes", "tensors")
    }
    compared = 0

    def same(left, right, path=()):
        nonlocal compared
        assert type(left) is type(right), path
        if isinstance(left, torch.Tensor):
            assert left.dtype == right.dtype and left.shape == right.shape, path
            assert left.stride() == right.stride(), path
            assert torch.equal(
                left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8)
            ), path
            compared += 1
        elif isinstance(left, np.ndarray):
            assert left.dtype == right.dtype and left.tobytes() == right.tobytes(), path
        elif isinstance(left, dict):
            assert list(left) == list(right), path
            for key in left:
                same(left[key], right[key], (*path, key))
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right), path
            for index, (a, b) in enumerate(zip(left, right)):
                same(a, b, (*path, index))
        elif isinstance(left, float):
            assert left.hex() == right.hex(), path
        else:
            assert left == right, path

    same(original, from_decoded)
    assert compared == 209


def test_compact_at_source_publish_verify_mac_preflight_and_materialize_without_raw_pt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Colab compact-at-source export writes zero raw .pt files and verifies/resumes end-to-end."""
    from research.sandbox_a_resident.training_compact import (
        publish_compact_snapshot,
        verify_compact_snapshot_export,
    )

    c1_commit = "c" * 40
    c2_commit = "d" * 40
    monkeypatch.setattr(
        training_units,
        "allowed_migration_source_commits",
        lambda **_kwargs: frozenset({TRAIN01_15_SOURCE_COMMIT, c1_commit}),
    )
    src_control = {
        "study": "sandbox-a-train01",
        "source_commit": c1_commit,
        "outer_iterations": 384,
        "snapshot_interval": 16,
    }
    dst_control = {**src_control, "source_commit": c2_commit}
    export_dir = tmp_path / "c1-exports"
    export_dir.mkdir(parents=True)

    state = sample_state()
    state["iteration"] = 96
    torch.manual_seed(123456)
    rng_before = torch.get_rng_state().clone()
    published = publish_compact_snapshot(
        export_dir,
        96,
        lambda path: torch.save(state, path),
        metadata={
            "allocation": allocation(src_control),
            "source_commit": c1_commit,
            "prepare_phase": "ready",
        },
        source_packet_id="packet-train01-15-c1-i0064",
    )
    assert torch.equal(torch.get_rng_state(), rng_before)
    publish(export_dir / "iteration-0096-actor.pth", lambda path: path.write_bytes(b"actor-96"))

    assert not (export_dir / "iteration-0096.pt").exists()
    assert list(export_dir.glob("*.pt")) == []
    assert (export_dir / "iteration-0096.pt.xz").is_file()
    assert (export_dir / "iteration-0096.pt.xz.manifest").is_file()
    assert (export_dir / "iteration-0096.json").is_file()
    assert published["compact_snapshot"]["schema"] == "sandbox-a-train-compact-export/v1"

    verified = verify_compact_snapshot_export(
        export_dir,
        96,
        source_packet_id="packet-train01-15-c1-i0064",
        source_commit=c1_commit,
    )
    assert verified["status"] == "PASS"
    allowlist_entry = verified["allowlist_entry"]

    # Adversarial tampering checks on Mac verification of compact-at-source export.
    bad_tensor_dir = tmp_path / "tampered-tensor"
    shutil.copytree(export_dir, bad_tensor_dir)
    man_file = bad_tensor_dir / "iteration-0096.pt.xz.manifest"
    man_obj = json.loads(man_file.read_text())
    man_obj["tensors"][0]["sha256"] = "0" * 64
    for suffix in ("", ".sha256", ".complete"):
        Path(f"{man_file}{suffix}").unlink()
    publish(man_file, lambda dst: dst.write_text(json.dumps(man_obj) + "\n"))
    info_file = bad_tensor_dir / "iteration-0096.json"
    info_obj = json.loads(info_file.read_text())
    info_obj["compact_snapshot"]["manifest"] = {
        "name": man_file.name,
        "size": man_file.stat().st_size,
        "sha256": digest(man_file),
    }
    for suffix in ("", ".sha256", ".complete"):
        Path(f"{info_file}{suffix}").unlink()
    publish(info_file, lambda dst: dst.write_text(json.dumps(info_obj) + "\n"))
    with pytest.raises(CompactResumeRefused, match="decoded tensor differs"):
        verify_compact_snapshot_export(bad_tensor_dir, 96)

    # Chain c1 -> c2 ledger, Mac preflight (validate_uploaded_ledger), and VM materialization.
    allowlist_path = tmp_path / "compact_allowlist.json"
    allowlist_path.write_text(json.dumps({"schema": COMPACT_ALLOWLIST_SCHEMA, "entries": []}))
    register_compact_allowlist_entry(allowlist_entry, path=allowlist_path)

    ledger, uploads, latest = build_ledger(
        export_dir, dst_control, source_control=src_control, allowlist_path=allowlist_path
    )
    assert latest == 96
    upload_dests = [row["dest"] for row in uploads]
    assert "resume/iteration-0096.pt" not in upload_dests
    assert "resume/iteration-0096.pt.sha256" not in upload_dests
    assert "resume/iteration-0096.pt.complete" not in upload_dests
    assert "resume/iteration-0096.pt.xz" in upload_dests
    assert "resume/iteration-0096.pt.xz.manifest" in upload_dests

    ledger, uploads = attach_compact_resume(
        ledger,
        uploads,
        export_dir,
        source_packet_id="packet-train01-15-c1-i0064",
        allowlist_path=allowlist_path,
    )
    packet = tmp_path / "packet-c2"
    packet.mkdir()
    (packet / "verified-ledger.json").write_text(json.dumps(ledger))
    broker = {"inputs": [{"dest": "b8-s3.pth", "path": "/x", "sha256": "0" * 64}, *uploads]}
    validate_uploaded_ledger(packet, broker, dst_control, allowlist_path=allowlist_path)

    vm_inputs = tmp_path / "vm-inputs-c2"
    for row in uploads:
        (vm_inputs / row["dest"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["path"], vm_inputs / row["dest"])
    materialized = materialize_compact_resume(ledger, vm_inputs, allowlist_path=allowlist_path)
    decoded_pt = vm_inputs / "resume/iteration-0096.pt"
    assert decoded_pt.is_file()
    assert Path(f"{decoded_pt}.sha256").read_text().strip() == allowlist_entry["original"]["sha256"]
    assert Path(f"{decoded_pt}.complete").read_text() == "complete\n"
    verify_decoded_state(
        torch.load(decoded_pt, map_location="cpu", weights_only=False), materialized
    )
    vm_ledger_path = vm_inputs / "verified-ledger.json"
    vm_ledger_path.write_text(json.dumps(ledger))
    loaded = load_ledger(
        vm_ledger_path, dst_control, input_root=vm_inputs, allowlist_path=allowlist_path
    )
    assert list(loaded) == [96]
