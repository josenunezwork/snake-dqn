"""Open immutable evidence and replay clock journals; summaries never authorize admission."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

HEX = re.compile(r"[0-9a-f]{64}\Z")
CLOCK_SCHEMA = "bl-evidence-clock/v2"
EXPECTED = {
    "on-time": (True, "delivered"),
    "publication-late": (False, "cutoff"),
    "consumption-late": (False, "cutoff"),
    "wrong-frame": (False, "stale"),
}


class EvidenceError(ValueError):
    """Missing, changed, foreign or insufficient evidence; admission stays blocked."""


def require(condition: bool, reason: str) -> None:
    """Raise a named admission defect rather than trusting a PASS label."""
    if not condition:
        raise EvidenceError(reason)


def digest_file(path: Path) -> str:
    """Hash bytes without loading large evidence files into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileRef:
    """One relative evidence path and the frozen digest expected by its consumer."""

    path: str
    sha256: str

    @classmethod
    def parse(cls, value: Any) -> FileRef:
        require(isinstance(value, Mapping), "file reference must be an object")
        path, digest = value.get("path"), value.get("sha256")
        require(
            isinstance(path, str)
            and bool(path)
            and not Path(path).is_absolute()
            and ".." not in Path(path).parts,
            "file reference needs a safe relative path",
        )
        require(isinstance(digest, str) and HEX.fullmatch(digest) is not None, "invalid sha256")
        return cls(path, digest)

    def resolve(self, root: Path) -> Path:
        require(
            root.is_absolute() and root.is_dir(), "evidence root must be an existing absolute dir"
        )
        current = root
        require(not current.is_symlink(), "symlink evidence root")
        # Refuse links in every component, including parents of the supplied root.
        for parent in (root, *root.parents):
            require(not parent.is_symlink(), "symlink evidence root parent")
        for part in Path(self.path).parts:
            current = current / part
            require(not current.is_symlink(), "symlink evidence component")
        require(current.resolve().is_relative_to(root.resolve()), "evidence escapes root")
        require(current.is_file(), f"missing evidence file: {self.path}")
        require(digest_file(current) == self.sha256, f"evidence hash mismatch: {self.path}")
        return current

    def read(self, root: Path, *, completed: bool = True) -> dict[str, Any]:
        path = self.resolve(root)
        payload = path.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == self.sha256, "evidence changed during read")
        value = json.loads(payload.decode("utf-8"), parse_constant=_nonfinite)
        require(isinstance(value, dict), "evidence JSON must be an object")
        if completed:
            from research.bl_paced_identity.events import canonical_bytes

            sidecar, marker = Path(str(path) + ".sha256"), Path(str(path) + ".complete")
            require(
                all(item.is_file() and not item.is_symlink() for item in (sidecar, marker)),
                "missing or unsafe export sidecars",
            )
            require(marker.read_bytes() == b"complete\n", "invalid completion marker")
            require(
                sidecar.read_text(encoding="utf-8").split() == [self.sha256, path.name],
                "export digest/name mismatch",
            )
            require(canonical_bytes(value) == payload, "noncanonical JSON export")
        return value


def _read_pinned_journal(root: Path, ref: FileRef) -> tuple:
    """Replay a private snapshot made from single-read, seal-pinned payloads.

    The existing reader validates chains and event structure. A private copy keeps
    its multiple reads from racing producer-side replacement of payloads/sidecars.
    """
    from research.bl_paced_identity.record import export_json, read_journal

    seal_path = ref.resolve(root)
    require(seal_path.name == "seal.json", "journal must reference terminal seal.json")
    seal = ref.read(root)
    directory = seal_path.parent
    hashes = seal.get("chunk_sha256")
    require(isinstance(hashes, list), "invalid sealed chunk inventory")
    expected = {
        f"chunk-{index:08d}.json{suffix}"
        for index in range(len(hashes))
        for suffix in ("", ".sha256", ".complete")
    }
    require({item.name for item in directory.glob("chunk-*")} == expected, "foreign/missing chunks")
    relative = directory.relative_to(root)
    header = FileRef.parse(
        {"path": str(relative / "header.json"), "sha256": seal.get("header_sha256")}
    ).read(root)
    with tempfile.TemporaryDirectory(prefix="m3-pinned-journal-") as temporary:
        snapshot = Path(temporary)
        export_json(snapshot / "seal.json", seal)
        export_json(snapshot / "header.json", header)
        for index, digest in enumerate(hashes):
            name = f"chunk-{index:08d}.json"
            chunk = FileRef.parse({"path": str(relative / name), "sha256": digest}).read(root)
            export_json(snapshot / name, chunk)
        return read_journal(snapshot)


def _nonfinite(value: str) -> None:
    raise EvidenceError(f"nonfinite JSON: {value}")


@dataclass(frozen=True)
class ClockEvidence:
    """Recomputed clock acceptance for exactly one candidate and source closure."""

    candidate_sha256: str
    source_binding_sha256: str
    journal_sha256: tuple[str, ...]


def load_clock(
    report: Mapping[str, Any], source_binding_sha256: str, candidate_sha256: str
) -> ClockEvidence:
    """Verify all four production journals against external C/source expectations.

    The execution receipt is an operator-attested provenance artifact, never the
    platform of this validator. Hashes bind attestations; they do not authenticate
    an untrusted producer. The eventual production provider must publish this
    evidence contract before it can qualify; synthetic prechecks cannot fill it.
    """
    from research.bl_paced_identity.events import CandidateIdentity, sha256
    from research.bl_paced_identity.replay import replay_events

    require(isinstance(report, Mapping), "clock report must be an object")
    require(report.get("schema") == CLOCK_SCHEMA, "clock evidence schema missing")
    require(report.get("verdict") == "PASS", "clock report is not PASS")
    require(
        isinstance(candidate_sha256, str) and HEX.fullmatch(candidate_sha256) is not None,
        "expected candidate digest missing",
    )
    require(report.get("candidate_sha256") == candidate_sha256, "clock candidate mismatch")
    require(report.get("source_binding_sha256") == source_binding_sha256, "clock source mismatch")
    root_string = report.get("root")
    require(isinstance(root_string, str), "clock root missing")
    root = Path(root_string)
    sources = report.get("source_sha256")
    require(isinstance(sources, dict) and bool(sources), "clock source closure missing")
    require(sha256(sources) == source_binding_sha256, "source closure digest mismatch")
    for path_string, digest in sources.items():
        require(
            isinstance(path_string, str) and Path(path_string).is_absolute(), "source path invalid"
        )
        path = Path(path_string)
        # Source paths are deliberately absolute, matching the existing receipt closure.
        FileRef.parse({"path": path.name, "sha256": digest}).resolve(path.parent)
    execution_ref = FileRef.parse(report.get("execution"))
    execution = execution_ref.read(root)
    require(execution.get("schema") == "bl-evidence-execution/v1", "execution schema missing")
    require(execution.get("candidate_sha256") == candidate_sha256, "execution candidate mismatch")
    require(
        execution.get("source_binding_sha256") == source_binding_sha256,
        "execution source mismatch",
    )
    require(
        execution.get("platform") == "Darwin"
        and execution.get("environment") == "local-mac"
        and execution.get("scope") == "production-clock-seam"
        and execution.get("synthetic") is False
        and execution.get("real_clock") is True
        and execution.get("real_transport") is True,
        "execution is not production Mac clock evidence",
    )
    cases = report.get("cases")
    require(isinstance(cases, dict) and set(cases) == set(EXPECTED), "clock cases incomplete")
    seals: list[str] = []
    for name, (accepted, reason) in EXPECTED.items():
        case = cases[name]
        require(isinstance(case, dict), f"clock case malformed: {name}")
        ref = FileRef.parse(case.get("journal_seal"))
        header, events, _ = _read_pinned_journal(root, ref)
        identity = CandidateIdentity(**header["identity"]).as_dict()
        require(sha256(identity) == candidate_sha256, "journal candidate/scheduler mismatch")
        warmup, load = identity["warmup"], identity["load_envelope"]
        require(
            isinstance(warmup, dict)
            and warmup.get("scope") != "synthetic_precheck"
            and warmup.get("qualified") is not False
            and isinstance(load, dict)
            and load.get("latency_qualification") is not False,
            "candidate explicitly declares unqualified synthetic precheck",
        )
        require(header.get("binding_sha256") == source_binding_sha256, "journal source mismatch")
        initial = header.get("initial_state", {})
        require(
            initial.get("synthetic") is False
            and initial.get("scope") == "production-clock-seam"
            and initial.get("execution_sha256") == execution_ref.sha256,
            "journal is synthetic or has foreign execution provenance",
        )
        replay = replay_events(header, events)
        require(replay.get("scheduler_verdict") == "PASS", f"journal replay failed: {name}")
        decisions = [event.data for event in events if event.kind == "decision"]
        require(len(decisions) == 1, f"clock case needs one decision: {name}")
        observed = decisions[0]
        require(
            observed.get("accepted") is accepted and observed.get("reason") == reason,
            f"wrong recomputed clock disposition: {name}",
        )
        require(
            case.get("accepted") is accepted
            and case.get("reason") == reason
            and case.get("scheduler_verdict") == "PASS",
            f"clock summary differs from journal: {name}",
        )
        latch = FileRef.parse(case.get("latch")).read(root)
        require(
            latch.get("schema") == "bl-evidence-game-latch/v1"
            and latch.get("candidate_sha256") == candidate_sha256
            and latch.get("execution_sha256") == execution_ref.sha256
            and latch.get("journal_seal_sha256") == ref.sha256,
            f"game latch binding mismatch: {name}",
        )
        require(
            latch.get("request_id") == observed.get("request_id")
            and latch.get("action") == observed.get("action")
            and type(latch.get("applied_ns")) is int
            and latch["applied_ns"] >= events[-1].time_ns,
            f"missing actual game latch observation: {name}",
        )
        if accepted:
            requests = [event.data["request"] for event in events if event.kind == "submit_attempt"]
            require(len(requests) == 1, "clock case needs one offered request")
            require(
                latch["applied_ns"] < requests[0]["consume_cutoff_ns"],
                "actual game latch missed consumption cutoff",
            )
        seals.append(ref.sha256)
    require(len(set(seals)) == 4, "clock case journals must be distinct")
    return ClockEvidence(candidate_sha256, source_binding_sha256, tuple(seals))


def clock_problems(
    report: Any, source_binding_sha256: str, candidate_sha256: str | None = None
) -> list[str]:
    """Fail closed for public bool/list validator callers, including malformed JSON."""
    try:
        load_clock(report, source_binding_sha256, candidate_sha256)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, AttributeError) as exc:
        return [str(exc)]
    return []
