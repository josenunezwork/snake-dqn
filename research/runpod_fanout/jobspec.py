"""Job files, policy, checkpoint allow-list and the Tier-1-only guard (stdlib only).

A job file is JSON::

    {"schema": "runpod-fanout-job/v1", "job_id": "v8probe-a", "tier": "tier1",
     "purpose": "...", "repo_commit": "<40 hex>", "max_wall_minutes": 60,
     "checkpoints": ["<sha256>", ...],
     "episodes": [{"wrapper": "apex-veto-v8-screen-v1", "arm": "B", "mix": "scripted",
                   "world_seed": 123, "world_index": 0, "horizon": 1000, "engine": "live",
                   "roster_member_sha256s": ["<sha256>", x5]}, ...]}

``checkpoints`` must be exactly the non-scripted roster members plus the hero, and every
one must be in ``checkpoint_allowlist.json``. Anything strict/serving is refused.

Allow-list entries may carry ``"root": "artifacts_root"`` (a trained checkpoint pinned where
the study wrote it); the default root is the policy's ``checkpoint_root``. (Same rule and code
as branch frp-v3-phaser-rp, commit 2e0947f's parent; ported for the FRP-v3 strict gate.)
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from research.runpod_fanout.wrappers import HERO_SHA256, SCRIPTED_SHA256, WRAPPERS

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "fanout_policy.json"
ALLOWLIST_PATH = HERE / "checkpoint_allowlist.json"
ALLOWLIST_ROOTS = ("checkpoint_root", "artifacts_root")
JOB_SCHEMA = "runpod-fanout-job/v1"
MIXES = ("frozen", "scripted", "mixed")
ENGINES = ("live", "simd")
MAX_HORIZON = 5000
ROSTER_WIDTH = 5
JOB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,23}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
# Files every job's commit must contain (the pod imports them from the git archive).
REQUIRED_REPO_FILES = ("research/runpod_fanout/episode.py", "research/runpod_fanout/wrappers.py")
PINNED_RUNNER_FILES = (
    "research/runpod_fanout/episode.py",
    "research/runpod_fanout/wrappers.py",
    "research/runpod_fanout/jobspec.py",
)


class JobError(ValueError):
    """The job file (or a request) is refused."""


def load_policy(path: Path = POLICY_PATH) -> Dict[str, Any]:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    if policy.get("schema") != "runpod-fanout-policy/v1":
        raise JobError("unknown policy schema")
    if policy["flavor"] != "cpu3c" or policy["cloud"] != "SECURE":
        raise JobError("policy must stay cpu3c / SECURE (user-approved)")
    return policy


def load_allowlist(path: Path = ALLOWLIST_PATH) -> Dict[str, Dict[str, str]]:
    """sha256 -> allow-list entry (path relative to the policy's ``root``: ``checkpoint_root``
    by default, or ``artifacts_root`` for a study's trained checkpoint)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") != "runpod-fanout-checkpoint-allowlist/v1":
        raise JobError("unknown allow-list schema")
    out: Dict[str, Dict[str, str]] = {}
    for row in raw["checkpoints"]:
        sha, rel = row["sha256"], row["path"]
        if not SHA_RE.match(sha) or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise JobError(f"bad allow-list entry {row!r}")
        if row.get("root", "checkpoint_root") not in ALLOWLIST_ROOTS:
            raise JobError(f"bad allow-list root in {row!r} (one of {ALLOWLIST_ROOTS})")
        if sha in out:
            raise JobError(f"duplicate allow-list sha {sha}")
        out[sha] = dict(row)
    return out


def forbidden_fragment(text: str, policy: Mapping[str, Any]) -> str | None:
    low = str(text).lower()
    for frag in policy["forbidden_name_fragments"]:
        if frag in low:
            return frag
    return None


def episode_group(ep: Mapping[str, Any]) -> str:
    return f"{ep['wrapper']}__h{int(ep['horizon'])}__{ep['engine']}"


def episode_filename(ep: Mapping[str, Any]) -> str:
    return f"{ep['arm']}-{ep['mix']}-{int(ep['world_seed'])}.json"


def episode_key(ep: Mapping[str, Any]) -> str:
    """Unique id and relative record path: ``<group>/<arm>-<mix>-<seed>.json``."""
    return f"{episode_group(ep)}/{episode_filename(ep)}"


def _check_episode(i: int, ep: Any, ckpts: set) -> List[str]:
    where = f"episodes[{i}]"
    if not isinstance(ep, Mapping):
        return [f"{where} is not an object"]
    expected = {
        "wrapper",
        "arm",
        "mix",
        "world_seed",
        "world_index",
        "horizon",
        "engine",
        "roster_member_sha256s",
    }
    if set(ep) != expected:
        return [f"{where} keys must be exactly {sorted(expected)}"]
    errors: List[str] = []
    wrapper = WRAPPERS.get(ep["wrapper"])
    if wrapper is None:
        return [f"{where} wrapper {ep['wrapper']!r} is not a registered Tier-1 wrapper"]
    if ep["arm"] not in wrapper["arms"]:
        errors.append(f"{where} arm {ep['arm']!r} not in {wrapper['arms']}")
    if ep["mix"] not in MIXES:
        errors.append(f"{where} mix {ep['mix']!r}")
    if ep["engine"] not in ENGINES:
        errors.append(f"{where} engine {ep['engine']!r}")
    elif ep["engine"] == "simd" and not (wrapper.get("simd") or {}).get(ep["arm"]):
        errors.append(f"{where} wrapper/arm has no SIMD mapping")
    for name, lo, hi in (
        ("world_seed", 0, 2**32 - 1),
        ("world_index", 0, 10**6),
        ("horizon", 1, MAX_HORIZON),
    ):
        value = ep[name]
        if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
            errors.append(f"{where} {name} must be an int in [{lo}, {hi}]")
    roster = ep["roster_member_sha256s"]
    if not isinstance(roster, list) or len(roster) != ROSTER_WIDTH:
        errors.append(f"{where} roster must list {ROSTER_WIDTH} sha256s")
    else:
        for sha in roster:
            if not isinstance(sha, str) or not SHA_RE.match(sha):
                errors.append(f"{where} roster entry {sha!r} is not a sha256")
            elif sha not in SCRIPTED_SHA256 and sha not in ckpts:
                errors.append(f"{where} roster member {sha[:12]} is not in the job's checkpoints")
    return errors


def validate_job(
    job: Any, policy: Mapping[str, Any], allowlist: Mapping[str, Mapping[str, str]]
) -> Dict[str, Any]:
    """Return the job (normalized) or raise :class:`JobError` listing every problem."""
    if not isinstance(job, Mapping):
        raise JobError("job must be a JSON object")
    expected = {
        "schema",
        "job_id",
        "tier",
        "purpose",
        "repo_commit",
        "max_wall_minutes",
        "checkpoints",
        "episodes",
    }
    if set(job) != expected:
        raise JobError(f"job keys must be exactly {sorted(expected)}")
    errors: List[str] = []
    if job["schema"] != JOB_SCHEMA:
        errors.append(f"schema must be {JOB_SCHEMA}")
    if (
        not isinstance(job["job_id"], str)
        or not JOB_ID_RE.match(job["job_id"])
        or "--" in job["job_id"]
        or job["job_id"].endswith("-")
    ):
        errors.append("job_id must match ^[a-z0-9][a-z0-9-]{2,23}$ without '--' or a final '-'")
    if job["tier"] not in policy["allowed_tiers"]:
        errors.append(
            f"tier {job['tier']!r} refused: RunPod runs Tier-1 only (no strict gates, "
            "no serving qualification)"
        )
    for label in ("job_id", "purpose", "tier"):
        frag = forbidden_fragment(job.get(label, ""), policy)
        if frag:
            errors.append(f"{label} contains {frag!r}: strict gates / serving refused on RunPod")
    if not isinstance(job["repo_commit"], str) or not COMMIT_RE.match(job["repo_commit"]):
        errors.append("repo_commit must be a full 40-hex commit id")
    mw = job["max_wall_minutes"]
    if isinstance(mw, bool) or not isinstance(mw, (int, float)) or not 5 <= mw <= 24 * 60:
        errors.append("max_wall_minutes must be in [5, 1440]")
    ckpts = job["checkpoints"]
    if not isinstance(ckpts, list) or len(set(ckpts)) != len(ckpts):
        errors.append("checkpoints must be a list of distinct sha256s")
        ckpts = []
    for sha in ckpts:
        if sha not in allowlist:
            errors.append(f"checkpoint {str(sha)[:16]} is not on the upload allow-list")
    if HERO_SHA256 not in ckpts:
        errors.append("checkpoints must include the hero (champion) sha256")
    episodes = job["episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise JobError("; ".join(errors + ["episodes must be a non-empty list"]))
    ckset = set(ckpts)
    keys = set()
    used = {HERO_SHA256}
    for i, ep in enumerate(episodes):
        errs = _check_episode(i, ep, ckset)
        errors.extend(errs)
        if errs:
            continue
        frag = forbidden_fragment(ep["wrapper"], policy)
        if frag:
            errors.append(f"episodes[{i}] wrapper contains {frag!r}")
        key = episode_key(ep)
        if key in keys:
            errors.append(f"duplicate episode {key}")
        keys.add(key)
        used.update(s for s in ep["roster_member_sha256s"] if s not in SCRIPTED_SHA256)
    unused = ckset - used
    if unused:
        errors.append(f"checkpoints not used by any episode (not uploaded): {sorted(unused)}")
    if errors:
        raise JobError("; ".join(errors))
    return dict(job)


def git(repo: Path, *args: str, binary: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=not binary, check=False
    )


def check_commit(repo: Path, job: Mapping[str, Any]) -> List[str]:
    """Problems with the job's repo commit (it must exist and contain the pod code)."""
    commit = job["repo_commit"]
    problems = []
    if git(repo, "cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        return [f"commit {commit} not found in {repo}"]
    modules = {WRAPPERS[ep["wrapper"]]["module"] for ep in job["episodes"]}
    paths = list(REQUIRED_REPO_FILES) + [m.replace(".", "/") + ".py" for m in sorted(modules)]
    for rel in paths:
        if git(repo, "cat-file", "-e", f"{commit}:{rel}").returncode != 0:
            problems.append(f"commit {commit[:12]} lacks {rel}")
    # The pod runs the commit's copy of the guard/registry/executor: it must be byte-equal
    # to the runner's own (validated) copy, so a job cannot point at a commit whose
    # wrappers map an id to a strict gate or change the pins.
    for rel in PINNED_RUNNER_FILES:
        at_commit = git(repo, "rev-parse", f"{commit}:{rel}").stdout.strip()
        local = git(repo, "hash-object", str(HERE / Path(rel).name)).stdout.strip()
        if not at_commit or at_commit != local:
            problems.append(f"{rel} at {commit[:12]} differs from the runner's copy")
    return problems


def git_archive(repo: Path, commit: str, dest: Path) -> str:
    """``git archive`` (tracked files only) of ``commit`` to ``dest`` (tar.gz); sha256."""
    done = git(repo, "archive", "--format=tar.gz", commit, binary=True)
    if done.returncode != 0:
        raise JobError(f"git archive failed: {done.stderr[-300:]!r}")
    Path(dest).write_bytes(done.stdout)
    return hashlib.sha256(done.stdout).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_checkpoints(
    shas: Iterable[str], policy: Mapping[str, Any], allowlist: Mapping[str, Mapping[str, str]]
) -> Dict[str, Path]:
    """sha256 -> local file, each verified against its pinned sha256."""
    out: Dict[str, Path] = {}
    for sha in shas:
        if sha not in allowlist:
            raise JobError(f"checkpoint {sha[:16]} not on the allow-list")
        root_key = allowlist[sha].get("root", "checkpoint_root")
        if root_key not in ALLOWLIST_ROOTS:
            raise JobError(f"allow-list root {root_key!r} for {sha[:16]} is not allowed")
        path = Path(policy[root_key]) / allowlist[sha]["path"]
        if not path.is_file():
            raise JobError(f"allow-listed checkpoint missing: {path}")
        actual = sha256_file(path)
        if actual != sha:
            raise JobError(f"{path} sha256 {actual} != pinned {sha}")
        out[sha] = path
    return out


TIMING_TOP_KEYS = ("wall_seconds", "platform", "fanout")


def strip_timing(value: Any) -> Any:
    """Drop every key containing ``seconds`` (any depth)."""
    if isinstance(value, Mapping):
        return {k: strip_timing(v) for k, v in value.items() if "seconds" not in str(k)}
    if isinstance(value, list):
        return [strip_timing(v) for v in value]
    return value


def deterministic_bytes(entry: Mapping[str, Any]) -> bytes:
    """Canonical bytes of an entry minus timing fields and platform/fanout stamps.

    Two runs of the same episode (re-dispatch, duplicate, Mac vs pod) must give equal
    bytes here; anything else is a divergence.
    """
    trimmed = {k: v for k, v in entry.items() if k not in TIMING_TOP_KEYS}
    return json.dumps(
        strip_timing(trimmed), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def summarize_job(job: Mapping[str, Any]) -> Dict[str, Any]:
    eps: Sequence[Mapping[str, Any]] = job["episodes"]
    by: Dict[str, int] = {}
    for ep in eps:
        k = f"{ep['wrapper']} {ep['engine']} H{ep['horizon']} {ep['mix']} arm {ep['arm']}"
        by[k] = by.get(k, 0) + 1
    return {"episodes": len(eps), "by_wrapper_engine_horizon_mix_arm": by}
