#!/usr/bin/env python3
"""World-seed disjointness preflight of the FRP-v3 strict gate (read-only, fail closed).

The gate's two banks (``apex-frp3-strict-dev-v1`` calibration, ``apex-frp3-strict-final-v1``
final; ``uint32_be(sha256("<domain>|worlds|<i>")[:4])``, the template recipe) must be disjoint
from every world this project has played, trained on or reserved. :func:`excluded_seeds` (the
spec's ``excluded_seeds``; ``prepare`` freezes the names and checks the banks against it) is the
union of:

1. the v8 strict package's ``excluded_seeds()`` (first 1000 seeds of every earlier
   domain/purpose up to v8, the task-aligned challenger namespaces, the strict pilot's observed
   seeds, seeds 0..999, the seeds saved in the v7/v8 artifacts);
2. :data:`NEWER_DOMAINS` (first 1000 seeds, purposes ``worlds`` and the web purposes): the v8
   strict banks and every study after it that this branch does not carry (v8 serving/census,
   v9, growth ceiling, reference/long-horizon banks, oracle rescue, LH-1 calibration and the
   LH-1 screen of this very candidate, FRP-v2 and FRP-v3);
3. every ``*DOMAIN*`` / ``*_ID`` string literal under ``research/`` on **every local branch**
   (``git grep`` over ``refs/heads``; unmerged studies included), purposes ``worlds`` and the
   web purposes;
4. **FRP-v3 and FRP-v2, analytically**: every Phase R / phase / extension evaluation bank and
   fidelity bank with its real purposes, and the first :data:`STREAM_PREFIX` indices of every
   training and hazard stream (the worlds the candidate was trained on: 10x FRP's own 2000-
   index reservation), plus both studies' written ``training_exclusions.json`` (sha256 checked
   against their ``namespace_report.json``);
5. every integer under a JSON key containing ``seed`` in every JSON file of every artifact root
   (``/Users/josenunez/Projects/ml/snake-dqn-artifacts``), this study's own root included, and
   every line of every ``*.jsonl`` file; a file (or line) too large or unparseable is scanned by
   a regex instead of being skipped (over-exclusion is the safe direction).

This gate's own namespaces are never dropped from the sets: if another study on any branch
declares one of them as a domain literal, the preflight raises.

Usage (the owner's ``preflight`` step; about a minute of one core)::

    ./venv/bin/python research/frp3_strict_20261005/preflight.py \\
        --out <artifacts>/frp3-m3s12-strict-20261005/preflight.json

Exit 0 iff the calibration bank (16) and the final bank up to the template cap (300) are
disjoint from every set and internally unique; 1 otherwise. It also prints the campaign-ledger
``namespaces`` fragment (registration is a separate, optional owner bookkeeping step).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
STUDY_ROOT_NAME = "frp3-m3s12-strict-20261005"  # this study's artifact root (scanned too)
NAMESPACES = {
    "calibration": "apex-frp3-strict-dev-v1",
    "final": "apex-frp3-strict-final-v1",
}
BANK_PURPOSE = "worlds"
CALIBRATION_COUNT = 16
FINAL_CHECK_COUNT = 300  # the template's N_MAX_CAP: every N_max prepare could use is checked
EXCLUSION_PREFIX = 1000
STREAM_PREFIX = 20_000
WEB_PURPOSES = ("watch", "play", "parity", "worlds")
_W = ("worlds",)
NEWER_DOMAINS: Dict[str, Sequence[str]] = {
    # v8 strict (its own banks are not in its excluded_seeds) and the v8 lanes after it
    "apex-veto-v8-strict-dev-v1": _W,
    "apex-veto-v8-strict-final-v1": _W,
    "apex-veto-v8-strict-smoke-v1": _W,
    "apex-veto-v8-web-serving-v1": WEB_PURPOSES,
    "apex-veto-v8-web-serving-smoke-v1": WEB_PURPOSES,
    "apex-veto-v8-census-v1": _W,
    "apex-veto-v8-census-smoke-v1": _W,
    "apex-veto-v9-dev-v1": _W,
    "apex-veto-v9-dev-smoke-v1": _W,
    "apex-veto-v9-screen-v1": _W,
    "apex-veto-v9-screen-smoke-v1": _W,
    "apex-veto-v9-softk-dev-v1": _W,
    "apex-veto-v9-softk-dev-smoke-v1": _W,
    "apex-growth-ceiling-v1": _W,
    "apex-growth-ceiling-smoke-v1": _W,
    "apex-v8-reference-bank-20261003": _W,
    "apex-v8-long-horizon-20261003": _W,
    "kill-opportunity-census-20261003-v1": _W,
    "opponent-robustness-20261003-v1": _W,
    # LH-1 (lh1-screen branch): calibration banks and this candidate's screen
    "apex-lh1-calibration-v1": _W,
    "apex-lh1-calibration-smoke-v1": _W,
    "apex-lh1-screen-frp3-s12-v1": _W,
    # FRP-v2 / FRP-v3 domains (their real purposes are added analytically below as well)
    "apex-frp-v2-train-v1": _W,
    "apex-frp-v2-hazard-v1": _W,
    "apex-frp-v2-fidelity-v1": _W,
    "apex-frp-v2-phase1-v1": _W,
    "apex-frp-v2-phase2-v1": _W,
    "apex-frp-v2-extension-v1": _W,
    "apex-frp-v2-smoke-v1": _W,
    "apex-frp-v3-train-v1": _W,
    "apex-frp-v3-hazard-v1": _W,
    "apex-frp-v3-fidelity-v1": _W,
    "apex-frp-v3-eval-v1": _W,
    "apex-frp-v3-smoke-v1": _W,
}
# FRP-v3 (branch frp-v3, research/frp_v3_20261005/spec.py) and FRP-v2 (frp-v2) constants,
# frozen copies (a test checks them against the branches when they exist).
FRP_V3 = {
    "train_domain": "apex-frp-v3-train-v1",
    "hazard_domain": "apex-frp-v3-hazard-v1",
    "fidelity_domain": "apex-frp-v3-fidelity-v1",
    "eval_domain": "apex-frp-v3-eval-v1",
    "smoke_domain": "apex-frp-v3-smoke-v1",
    "seeds": (10, 11, 12, 13, 14),
    "actors": 5,
    "eval_worlds_per_seed": 32,
}
FRP_V2 = {
    "train_domain": "apex-frp-v2-train-v1",
    "hazard_domain": "apex-frp-v2-hazard-v1",
    "fidelity_domain": "apex-frp-v2-fidelity-v1",
    "phase1_domain": "apex-frp-v2-phase1-v1",
    "phase2_domain": "apex-frp-v2-phase2-v1",
    "extension_domain": "apex-frp-v2-extension-v1",
    "smoke_domain": "apex-frp-v2-smoke-v1",
    "seeds": (0, 1, 2, 3, 4),
    "arms": ("C", "M"),
    "actors": 5,
}
FRP_NAMESPACE_DIRS = {
    "frp-v3": ARTIFACTS / "frp-v3-20261005" / "namespaces",
    "frp-v2": ARTIFACTS / "frp-v2-20261004" / "namespaces",
}
FIDELITY_PURPOSES = ("f1", "f2", "f3", "f3-C", "f3-M", "f4", "f4-unused", "f5", "f5-v8", "f6")
BANK_PREFIX = 2000  # per real evaluation purpose (FRP banks hold 32-64 worlds)
SCAN_KEY_FRAGMENT = "seed"
SCAN_MAX_BYTES = 8 << 20
_REGEX_SEED = re.compile(r'"([^"]*seed[^"]*)"\s*:\s*(\[[^\]]*\]|-?\d+)')
_INT = re.compile(r"-?\d+")
_BRANCH_LITERAL = re.compile(
    r"^\s*[A-Z0-9_]*(?:DOMAIN|_ID)[A-Z0-9_]*\s*=\s*\"([a-z0-9][a-z0-9./_-]*)\"", re.M
)


def uint32_seed(domain: str, purpose: str, index: int) -> int:
    """The dev_screen / template recipe: ``uint32_be(sha256("<domain>|<purpose>|<i>")[:4])``."""
    digest = hashlib.sha256(f"{domain}|{purpose}|{int(index)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def prefix(domain: str, purpose: str, count: int) -> List[int]:
    return [uint32_seed(domain, purpose, i) for i in range(int(count))]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gate_banks(final_count: int = FINAL_CHECK_COUNT) -> Dict[str, List[int]]:
    return {
        "calibration": prefix(NAMESPACES["calibration"], BANK_PURPOSE, CALIBRATION_COUNT),
        "final": prefix(NAMESPACES["final"], BANK_PURPOSE, final_count),
    }


# ---------------------------------------------------------------- the exclusion sets


def domain_prefixes(table: Mapping[str, Sequence[str]]) -> Dict[str, List[int]]:
    return {
        f"{domain}/{purpose}[0:{EXCLUSION_PREFIX}]": prefix(domain, purpose, EXCLUSION_PREFIX)
        for domain, purposes in sorted(table.items())
        for purpose in purposes
    }


def branch_domain_literals(repo: Path = REPO) -> List[str]:
    """``*DOMAIN*`` / ``*_ID`` literals under ``research/`` on every local branch."""
    refs = subprocess.run(
        ["git", "-C", str(repo), "for-each-ref", "--format=%(refname)", "refs/heads"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    found: set = set()
    for ref in refs:
        done = subprocess.run(
            ["git", "-C", str(repo), "grep", "-h", "-E", "DOMAIN|_ID", ref, "--", "research"],
            capture_output=True,
            text=True,
        )
        if done.returncode not in (0, 1):  # 1 = no match
            raise RuntimeError(f"git grep failed on {ref}: {done.stderr.strip()}")
        literals = set(_BRANCH_LITERAL.findall(done.stdout))
        own = sorted(literals & set(NAMESPACES.values()))
        if own:  # another study declared this gate's namespace: never share it (fail closed)
            raise RuntimeError(f"this gate's namespace {own} is a domain literal on {ref}")
        found.update(literals)
    if not found:
        raise RuntimeError("no domain literal found on any branch (git grep broken?)")
    return sorted(found)


def frp_streams() -> Dict[str, List[int]]:
    """FRP-v3 / FRP-v2 evaluation banks and training-like streams with their real purposes."""
    v3, v2 = FRP_V3, FRP_V2
    out: Dict[str, List[int]] = {}

    def add(domain: str, purpose: str, count: int) -> None:
        out[f"{domain}/{purpose}[0:{count}]"] = prefix(domain, purpose, count)

    for seed in v3["seeds"]:
        add(v3["eval_domain"], f"seed{seed}", BANK_PREFIX)
        for actor in range(v3["actors"]):
            add(v3["train_domain"], f"seed{seed}/actor{actor}", STREAM_PREFIX)
    for actor in range(v3["actors"]):
        add(v3["hazard_domain"], f"actor{actor}", STREAM_PREFIX)
    for name in FIDELITY_PURPOSES:
        add(v3["fidelity_domain"], name, BANK_PREFIX)
        add(v2["fidelity_domain"], name, BANK_PREFIX)
    for seed in v2["seeds"]:
        for domain in (v2["phase1_domain"], v2["phase2_domain"], v2["extension_domain"]):
            add(domain, f"seed{seed}", BANK_PREFIX)
        for actor in range(v2["actors"]):
            add(v2["train_domain"], f"seed{seed}/actor{actor}", STREAM_PREFIX)
            add(v2["train_domain"], f"seed{seed}/ext/actor{actor}", STREAM_PREFIX)
    for arm in v2["arms"]:
        for actor in range(v2["actors"]):
            add(v2["hazard_domain"], f"arm{arm}/actor{actor}", STREAM_PREFIX)
    for phase in ("1", "2", "ext"):
        for seed in (0, 1):
            add(v2["smoke_domain"], f"phase{phase}/seed{seed}", BANK_PREFIX)
    return out


def frp_written_exclusions(dirs: Mapping[str, Path] = FRP_NAMESPACE_DIRS) -> Dict[str, List[int]]:
    """Each FRP study's ``training_exclusions.json`` (the union of every bank it knew), sha256
    checked against its ``namespace_report.json`` (a missing or changed file raises)."""
    out: Dict[str, List[int]] = {}
    for name, directory in sorted(dirs.items()):
        report = json.loads((Path(directory) / "namespace_report.json").read_text())
        recorded = report["training_exclusions"]
        path = Path(directory) / "training_exclusions.json"
        if sha256_file(path) != recorded["sha256"]:
            raise ValueError(f"{path} does not hash to {name}'s namespace report")
        values = json.loads(path.read_text(encoding="utf-8"))
        if len(values) != int(recorded["count"]):
            raise ValueError(f"{path} count differs from {name}'s namespace report")
        out[f"{name} training_exclusions.json"] = [int(v) for v in values]
        for bank_name, seeds in (report.get("banks") or {}).items():
            if isinstance(seeds, list):
                out[f"{name} bank {bank_name}"] = [int(v) for v in seeds]
    return out


def collect(value: Any, out: set, under: bool = False) -> None:
    """Every int under a key containing ``seed`` (any depth)."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            collect(item, out, under or SCAN_KEY_FRAGMENT in str(key))
    elif isinstance(value, list):
        for item in value:
            collect(item, out, under)
    elif under and isinstance(value, int) and not isinstance(value, bool):
        out.add(int(value))


def regex_seeds(text: str) -> set:
    found: set = set()
    for _key, value in _REGEX_SEED.findall(text):
        found.update(int(v) for v in _INT.findall(value))
    return found


def _scan_jsonl(path: Path, seeds: set) -> bool:
    """One JSON document per line; a line that does not parse is regex-scanned. True if any
    line needed the regex."""
    used_regex = False
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                collect(json.loads(line), seeds)
            except ValueError:
                used_regex = True
                seeds |= regex_seeds(line)
    return used_regex


def scan_artifacts(root: Path = ARTIFACTS, skip: Iterable[str] = ()) -> Dict:
    """root name -> {"files", "regex_files", "seeds"} for every artifact root but ``skip``
    (nothing is skipped by default: this study's own root is scanned too, so a seed it already
    played can never pass as fresh; ``prepare`` refuses a run root that is not NOT_STARTED)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"artifact root {root} missing")
    skipped = set(skip)
    out: Dict[str, Any] = {}
    for base in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in skipped):
        seeds: set = set()
        files = regex_files = 0
        for path in sorted(base.rglob("*.jsonl")):
            if path.is_file():
                files += 1
                regex_files += int(_scan_jsonl(path, seeds))
        for path in sorted(base.rglob("*.json")):
            if not path.is_file():
                continue
            files += 1
            data: Any = None
            if path.stat().st_size <= SCAN_MAX_BYTES:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (ValueError, UnicodeDecodeError):
                    data = None
            if data is None:
                regex_files += 1
                seeds |= regex_seeds(path.read_text(encoding="utf-8", errors="replace"))
            else:
                collect(data, seeds)
        out[base.name] = {"files": files, "regex_files": regex_files, "seeds": sorted(seeds)}
    return out


def excluded_seeds(
    *, scan: bool = True, artifacts: Path = ARTIFACTS, repo: Path = REPO
) -> Dict[str, List[int]]:
    """Every earlier namespace (items 1-5 of the module docstring); ``scan=False`` (tests
    only) drops the artifact scan."""
    from research.apex_veto_v8_strict_20261003 import spec as v8spec

    sets: Dict[str, List[int]] = {k: list(v) for k, v in v8spec.excluded_seeds().items()}
    table: Dict[str, Sequence[str]] = {}
    for name in branch_domain_literals(repo):
        table[name] = WEB_PURPOSES
    for name, purposes in NEWER_DOMAINS.items():
        table[name] = tuple(dict.fromkeys([*table.get(name, ()), *purposes]))
    sets.update(domain_prefixes(table))
    sets.update(frp_streams())
    sets.update(frp_written_exclusions())
    if scan:
        for name, row in scan_artifacts(artifacts).items():
            sets[f"saved:{name}"] = row["seeds"]
    return sets


def report(final_count: int = FINAL_CHECK_COUNT, *, scan: bool = True) -> Dict[str, Any]:
    began = time.monotonic()
    banks = gate_banks(final_count)
    excluded = excluded_seeds(scan=scan)
    problems = []
    for name, seeds in banks.items():
        if len(set(seeds)) != len(seeds):
            problems.append(f"duplicate seeds within {name}")
    if set(banks["calibration"]) & set(banks["final"]):
        problems.append("calibration and final banks share seeds")
    every = set(banks["calibration"]) | set(banks["final"])
    overlaps = {}
    for label, values in excluded.items():
        hit = sorted(every & {int(v) for v in values})
        if hit:
            overlaps[label] = hit
    if overlaps:
        problems.append(f"{len(overlaps)} excluded sets overlap the gate banks")
    return {
        "schema": "frp3-m3s12-strict-preflight/v1",
        "namespaces": dict(NAMESPACES),
        "recipe": "uint32 big-endian prefix of sha256('<domain>|worlds|<index>')",
        "counts": {name: len(seeds) for name, seeds in banks.items()},
        "excluded_sets": len(excluded),
        "excluded_seed_total": len({int(v) for values in excluded.values() for v in values}),
        "artifact_scan": scan,
        "overlaps": overlaps,
        "problems": problems,
        "disjoint": not problems,
        "seconds": round(time.monotonic() - began, 1),
        "campaign_fragment": {
            "namespaces": {
                "recipe": "uint32 big-endian prefix of sha256('<domain>|<purpose>|<index>')",
                "domains": sorted(NAMESPACES.values()),
                "excludes": "every earlier bank (research/frp3_strict_20261005/preflight.py), "
                "incl. FRP-v2/FRP-v3 banks and training streams, LH-1 and every artifact seed",
            }
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="write the report (create-only)")
    parser.add_argument("--final-count", type=int, default=FINAL_CHECK_COUNT)
    args = parser.parse_args(argv)
    result = report(args.final_count)
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "counts",
                    "excluded_sets",
                    "disjoint",
                    "problems",
                    "excluded_seed_total",
                    "seconds",
                )
            }
        )
    )
    return 0 if result["disjoint"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
