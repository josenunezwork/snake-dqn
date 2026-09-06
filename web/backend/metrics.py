"""Dashboard data: checkpoint inventory + parsed tournament-eval results.

Reads the project's existing artifacts (saved_snakes/, runs/**/latest_pqn.pth,
logs/eval_*.json) — no new tracking system. Pure functions, safe to call
per-request (the obs_spec probe is cached by (path, mtime)).

Two eval-JSON schemas are understood:

- Legacy (pre-redesign ``tournament_eval``): top-level ``summaries`` with
  per-run keys ``mean_mass`` / ``survival`` — the retired "mean mass over
  alive frames" gate metric. Rows parsed from it carry
  ``metric: "legacy_mean_mass"``.
- Repaired gate (blueprint P0): top-level ``candidates`` / ``baseline_runs``
  with per-run keys ``mass_integral`` / ``survival_fraction``. Rows carry
  ``metric: "mass_integral"`` and sort ahead of legacy rows so the two metric
  families are never silently ranked against each other.
"""

from __future__ import annotations

import datetime
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

from web.backend.checkpoints import list_checkpoint_catalog

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVED_DIR = os.path.join(REPO_ROOT, "saved_snakes")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
LOGS_DIR = os.path.join(REPO_ROOT, "logs")

# obs_spec probe cache: path -> (mtime, spec). torch.load unpickles the whole
# blob, so do it at most once per (path, mtime) for the checkpoint listing.
_OBS_SPEC_CACHE: Dict[str, Tuple[float, str]] = {}


def _checkpoint_obs_spec(path: str) -> str:
    """Best-effort obs_spec ("vector61" | "raster31v2" | "unknown") for a checkpoint.

    Cached by (path, mtime) so repeated /api/checkpoints requests do not re-read
    checkpoint files. Unreadable/non-checkpoint files report "unknown".
    """
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return "unknown"
    cached = _OBS_SPEC_CACHE.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        import torch

        from src.model.inference_agent import InferenceAgent

        blob = torch.load(path, map_location="cpu", weights_only=False)
        spec = str(InferenceAgent._detect_obs_spec(blob))
    except Exception:
        spec = "unknown"
    _OBS_SPEC_CACHE[path] = (mtime, spec)
    return spec


def _checkpoint_entry(path: str, name: str) -> Dict[str, object]:
    return {
        "name": name,
        "path": os.path.relpath(path, REPO_ROOT),
        "size_mb": round(os.path.getsize(path) / 1e6, 2),
        "obs_spec": _checkpoint_obs_spec(path),
    }


def list_checkpoints() -> List[Dict[str, object]]:
    """All .pth files under saved_snakes/ plus runs/**/latest_pqn.pth.

    Each entry carries ``obs_spec`` so the model picker can distinguish
    vector61 champions from raster31v2 candidates instead of presenting one
    undifferentiated flat list. Training-run outputs (``runs/**/latest_pqn.pth``)
    are listed under their repo-relative path so they are unambiguous next to
    the saved_snakes basenames.
    """
    return [
        _checkpoint_entry(checkpoint.path, checkpoint.name)
        for checkpoint in list_checkpoint_catalog(REPO_ROOT, SAVED_DIR, RUNS_DIR)
    ]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _run_mass(run: Dict[str, object]) -> float:
    """Headline mass for one run: repaired-gate mass_integral, else legacy mean_mass."""
    return float(run.get("mass_integral", run.get("mean_mass", 0)) or 0)


def _run_survival(run: Dict[str, object]) -> float:
    return float(run.get("survival_fraction", run.get("survival", 0)) or 0)


def _file_date(path: str) -> Optional[str]:
    """File mtime as a local ISO timestamp (seconds precision)."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    return datetime.datetime.fromtimestamp(mtime).isoformat(timespec="seconds")


def _row_from_runs(
    runs: List[Dict[str, object]],
    *,
    file: str,
    date: Optional[str],
    candidate: str,
    opponent: str,
    frames: object,
    n: object,
    mass_ci: Optional[float] = None,
) -> Dict[str, object]:
    """One leaderboard row from a list of per-seed run dicts (either schema).

    ``mean_mass`` stays the headline-mass column name for frontend
    compatibility; ``metric`` says which metric filled it.
    """
    new_schema = any("mass_integral" in r for r in runs)
    row: Dict[str, object] = {
        "file": file,
        "date": date,
        "candidate": candidate,
        "opponent": opponent,
        "frames": frames,
        "n": n if n is not None else len(runs),
        "metric": "mass_integral" if new_schema else "legacy_mean_mass",
        "mean_mass": round(_mean([_run_mass(r) for r in runs]), 1),
        "max_mass": round(max((float(r.get("max_mass", 0) or 0) for r in runs), default=0), 1),
        "survival": round(_mean([_run_survival(r) for r in runs]), 3),
        "kills": round(_mean([float(r.get("kills", 0) or 0) for r in runs]), 3),
    }
    if new_schema:
        row["mean_mass_alive"] = round(
            _mean([float(r.get("mean_mass_alive", 0) or 0) for r in runs]), 1
        )
    if mass_ci is not None:
        row["mass_ci"] = round(float(mass_ci), 2)
    return row


def _rows_legacy(
    data: Dict[str, object], path: str, date: Optional[str]
) -> List[Dict[str, object]]:
    """Rows from the pre-redesign format (top-level ``summaries``)."""
    rows: List[Dict[str, object]] = []
    for summ in data.get("summaries", []):
        runs = summ.get("runs", [])
        if not runs:
            continue
        rows.append(
            _row_from_runs(
                runs,
                file=os.path.basename(path),
                date=date,
                candidate=os.path.basename(str(summ.get("candidate", "?"))),
                opponent=os.path.basename(str(data.get("opponent", "?"))),
                frames=data.get("frames"),
                n=summ.get("n"),
            )
        )
    return rows


def _rows_repaired_gate(
    data: Dict[str, object], path: str, date: Optional[str]
) -> List[Dict[str, object]]:
    """Rows from the repaired-gate format (top-level ``candidates``/``baseline_runs``).

    Emits one row per (agent, opponent mix): every candidate's per-mix runs plus
    the baseline's, so the incumbent it was gated against stays visible.
    """
    rows: List[Dict[str, object]] = []
    file = os.path.basename(path)
    frames = data.get("frames")

    def add(candidate: str, mix: str, runs, summary) -> None:
        if not runs:
            return
        summary = summary if isinstance(summary, dict) else {}
        rows.append(
            _row_from_runs(
                runs,
                file=file,
                date=date,
                candidate=os.path.basename(str(candidate)),
                opponent=f"{mix} mix",
                frames=frames,
                n=summary.get("n"),
                mass_ci=summary.get("mass_integral_ci"),
            )
        )

    for cand in data.get("candidates", []):
        if not isinstance(cand, dict):
            continue
        per_mix = cand.get("per_mix")
        if not isinstance(per_mix, dict):
            continue  # candidate that errored out mid-gate: nothing to rank
        for mix, entry in per_mix.items():
            if isinstance(entry, dict):
                add(
                    str(cand.get("candidate", "?")),
                    str(mix),
                    entry.get("runs", []),
                    entry.get("summary"),
                )

    baseline = str(data.get("baseline", "baseline"))
    baseline_runs = data.get("baseline_runs")
    baseline_summaries = data.get("baseline_summaries")
    if isinstance(baseline_runs, dict):
        summaries = baseline_summaries if isinstance(baseline_summaries, dict) else {}
        for mix, runs in baseline_runs.items():
            add(baseline, str(mix), runs, summaries.get(mix))
    return rows


def eval_leaderboard() -> List[Dict[str, object]]:
    """Aggregate every logs/eval_*.json / reverify_*.json into per-run rows.

    Understands both eval schemas (see module docstring). Repaired-gate
    (``mass_integral``) rows rank first, then legacy rows — never interleaved,
    since the two metrics are not comparable.
    """
    rows: List[Dict[str, object]] = []
    patterns = ["eval_*.json", "reverify_*.json"]
    seen = set()
    for pat in patterns:
        for path in sorted(glob.glob(os.path.join(LOGS_DIR, pat))):
            if path in seen:
                continue
            seen.add(path)
            try:
                data = json.load(open(path))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            date = _file_date(path)
            if "candidates" in data or "baseline_runs" in data:
                rows.extend(_rows_repaired_gate(data, path, date))
            elif "summaries" in data:
                rows.extend(_rows_legacy(data, path, date))
    rows.sort(key=lambda r: (0 if r["metric"] == "mass_integral" else 1, -float(r["mean_mass"])))
    return rows


def dashboard() -> Dict[str, object]:
    return {
        "checkpoints": list_checkpoints(),
        "leaderboard": eval_leaderboard(),
    }
