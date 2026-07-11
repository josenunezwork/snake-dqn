"""Dashboard data: checkpoint inventory + parsed tournament-eval results.

Reads the project's existing artifacts (saved_snakes/, logs/eval_*.json) — no
new tracking system. Pure functions, safe to call per-request.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVED_DIR = os.path.join(REPO_ROOT, "saved_snakes")
LOGS_DIR = os.path.join(REPO_ROOT, "logs")


def list_checkpoints() -> List[Dict[str, object]]:
    """All .pth files under saved_snakes/ with size (MB)."""
    out = []
    for path in sorted(glob.glob(os.path.join(SAVED_DIR, "*.pth"))):
        out.append(
            {
                "name": os.path.basename(path),
                "size_mb": round(os.path.getsize(path) / 1e6, 2),
            }
        )
    return out


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def eval_leaderboard() -> List[Dict[str, object]]:
    """Aggregate every logs/eval_*.json / reverify_*.json into per-run rows."""
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
            if not isinstance(data, dict) or "summaries" not in data:
                continue
            for summ in data.get("summaries", []):
                runs = summ.get("runs", [])
                if not runs:
                    continue
                rows.append(
                    {
                        "file": os.path.basename(path),
                        "candidate": os.path.basename(str(summ.get("candidate", "?"))),
                        "opponent": os.path.basename(str(data.get("opponent", "?"))),
                        "frames": data.get("frames"),
                        "n": summ.get("n", len(runs)),
                        "mean_mass": round(_mean([r.get("mean_mass", 0) for r in runs]), 1),
                        "max_mass": round(max((r.get("max_mass", 0) for r in runs), default=0), 1),
                        "survival": round(_mean([r.get("survival", 0) for r in runs]), 3),
                        "kills": round(_mean([r.get("kills", 0) for r in runs]), 3),
                    }
                )
    rows.sort(key=lambda r: r["mean_mass"], reverse=True)
    return rows


def dashboard() -> Dict[str, object]:
    return {
        "checkpoints": list_checkpoints(),
        "leaderboard": eval_leaderboard(),
    }
