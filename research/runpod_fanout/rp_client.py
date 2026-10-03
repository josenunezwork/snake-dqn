"""Thin wrapper over the runpod skill's ``rp.py`` (the only path to the API key).

The key stays inside ``rp.py`` (Keychain service ``runpod-api``); this module never reads,
prints or forwards it. Spending calls pass ``--confirm`` only when the caller passes
``confirm=True`` (the runner does so only under ``run --confirm``).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

RP = Path.home() / ".claude/skills/runpod/scripts/rp.py"


class RunPodError(RuntimeError):
    """An rp.py call failed (HTTP error, refusal, timeout or unparsable output)."""

    def __init__(self, message: str, payload: Any = None):
        super().__init__(message)
        self.payload = payload


class RpClient:
    """Calls ``python3 rp.py ...``; every method returns parsed JSON."""

    def __init__(self, rp_path: Path = RP, timeout: float = 90.0):
        self.rp_path = Path(rp_path)
        self.timeout = timeout

    def _call(self, *args: str) -> Any:
        try:
            done = subprocess.run(
                ["python3", str(self.rp_path), *args],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RunPodError(f"rp.py {args[:2]} timed out") from exc
        try:
            payload = json.loads(done.stdout) if done.stdout.strip() else None
        except json.JSONDecodeError:
            payload = done.stdout[-500:]
        if done.returncode != 0:
            raise RunPodError(f"rp.py {args[:3]} exit {done.returncode}", payload)
        return payload

    # reads
    def balance(self) -> float:
        out = self._call("gql", "query { myself { clientBalance currentSpendPerHr } }")
        return float(out["myself"]["clientBalance"])

    def list_pods(self) -> List[Dict[str, Any]]:
        out = self._call("rest", "GET", "/pods")
        return list(out or [])

    def get_pod(self, pod_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self._call("rest", "GET", f"/pods/{pod_id}")
        except RunPodError as exc:
            if isinstance(exc.payload, dict) and exc.payload.get("http_status") == 404:
                return None
            raise

    def gql(self, query: str, variables: Dict[str, Any]) -> Any:
        return self._call("gql", query, "--vars", json.dumps(variables))

    def stock(self, flavor: str, vcpu: int, ram_gb: int, dc: str) -> Dict[str, Any]:
        q = (
            "query($dc:String,$inst:String){ cpuFlavors { id specifics(input:{dataCenterId:$dc,"
            " instanceId:$inst}) { stockStatus securePrice } } }"
        )
        out = self.gql(q, {"dc": dc, "inst": f"{flavor}-{vcpu}-{ram_gb}"})
        for row in out["cpuFlavors"]:
            if row["id"] == flavor and row.get("specifics"):
                return dict(row["specifics"])
        return {"stockStatus": None, "securePrice": None}

    def datacenters(self) -> List[str]:
        out = self.gql("query { dataCenters { id listed } }", {})
        return [d["id"] for d in out["dataCenters"] if d.get("listed")]

    # spending / changing (dry run unless confirm)
    def create_pod(self, body_path: Path, max_hourly: float, confirm: bool) -> Any:
        args = ["rest", "POST", "/pods", "-d", f"@{body_path}", "--max-hourly", str(max_hourly)]
        return self._call(*args, *(["--confirm"] if confirm else []))

    def delete_pod(self, pod_id: str, confirm: bool) -> Any:
        return self._call("rest", "DELETE", f"/pods/{pod_id}", *(["--confirm"] if confirm else []))
