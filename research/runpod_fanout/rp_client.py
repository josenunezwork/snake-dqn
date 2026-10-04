"""Thin wrapper over the runpod skill's ``rp.py`` (the only path to the API key).

The key stays inside ``rp.py`` (Keychain service ``runpod-api``); this module never reads,
prints or forwards it. Spending calls pass ``--confirm`` only when the caller passes
``confirm=True`` (the runner does so only under ``run --confirm``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from research.runpod_fanout.tls import ca_file  # noqa: E402

RP = Path.home() / ".claude/skills/runpod/scripts/rp.py"


class RunPodError(RuntimeError):
    """An rp.py call failed (HTTP error, refusal, timeout or unparsable output)."""

    def __init__(self, message: str, payload: Any = None):
        super().__init__(message)
        self.payload = payload


class RpClient:
    """Calls ``python3 rp.py ...``; every method returns parsed JSON."""

    def __init__(self, rp_path: Path = RP, timeout: float = 90.0, python: Optional[str] = None):
        self.rp_path = Path(rp_path)
        self.timeout = timeout
        # The current interpreter (venv), never whatever "python3" PATH yields under launchd.
        self.python = python or sys.executable

    @staticmethod
    def env() -> Dict[str, str]:
        """rp.py's environment: verified TLS via certifi when the interpreter lacks CAs."""
        env = dict(os.environ)
        ca = ca_file()
        if ca and not env.get("SSL_CERT_FILE"):
            env["SSL_CERT_FILE"] = ca
        return env

    def _call(self, *args: str) -> Any:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise RuntimeError("refusing a real rp.py call under pytest")
        try:
            # Own session: a terminal Ctrl-C must not kill a POST that is already in flight
            # (the runner then knows the outcome and can track or delete the pod).
            done = subprocess.run(
                [self.python, str(self.rp_path), *args],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                start_new_session=True,
                env=self.env(),
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            raise RunPodError(f"rp.py {args[:2]} timed out") from exc
        try:
            payload = json.loads(done.stdout) if done.stdout.strip() else None
        except json.JSONDecodeError:
            payload = done.stdout[-500:]
        if done.returncode != 0:
            # rp.py never prints the key; its stderr says why (no Keychain access, TLS, ...).
            raise RunPodError(
                f"rp.py {args[:3]} exit {done.returncode}: {done.stderr.strip()[-300:]}", payload
            )
        return payload

    # reads
    def balance(self) -> float:
        out = self._call("gql", "query { myself { clientBalance currentSpendPerHr } }")
        return float(out["myself"]["clientBalance"])

    def list_pods(self) -> List[Dict[str, Any]]:
        out = self._call("rest", "GET", "/pods")
        if not isinstance(out, list):  # empty/odd output is an error, never "no pods"
            raise RunPodError(f"GET /pods returned {type(out).__name__}, not a list", out)
        return out

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

    # ------------------------------------------------------------ serverless / storage
    # Reads are plain; anything that creates, changes, deletes or runs is a dry run unless
    # ``confirm=True`` (rp.py prints the request it would send).
    def _list(self, path: str, *query: str) -> List[Dict[str, Any]]:
        args = ["rest", "GET", path]
        for q in query:
            args += ["-q", q]
        out = self._call(*args)
        if not isinstance(out, list):  # empty/odd output is an error, never "nothing"
            raise RunPodError(f"GET {path} returned {type(out).__name__}, not a list", out)
        return out

    def _get_or_none(self, path: str, *query: str) -> Optional[Dict[str, Any]]:
        args = ["rest", "GET", path]
        for q in query:
            args += ["-q", q]
        try:
            return self._call(*args)
        except RunPodError as exc:
            if isinstance(exc.payload, dict) and exc.payload.get("http_status") == 404:
                return None
            raise

    def _write(self, method: str, path: str, body: Any, confirm: bool) -> Any:
        args = ["rest", method, path]
        if body is not None:
            args += ["-d", body if isinstance(body, str) else json.dumps(body)]
        return self._call(*args, *(["--confirm"] if confirm else []))

    def list_endpoints(self) -> List[Dict[str, Any]]:
        return self._list("/endpoints")

    def get_endpoint(self, endpoint_id: str, workers: bool = False) -> Optional[Dict[str, Any]]:
        query = ("includeWorkers=true",) if workers else ()
        return self._get_or_none(f"/endpoints/{endpoint_id}", *query)

    def create_endpoint(self, body_path: Path, confirm: bool) -> Any:
        return self._write("POST", "/endpoints", f"@{body_path}", confirm)

    def update_endpoint(self, endpoint_id: str, body: Dict[str, Any], confirm: bool) -> Any:
        return self._write("PATCH", f"/endpoints/{endpoint_id}", body, confirm)

    def delete_endpoint(self, endpoint_id: str, confirm: bool) -> Any:
        return self._write("DELETE", f"/endpoints/{endpoint_id}", None, confirm)

    def sls(
        self,
        endpoint_id: str,
        op: str,
        job_id: Optional[str] = None,
        body: Optional[Dict[str, Any]] = None,
        confirm: bool = False,
    ) -> Any:
        """``rp.py sls``: run/cancel/retry/purge-queue need ``confirm``; health/status read."""
        args = ["sls", endpoint_id, op] + ([job_id] if job_id else [])
        if body is not None:
            args += ["-d", json.dumps(body, separators=(",", ":"))]
        return self._call(*args, *(["--confirm"] if confirm else []))

    def endpoint_billing(self, endpoint_id: str, start_iso: str, end_iso: str) -> float:
        rows = self._list(
            "/billing/endpoints",
            f"endpointId={endpoint_id}",
            "bucketSize=hour",
            f"startTime={start_iso}",
            f"endTime={end_iso}",
        )
        return float(sum(float(r.get("amount") or 0.0) for r in rows))

    def list_volumes(self) -> List[Dict[str, Any]]:
        return self._list("/networkvolumes")

    def get_volume(self, volume_id: str) -> Optional[Dict[str, Any]]:
        return self._get_or_none(f"/networkvolumes/{volume_id}")

    def create_volume(self, body: Dict[str, Any], confirm: bool) -> Any:
        return self._write("POST", "/networkvolumes", body, confirm)

    def delete_volume(self, volume_id: str, confirm: bool) -> Any:
        return self._write("DELETE", f"/networkvolumes/{volume_id}", None, confirm)

    def list_templates(self) -> List[Dict[str, Any]]:
        return self._list("/templates")

    def create_template(self, body_path: Path, confirm: bool) -> Any:
        return self._write("POST", "/templates", f"@{body_path}", confirm)

    def delete_template(self, template_id: str, confirm: bool) -> Any:
        return self._write("DELETE", f"/templates/{template_id}", None, confirm)
