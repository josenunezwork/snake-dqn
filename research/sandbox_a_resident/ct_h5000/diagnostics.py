"""Bounded, logging-only death, length and replay journals, including incomplete exits."""

from __future__ import annotations

import gzip
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from research.sandbox_a_resident.ct_h5000.units import publish_bounded_export


class DiagnosticJournal:
    """Spool compressed JSONL outside exports; publish SHA/completion-last on every exit."""

    names = ("deaths", "length-histograms", "replay-ties")
    max_bytes = 96 * 1024 * 1024

    def __init__(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="ct-h5000-diagnostics-")
        self.paths = {name: Path(self.temp.name) / f"{name}.jsonl.gz" for name in self.names}
        self.streams = {
            name: gzip.open(path, "wt", encoding="utf-8") for name, path in self.paths.items()
        }
        self.counts = dict.fromkeys(self.names, 0)
        self.closed = False
        self.published: dict[str, Any] | None = None

    def record(self, name: str, row: Mapping[str, Any]) -> None:
        """Record every event; refuse explicit overflow rather than silently losing records."""
        self.streams[name].write(json.dumps(dict(row), separators=(",", ":")) + "\n")
        self.counts[name] += 1
        if self.counts[name] % 256 == 0:
            self.streams[name].flush()
            if sum(p.stat().st_size for p in self.paths.values()) > self.max_bytes:
                raise RuntimeError("diagnostic journal exceeds 96 MiB compressed bound")

    def publish(self, output: Path, total_cap: int) -> dict[str, Any]:
        """Idempotently seal journals before terminal summaries or fault propagation."""
        if self.published is not None:
            return self.published
        if not self.closed:
            for stream in self.streams.values():
                stream.close()
            self.closed = True
        if sum(p.stat().st_size for p in self.paths.values()) > self.max_bytes:
            raise RuntimeError("diagnostic journal exceeds 96 MiB compressed bound")
        exports = {}
        for name, path in self.paths.items():
            exports[name] = publish_bounded_export(
                output / path.name,
                lambda dst, src=path: shutil.copyfile(src, dst),
                export_root=output,
                max_total_bytes=total_cap,
            )
        self.published = {
            "schema": "t34-ct-h5000-diagnostics/v1",
            "counts": self.counts,
            "exports": exports,
            "logging_only": True,
        }
        return self.published

    def close(self) -> None:
        """Release spool resources, preserving already sealed exports."""
        for stream in self.streams.values():
            stream.close()
        self.temp.cleanup()
