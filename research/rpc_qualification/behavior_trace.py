"""Pure evidence projection shared by differential tests and offline validation."""

from __future__ import annotations

import re


def strip_evidence(value, *, preserve_errors: bool = False):
    """Remove evidence-only fields while retaining behavior-bearing facts."""
    evidence_keys = {
        "operation_id",
        "attempt_id",
        "logical_operation_id",
        "argument_sha256",
        "request_id",
        "request_sha256",
        "returned_request_id",
        "returned_request_sha256",
        "reply_kind",
        "reply_sha256",
        "response_sha256",
        "started_monotonic",
        "finished_monotonic",
        "elapsed_seconds",
        "call_site",
        "evidence_role",
        "transfer_role",
        "transfer_plane",
        "parse_success",
        "failure_class",
        "parent_transfer_id",
        "child_transfer_id",
        "transfer_id",
        "whole_file_sha256",
    }
    if isinstance(value, dict):
        return {
            key: strip_evidence(item, preserve_errors=(key == "outcome"))
            for key, item in value.items()
            if key not in evidence_keys and (preserve_errors or key != "error")
        }
    if isinstance(value, list):
        return [strip_evidence(item, preserve_errors=preserve_errors) for item in value]
    if isinstance(value, str):
        value = re.sub(
            r"/(?:private/)?var/folders/[^ ]+?/(?:tmp|T)/[^ ]+\.(?:py|json)",
            "<file>",
            value,
        )
        return value.replace("/tmp/fake-config", "<config>")
    return value
