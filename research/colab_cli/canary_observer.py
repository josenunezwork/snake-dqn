"""Record a real CUDA action-selection call during a Colab transport canary.

The observer is deliberately inert outside a canary.  A canary should attest
that the uploaded, pinned package actually made an action-space decision on a
CUDA tensor; a CUDA availability check or a generic torch warm-up is not that
evidence.
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any


class CanaryObservationError(RuntimeError):
    """The CUDA canary observer could not make durable, attributable evidence."""


_INSTALLED = False


def _configuration() -> tuple[Path, Path, str]:
    root_value = os.environ.get("COLAB_CLI_ROOT")
    package = os.environ.get("SNAKE_COLAB_CANARY_PACKAGE")
    if not root_value or not package:
        raise CanaryObservationError(
            "CUDA canary requires COLAB_CLI_ROOT and SNAKE_COLAB_CANARY_PACKAGE"
        )
    root = Path(root_value).resolve()
    if package.startswith("/") or any(part in {"", ".", ".."} for part in package.split("/")):
        raise CanaryObservationError(
            "CUDA canary package must be a normalized repository-relative path"
        )
    package_root = (root / "repo" / package).resolve()
    try:
        package_root.relative_to((root / "repo").resolve())
    except ValueError as exc:
        raise CanaryObservationError("CUDA canary package escapes pinned repository") from exc
    if not package_root.is_dir():
        raise CanaryObservationError("CUDA canary package is absent from pinned repository")
    return root, package_root, package


def _caller(package_root: Path, repository: Path) -> str | None:
    """Return the nearest package frame, excluding torch and bootstrap frames."""
    for frame in reversed(traceback.extract_stack()[:-1]):
        try:
            path = Path(frame.filename).resolve()
            relative = path.relative_to(package_root)
        except (OSError, ValueError):
            continue
        return f"{package_root.relative_to(repository) / relative}:{frame.lineno}:{frame.name}"
    return None


def _eligible(tensor: Any) -> bool:
    shape = getattr(tensor, "shape", None)
    device = getattr(tensor, "device", None)
    if shape is None or device is None:
        return False
    try:
        return getattr(device, "type", None) == "cuda" and len(shape) >= 1 and int(shape[-1]) == 6
    except (TypeError, ValueError):
        return False


def _write_once(root: Path, observation: dict[str, Any]) -> None:
    directory = root / "canary-observations"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = directory / f"{os.getpid()}.json"
    encoded = (json.dumps(observation, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{os.getpid()}.", dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # link is exclusive and atomic: a duplicate PID record proves this
            # process already emitted its first successful action decision.
            os.link(temporary, target)
        except FileExistsError:
            return
    finally:
        temporary.unlink(missing_ok=True)


def _observe(
    operation: str,
    tensor: Any,
    invoke: Callable[[], Any],
    *,
    root: Path,
    package_root: Path,
    package: str,
    torch: Any,
) -> Any:
    result = invoke()
    if not _eligible(tensor):
        return result
    caller = _caller(package_root, root / "repo")
    if caller is None:
        return result
    # A call only becomes evidence after its result exists, it is attributable
    # to the requested package, CUDA has completed it, and the row is durable.
    try:
        torch.cuda.synchronize(tensor.device)
        _write_once(
            root,
            {
                "version": 1,
                "pid": os.getpid(),
                "root": str(root),
                "package": package,
                "cuda_decisions": 1,
                "operation": operation,
                "caller": caller,
                "device": str(tensor.device),
                "shape": [int(value) for value in tensor.shape],
            },
        )
    except BaseException as exc:
        raise CanaryObservationError(
            f"CUDA canary observation failed after {operation}: {exc}"
        ) from exc
    return result


def install() -> None:
    """Install action-selection wrappers when this interpreter is a CUDA canary."""
    global _INSTALLED
    if os.environ.get("SNAKE_COLAB_CANARY") != "1" or _INSTALLED:
        return
    root, package_root, package = _configuration()
    try:
        import torch
    except BaseException as exc:
        raise CanaryObservationError("CUDA canary could not import torch") from exc

    tensor_argmax = torch.Tensor.argmax
    torch_argmax = torch.argmax
    multinomial = torch.multinomial

    def wrapped_tensor_argmax(tensor: Any, *args: Any, **kwargs: Any) -> Any:
        return _observe(
            "Tensor.argmax",
            tensor,
            lambda: tensor_argmax(tensor, *args, **kwargs),
            root=root,
            package_root=package_root,
            package=package,
            torch=torch,
        )

    def wrapped_argmax(tensor: Any, *args: Any, **kwargs: Any) -> Any:
        return _observe(
            "torch.argmax",
            tensor,
            lambda: torch_argmax(tensor, *args, **kwargs),
            root=root,
            package_root=package_root,
            package=package,
            torch=torch,
        )

    def wrapped_multinomial(tensor: Any, *args: Any, **kwargs: Any) -> Any:
        return _observe(
            "torch.multinomial",
            tensor,
            lambda: multinomial(tensor, *args, **kwargs),
            root=root,
            package_root=package_root,
            package=package,
            torch=torch,
        )

    torch.Tensor.argmax = wrapped_tensor_argmax
    torch.argmax = wrapped_argmax
    torch.multinomial = wrapped_multinomial
    _INSTALLED = True


def collect(root: Path | str, package: str) -> list[dict[str, Any]]:
    """Load and strictly validate per-process CUDA observations for a canary."""
    resolved_root = Path(root).resolve()
    directory = resolved_root / "canary-observations"
    if not directory.is_dir():
        raise CanaryObservationError("CUDA canary produced no observation directory")
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CanaryObservationError(f"invalid CUDA canary observation: {path.name}") from exc
        if not isinstance(value, dict):
            raise CanaryObservationError(f"invalid CUDA canary observation: {path.name}")
        required = ("root", "package", "cuda_decisions", "operation", "caller", "device", "shape")
        if any(key not in value for key in required):
            raise CanaryObservationError(f"incomplete CUDA canary observation: {path.name}")
        if value["root"] != str(resolved_root) or value["package"] != package:
            raise CanaryObservationError(f"foreign CUDA canary observation: {path.name}")
        if not isinstance(value["caller"], str) or not value["caller"].startswith(package + "/"):
            raise CanaryObservationError(f"unattributed CUDA canary observation: {path.name}")
        shape = value["shape"]
        if (
            type(value["cuda_decisions"]) is not int
            or value["cuda_decisions"] <= 0
            or not isinstance(shape, list)
            or not shape
            or type(shape[-1]) is not int
            or not isinstance(value["device"], str)
            or not value["device"].startswith("cuda")
            or shape[-1] != 6
        ):
            raise CanaryObservationError(f"invalid CUDA decision observation: {path.name}")
        records.append(value)
    if not records:
        raise CanaryObservationError("CUDA canary produced no real action decision")
    return records
