"""Frozen identity and code generation for the response-loss qualification.

The selected RPC body deliberately contains no supervisor generation path. A
qualification workload publishes one create-only locator per frozen unit ID;
the runtime body resolves that locator and verifies the complete frozen
identity before it invokes the unit. Thus the pack and the dispatched request
hash the same exact UTF-8 code.
"""

from __future__ import annotations

import hashlib

LOCATOR_ROOT = "/content/.snake-rpc-qualification-locators"


def locator_for_unit(unit_id: str) -> str:
    """Return the fixed, collision-resistant runtime locator for ``unit_id``."""
    if not isinstance(unit_id, str) or not unit_id:
        raise ValueError("injection locator needs a nonempty unit ID")
    return f"{LOCATOR_ROOT}/{hashlib.sha256(unit_id.encode()).hexdigest()}.json"


def _require_identity(
    *, namespace: str, allocation: str, index: int, unit_id: str, request_id: str
) -> None:
    """Reject incomplete or internally inconsistent frozen unit identities."""
    if (
        not isinstance(namespace, str)
        or not namespace
        or not isinstance(allocation, str)
        or not allocation
        or not isinstance(index, int)
        or index < 0
        or not isinstance(unit_id, str)
        or unit_id != f"{namespace}/{allocation}/{index:08d}"
        or not isinstance(request_id, str)
        or len(request_id) != 32
        or any(character not in "0123456789abcdef" for character in request_id)
    ):
        raise ValueError("injection RPC code needs one complete frozen unit identity")


def unit_rpc_code(
    *,
    locator: str,
    namespace: str,
    allocation: str,
    index: int,
    unit_id: str,
    request_id: str,
) -> str:
    """Generate the exact generation-independent body for one injected RPC."""
    if not isinstance(locator, str) or not locator:
        raise ValueError("injection RPC code needs a frozen locator")
    _require_identity(
        namespace=namespace,
        allocation=allocation,
        index=index,
        unit_id=unit_id,
        request_id=request_id,
    )
    return (
        "import hashlib, json, os, subprocess\nfrom pathlib import Path\n"
        f"locator=Path({locator!r})\n"
        "matches=list(locator.parent.glob(locator.name))\n"
        "assert len(matches)==1 and matches[0].is_file()\n"
        "spec=json.loads(matches[0].read_text())\n"
        f"assert spec['namespace']=={namespace!r} and spec['allocation']=={allocation!r}\n"
        f"assert spec['index']=={index!r} and spec['unit_id']=={unit_id!r}\n"
        f"assert spec['request_id']=={request_id!r}\n"
        "descriptor=Path(spec['descriptor_path'])\n"
        "assert descriptor.is_file()\n"
        "assert hashlib.sha256(descriptor.read_bytes()).hexdigest()==spec['descriptor_sha256']\n"
        "descriptor_value=json.loads(descriptor.read_text())\n"
        "assert all(descriptor_value[key]==spec[key] for key in "
        "('namespace','allocation','index','unit_id','request_id'))\n"
        "root=Path(spec['runtime_root'])\n"
        "assert root.is_dir() and spec['generation']==root.name\n"
        "env=dict(os.environ, PYTHONPATH=str(root/'frozen')+os.pathsep+str(root/'repo'))\n"
        "py=str(root/'.venv'/'bin'/'python')\n"
        "subprocess.run([py,'-m','research.rpc_qualification.loop','unit',"
        "'--descriptor',str(descriptor),'--request-id',spec['request_id']],"
        "env=env,cwd=root/'frozen',check=True)\n"
        "result=json.loads(descriptor.read_text())\n"
    )


def unit_rpc_sha256(
    *,
    locator: str,
    namespace: str,
    allocation: str,
    index: int,
    unit_id: str,
    request_id: str,
) -> str:
    """Hash exactly the UTF-8 request body produced by :func:`unit_rpc_code`."""
    return hashlib.sha256(
        unit_rpc_code(
            locator=locator,
            namespace=namespace,
            allocation=allocation,
            index=index,
            unit_id=unit_id,
            request_id=request_id,
        ).encode("utf-8")
    ).hexdigest()
