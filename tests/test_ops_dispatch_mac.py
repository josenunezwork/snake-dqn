"""Verify the explicit Mac dispatch route without launching a job."""

import json
import subprocess
from pathlib import Path


def test_mac_route_preserves_arguments_and_exit(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "tools/ops/dispatch.sh"
    script = tmp_path / "dispatch.sh"
    script.write_bytes(source.read_bytes())
    target = tmp_path / "mac_run.sh"
    target.write_text(
        "#!/usr/bin/env python3\nimport json,sys\n"
        "print(json.dumps(sys.argv[1:]))\nsys.exit(17)\n"
    )
    target.chmod(0o755)
    argv = ["--repo", "repo with spaces", "job", "pin", "--", "printf", "literal $()"]
    result = subprocess.run(
        ["bash", str(script), "--mac", *argv], capture_output=True, text=True, check=False
    )
    assert result.returncode == 17
    assert json.loads(result.stdout) == argv
