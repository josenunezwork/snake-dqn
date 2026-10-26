"""Repeat existing label-blind checks without replacing any sealed artifact.

The original create-only verifier cannot run twice. Execute its unchanged imports,
helpers, and verification prefix, stopping before its first artifact write. Bind
this revalidation to the original verifier and report bytes.
"""

import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path('/Users/josenunez/Projects/ml/snake-dqn-f5planpub')
    dest = root / 'research/bl_residual/ledger/F5_W_PLAN_SEAL_VERIFICATION_20261026'
    artifact = Path(
        '/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024'
    ) / dest.name
    source = dest / 'VERIFY_PLAN_SEAL.py'
    report_path = dest / 'PLAN_SEAL_VERIFICATION.json'
    report = json.loads(report_path.read_bytes())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report['verifier_script_sha256']
    tree = ast.parse(source.read_text(), filename=str(source))
    original_main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main'
    )
    assert ast.unparse(original_main.body[0]) == 'assert not ARTIFACT.exists() and (not DEST.exists())'
    stop = next(
        i for i, node in enumerate(original_main.body)
        if isinstance(node, ast.Expr) and ast.unparse(node) == 'ARTIFACT.mkdir()'
    )
    original_main.body = original_main.body[1:stop]
    tree.body = [node for node in tree.body if not isinstance(node, ast.If)]
    namespace = {'__file__': str(source), '__name__': 'sealed_verifier_read_only_revalidation'}
    exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), namespace)
    namespace['main']()
    checks = namespace['CHECKS']
    for name, result in checks.items():
        expected = report['checks'][name]
        if name == 'publication_precedes_verification':
            result = json.loads(json.dumps(result))
            result['evidence']['origin_main'] = expected['evidence']['origin_main']
        assert result == expected, name
    assert len(checks) == 12
    claim = json.loads((dest / 'PREFIT_STAGE_CLAIM.json').read_bytes())
    namespace['verify_claim'](dest / 'PREFIT_STAGE_CLAIM.json', {
        key: claim[key] for key in (
            'namespace', 'allocation_sha256', 'source_sha256',
            'selection_sha256', 'batch_plan_sha256'
        )
    })
    assert claim['batch_plan_sha256'] == report['internal_plan_sha256']
    from research.bl_ppo.exports import export_bytes, export_json, verify_export

    for path in sorted(dest.iterdir()):
        if path.name == Path(__file__).name:
            continue
        assert path.read_bytes() == (artifact / path.name).read_bytes()
        if not path.name.endswith(('.sha256', '.complete')):
            verify_export(path)
    result = {
        'schema': 'bl-f5-plan-seal-read-only-revalidation/v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS',
        'checks_repeated_and_equal_to_seal': list(checks),
        'prospective_claim_reverified': True,
        'plan_seal_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'original_verifier_sha256': report['verifier_script_sha256'],
        'revalidation_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'output_publication_commit': report['output_publication_commit'],
        'origin_main': subprocess.check_output(
            ['git', 'rev-parse', 'origin/main'], cwd=root
        ).decode().strip(),
        'scope': 'Same allowed actual-plan/status-only checks; no sampler replay or plan replacement.',
        'fit_authorized': False,
        'provider_calls': 0,
        'dispatches': 0,
    }
    export_bytes(artifact / Path(__file__).name, Path(__file__).read_bytes())
    export_json(artifact / 'PLAN_SEAL_REVALIDATION.json', result)
    for path in sorted(artifact.iterdir()):
        target = dest / path.name
        if target.exists():
            assert target.read_bytes() == path.read_bytes()
        else:
            with target.open('xb') as stream:
                stream.write(path.read_bytes())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
