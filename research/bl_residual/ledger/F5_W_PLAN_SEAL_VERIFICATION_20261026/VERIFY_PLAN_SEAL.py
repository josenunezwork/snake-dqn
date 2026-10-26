"""Verify only returned row indices/RNG and admitted status-only F5 metadata on Mac.

Never call verify_batch_plan/materialize_batch_plan, read_selected_package, fit,
CUDA, provider APIs, or open original/selected label or outcome payloads.
"""
import gzip
import hashlib
import json
import platform
import random
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from research.bl_ppo.exports import export_bytes, export_json, verify_export
from research.bl_residual.f5_data import metadata_cache, selected_corpus_identity
from research.bl_residual.f5_environment import compare_environment, receipt_sha256
from research.bl_residual.f5_fit import BatchPlan, F5Recipe, RngBoundary, _digest_plan
from research.bl_residual.f5_job import _source_admission
from research.bl_residual.f5_protocol import stage_claim, verify_claim

ROOT = Path('/Users/josenunez/Projects/ml/snake-dqn-f5planpub')
BASE = Path('/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024')
PACKAGE = BASE / 'F5_W_PLAN_OPERATIONAL_20261026_ROUND08_02_SUCCESSOR_REQUAL'
LEDGER = ROOT / 'research/bl_residual/ledger' / PACKAGE.name
ARTIFACT = BASE / 'F5_W_PLAN_SEAL_VERIFICATION_20261026'
DEST = ROOT / 'research/bl_residual/ledger' / ARTIFACT.name
PUBLICATION_COMMIT = '048e50de86b6728b705d74dd0dbd869acfa3e794'
CHECKS = {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def checked(condition, name, evidence):
    if not condition:
        raise ValueError(name)
    CHECKS[name] = {'status': 'PASS', 'evidence': evidence}


def tensor_sha(value):
    return sha(value.detach().cpu().contiguous().numpy().tobytes())


def binding(path):
    return {'path': str(path), 'sha256': file_sha(path), 'size': path.stat().st_size}


def verify_publication():
    subprocess.run(['git', 'merge-base', '--is-ancestor', PUBLICATION_COMMIT, 'origin/main'], check=True)
    publication = LEDGER / 'OUTPUT_PUBLICATION_01.json'
    verify_export(publication)
    report = json.loads(publication.read_bytes())
    for rel, row in report['published_files'].items():
        path = LEDGER / rel
        raw = path.read_bytes()
        remote = subprocess.check_output(['git', 'show', f'origin/main:{path.relative_to(ROOT)}'])
        assert raw == remote and sha(raw) == row['sha256'] and len(raw) == row['size']
    for row in report['entries']:
        rep = row['representation']
        if 'path' in rep:
            raw = (LEDGER / rep['path']).read_bytes()
        else:
            index = json.loads((LEDGER / rep['index']).read_bytes())
            compressed = b''.join((LEDGER / 'output-01' / p['path']).read_bytes() for p in index['parts'])
            assert sha(compressed) == index['compressed_sha256']
            raw = gzip.decompress(compressed)
        assert sha(raw) == row['sha256'] and len(raw) == row['size']
        assert raw == Path(row['source_path']).read_bytes()
    # Explicit original-file coverage, including unchanged original sidecar bytes.
    original_files = sorted(p for p in (PACKAGE / 'output-01').rglob('*') if p.is_file())
    expected = {r['relative_path'] for r in report['entries']}
    assert {str(p.relative_to(PACKAGE / 'output-01')) for p in original_files if not p.name.endswith(('.sha256', '.complete'))} == expected
    for path in original_files:
        if path.name.endswith(('.sha256', '.complete')):
            assert path.read_bytes() == (LEDGER / 'output-01' / path.relative_to(PACKAGE / 'output-01')).read_bytes()
    checked(True, 'publication_precedes_verification', {'commit': PUBLICATION_COMMIT, 'origin_main': subprocess.check_output(['git', 'rev-parse', 'origin/main']).decode().strip(), 'representation_blobs': len(report['published_files']), 'opaque_payloads': len(report['entries']), 'all_original_files': len(original_files), 'local_equals_origin': True, 'all_payloads_byte_exact': True})


def rng_boundary(value):
    assert isinstance(value, RngBoundary)
    assert isinstance(value.python_rng_state, tuple) and len(value.python_rng_state) == 3
    random.Random().setstate(value.python_rng_state)
    n = value.numpy_rng_state
    assert isinstance(n, tuple) and len(n) == 5 and n[0] == 'MT19937'
    assert isinstance(n[1], np.ndarray) and n[1].dtype == np.uint32 and n[1].shape == (624,)
    np.random.RandomState().set_state(n)
    c = value.torch_cpu_rng_state
    assert c.device.type == 'cpu' and c.dtype == torch.uint8 and c.shape == (5056,)
    torch.Generator(device='cpu').set_state(c)
    assert isinstance(value.torch_cuda_rng_states, tuple) and len(value.torch_cuda_rng_states) == 1
    for state in value.torch_cuda_rng_states:
        assert state.dtype == torch.uint8 and state.shape == (16,) and state.device.type == 'cpu'
    assert value.named_generator_states == ()
    return {'type': 'RngBoundary', 'python_state_sha256': sha(canonical(value.python_rng_state)), 'python_version': value.python_rng_state[0], 'numpy_algorithm': n[0], 'numpy_state_sha256': sha(n[1].tobytes()), 'numpy_position': int(n[2]), 'numpy_has_gauss': int(n[3]), 'numpy_cached_gaussian': float(n[4]), 'torch_cpu': {'dtype': 'uint8', 'shape': list(c.shape), 'sha256': tensor_sha(c)}, 'torch_cuda': [{'dtype': 'uint8', 'shape': list(s.shape), 'sha256': tensor_sha(s)} for s in value.torch_cuda_rng_states], 'named_generators': [], 'cuda_restore_executed': False}


def main():
    assert not ARTIFACT.exists() and not DEST.exists()
    assert platform.system() == 'Darwin' and platform.machine() == 'arm64'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    verify_publication()
    out = PACKAGE / 'output-01'
    receipt = json.loads((out / 'receipt.json').read_bytes())
    terminal = json.loads((out / 'terminal.json').read_bytes())
    checked(receipt['status'] == 'PASS' and receipt['stop_verified'] is True and receipt['cleanup_error'] is None and terminal['status'] == 'succeeded' and terminal['returncode'] == 0, 'terminal_and_saved_release', {'receipt': binding(out / 'receipt.json'), 'elapsed_seconds': receipt['elapsed_seconds'], 'saved_sessions': json.loads((LEDGER / 'release-state-01/sessions.json').read_bytes()), 'endpoint': receipt['endpoint'], 'generation': receipt['generation']})
    assert json.loads((LEDGER / 'release-state-01/sessions.json').read_bytes()) == {}
    exports = {r['path']: r for r in receipt['exports']}
    for name, row in exports.items():
        path = out / 'exports' / name
        assert verify_export(path) == row['sha256'] and path.stat().st_size == row['size']
    checked(len(exports) == 9, 'nine_immutable_exports', {'inventory': receipt['exports'], 'paired_fit_state_payloads': 'opaque hash/copy only; never deserialized'})
    control_path = PACKAGE / 'control.json'
    ledger_path = PACKAGE / 'verified-ledger.json'
    verify_export(control_path)
    verify_export(ledger_path)
    control = json.loads(control_path.read_bytes())
    ledger = json.loads(ledger_path.read_bytes())
    assert control['allocation'] == sha(canonical({k: v for k, v in control.items() if k not in ('allocation', 'ledger_sha256')}))
    assert control['ledger_sha256'] == file_sha(ledger_path)
    checked(ledger['namespace'] == control['namespace'] and ledger['allocation'] == control['allocation'] and ledger['allocation_sha256'] == control['identity']['allocation_sha256'] and ledger['units'] == [] and ledger['status'] == 'EMPTY_VERIFIED_LEDGER_NO_COMPLETED_UNITS', 'empty_paired_ledger_and_control', {'control': binding(control_path), 'control_canonical_sha256': sha(canonical(control)), 'control_allocation': control['allocation'], 'ledger': binding(ledger_path), 'units': 0})
    _source_admission(control)  # Source/config only; no environment collection or data read.
    for rel, digest in control['source_hashes'].items():
        raw = subprocess.check_output(['git', 'show', f"{control['source_commit']}:{rel}"])
        assert sha(raw) == digest
    checked(control['source_commit'] == receipt['commit'] and control['scientific_admission'] is False, 'published_runtime_source_and_loss', {'source_commit': control['source_commit'], 'source_files': len(control['source_hashes']), 'source_map_sha256': sha(canonical(control['source_hashes'])), 'loss_sha256': control['identity']['loss_sha256'], 'current_and_pinned_git_bytes_equal': True})
    plan_path = out / 'exports/prepare/f5-batch-plan.pt'
    prepare = json.loads((out / 'exports/prepare/prepare-receipt.json').read_bytes())
    plan = torch.load(plan_path, map_location='cpu', weights_only=False)  # Only the authenticated row-index/RNG payload.
    checked(isinstance(plan, BatchPlan) and plan.recipe == F5Recipe(arm='C') and plan.indices.shape == (2000, 256) and plan.indices.dtype == torch.int64 and plan.indices.device.type == 'cpu', 'actual_plan_shape_and_recipe', {'shape': list(plan.indices.shape), 'entries': plan.indices.numel(), 'dtype': str(plan.indices.dtype), 'recipe': asdict(plan.recipe), 'payload': binding(plan_path)})
    checked(asdict(plan.identity) == control['identity'] and plan.sha256 == _digest_plan(plan.indices, plan.initial_generator_state, plan.final_generator_state, plan.identity, plan.recipe) and prepare['batch_plan_sha256'] == plan.sha256 and prepare['plan_export_sha256'] == file_sha(plan_path) and prepare['rows'] == 512000 and prepare['allocation'] == control['allocation'] and prepare['namespace'] == control['namespace'] and prepare['kind'] == 'PREPARED_COMMON_BATCH_PLAN', 'plan_internal_digest_and_prepare_bindings', {'internal_plan_sha256': plan.sha256, 'identity': asdict(plan.identity), 'prepare_receipt': binding(out / 'exports/prepare/prepare-receipt.json')})
    folder = BASE / 'f5-selected960-49fd1dfa'
    scan_path, inventory_path = folder / 'scan.json', folder / 'extractor-input-inventory.json'
    assert file_sha(scan_path) == '8b4910213475266038c8a6a65ad9aea00cd301eba08311f44bb59bc23c312ac4'
    assert file_sha(inventory_path) == 'cc20e1e7f5c321de5a1f63b94945c92c15c852a8499e88ab56659d1d2d7c1615'
    metadata = metadata_cache(json.loads(scan_path.read_bytes()), json.loads(inventory_path.read_bytes()))
    packaging_path = BASE / 'F5_W_PREFIT_20261025/inputs-02/packaging-receipt.json'
    verify_export(packaging_path)
    packaging = json.loads(packaging_path.read_bytes())
    assert selected_corpus_identity(metadata, packaging['selected_payload_sha256']) == plan.identity.corpus_sha256 == packaging['corpus_sha256']
    assert prepare['compact_package_sha256'] == packaging['package_sha256'] == control['input_hashes']['selected-training.tar.xz']
    ordinary = torch.tensor([r['ordinary'] for r in metadata.rows], dtype=torch.bool)
    assert plan.indices.min().item() >= 0 and plan.indices.max().item() < len(metadata.rows)
    mapped = ordinary[plan.indices]
    checked(not mapped[:, :128].any().item() and mapped[:, 128:].all().item(), 'every_batch_ordered_128_doom_then_128_ordinary', {'updates_checked': 2000, 'doom_entries': 256000, 'ordinary_entries': 256000, 'index_min': plan.indices.min().item(), 'index_max': plan.indices.max().item(), 'eligible_compact_rows': len(metadata.rows), 'doom_eligibility_replay': 'not performed; would require death labels'})
    original_indices = np.asarray([r['source_row_index'] for r in metadata.rows], dtype='<i8')[plan.indices.numpy()]
    rowids = hashlib.sha256()
    for index in plan.indices.flatten().tolist():
        raw = metadata.rows[index]['row_id'].encode()
        rowids.update(len(raw).to_bytes(8, 'little'))
        rowids.update(raw)
    checked(len(metadata.selected_lives) == 960 and len(metadata.rows) == packaging['row_count'] and metadata.source.sha256 == packaging['original_source']['sha256'], 'original_rows_corpus_exclusions_and_order', {'scan': binding(scan_path), 'inventory': binding(inventory_path), 'packaging_receipt': binding(packaging_path), 'corpus_sha256': plan.identity.corpus_sha256, 'compact_projection_sha256': packaging['projection_sha256'], 'ordered_metadata_rows_sha256': sha((json.dumps(metadata.rows, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()), 'ordered_plan_indices_sha256': tensor_sha(plan.indices), 'ordered_original_source_indices_sha256': sha(original_indices.tobytes()), 'ordered_original_row_ids_sha256': rowids.hexdigest(), 'row_id_hash_encoding': 'each UTF-8 row_id prefixed by uint64 little-endian byte length, update-major', 'selected_lives': len(metadata.selected_lives), 'eligible_lives': len(metadata.eligible_lives), 'excluded_parents': sorted(metadata.excluded_parents), 'original_source': asdict(metadata.source), 'selected_payload_not_opened_or_hashed': True})
    generator_states = {}
    for name in ('initial_generator_state', 'final_generator_state'):
        value = getattr(plan, name)
        assert value.dtype == torch.uint8 and value.shape == (5056,) and value.device.type == 'cpu'
        torch.Generator(device='cpu').set_state(value)
        generator_states[name] = {'dtype': 'uint8', 'shape': [5056], 'sha256': tensor_sha(value)}
    assert torch.equal(torch.Generator(device='cpu').manual_seed(3403).get_state(), plan.initial_generator_state)
    assert not torch.equal(plan.initial_generator_state, plan.final_generator_state)
    initial, final = rng_boundary(plan.initial_rng), rng_boundary(plan.final_rng)
    checked(initial == final, 'typed_dedicated_and_global_rng_boundaries', {'dedicated_cpu_generator': generator_states, 'seed_3403_initial_state_equal': True, 'dedicated_final_state_is_valid_and_changed': True, 'initial_global': initial, 'final_global': final, 'global_rng_unchanged': True, 'state_bytes_preserved_in_actual_plan_payload': True, 'binding_caveat': 'global RNGs are bound by payload SHA, not internal plan digest', 'sampler_replay_or_plan_regeneration': False})
    env = prepare['environment']
    assert env['reference_raw'] == control['environment_capture']
    comparison = compare_environment(control['environment_capture'], env['assignment_raw'], mode='cross-assignment')
    assert comparison == env['comparison']
    assert prepare['closing_environment'] == compare_environment(env['assignment_raw'], env['assignment_raw'], mode='same-assignment')
    assignment = json.loads((out / 'exports/paired/assignment-environment.json').read_bytes())
    checked(assignment == env['assignment_raw'] and assignment['environment_descriptor_sha256'] == plan.identity.environment_sha256, 'admitted_v5_environment_and_closing_boundary', {'reference_raw_sha256': receipt_sha256(env['reference_raw']), 'actual_raw_sha256': receipt_sha256(assignment), 'cross_assignment_invariants_equal': True, 'closing_same_assignment_raw_hashes_equal': True, 'closing_capture_scope': 'producer-attested same-assignment report; recomputed report equality on exported raw assignment', 'gpu': assignment['gpu'], 'gpu_uuid': assignment['gpu_uuid'], 'cpu_count': assignment['cpu_count'], 'physical_or_numerical_cross_assignment_equivalence_claimed': False})
    qualification = json.loads((out / 'exports/qualification/qualification.json').read_bytes())
    checked(qualification['status'] == 'COMPLETE_TRANSPORT_ONLY_NOT_TIMING_QUALIFIED' and qualification['fit_or_optimizer_executed'] is False, 'scope_transport_only_no_fit', {'qualification_status': qualification['status'], 'fit_or_optimizer_executed': False, 'scientific_admission': False})
    selection = ROOT / 'research/bl_residual/ledger/F5_REGISTRATION_20261025/F5_SELECTION_W_20261025_20261026T032844Z.json'
    assert file_sha(selection) == 'b4a880816bcd050cc36aebed7f73547a37a5fc6890889fff7203771a6449d3c3'
    # This is a prospective claim, not a consumed outcome-look token or fit admission.
    ARTIFACT.mkdir()
    claim_expected = {'namespace': control['namespace'], 'allocation_sha256': plan.identity.allocation_sha256, 'source_sha256': plan.identity.source_sha256, 'selection_sha256': file_sha(selection), 'batch_plan_sha256': plan.sha256}
    stage_claim(ARTIFACT / 'PREFIT_STAGE_CLAIM.json', **claim_expected)
    verify_claim(ARTIFACT / 'PREFIT_STAGE_CLAIM.json', claim_expected)
    checked(True, 'prospective_stage_claim_unconsumed', {'claim': binding(ARTIFACT / 'PREFIT_STAGE_CLAIM.json'), 'look_token_consumed': False, 'fit_admission': False})
    remaining = [
        'Verify and publish separately accepted timing-02 L4 output: all 12 disjoint source-native workloads/parity, optimizer/resume calibration, immutable exports, actual environment and verified release. No claim about concurrent package state is made here.',
        'Complete uncontended Mac companion timings/calibration and measured feasibility forecasts; registered two workers nice15 and one math thread; production cache/export and full inventory forecasts; transfer <=150 s and whole staging <=300 s; retain original cumulative elapsed and remaining envelopes without reset.',
        'Seal exact final science inventory/control/actual plan/independent acceptance/empty paired ledger/prospective sole-stage claim and token; bind with f5_package.bind_qualification_inventory; resolve its local-upload final-fit transport requirement; run and verify matching current-source full-inventory source-native canary after files exist; publish independent actual-plan/final operational ACCEPT_F5_FIT before either fit.'
    ]
    report = {'schema': 'bl-f5-plan-seal-verification/v1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'decision': 'PASS_LABEL_BLIND_PLAN_SEAL_VERIFICATION', 'fit_authorized': False, 'job_id': receipt['job_id'], 'output_publication_commit': PUBLICATION_COMMIT, 'actual_plan': binding(plan_path), 'internal_plan_sha256': plan.sha256, 'mac': {'system': platform.system(), 'machine': platform.machine(), 'python': platform.python_version(), 'torch': torch.__version__, 'cpu_threads': 1, 'cuda_executed': False}, 'checks': CHECKS, 'remaining_gates': remaining, 'scope_limitations': ['Every required structural/order/stratum/RNG/identity check above uses actual returned plan plus status-only metadata.', 'Death-label-dependent doom eligibility and exact life-uniform sampler regeneration were not repeated on Mac. They remain bound to the accepted source and L4 producer; no replacement plan was generated.', 'CPU RNG state validation on Mac is structural and seed-initial-state verification, not a numerical cross-platform or CUDA restore proof.', 'No label/outcome/selected-training payload opened or hashed; paired payloads handled only as opaque bytes; no numerical science look or fit, Colab, dispatch, timing package/worktree access, serving change or release.'], 'authority': ['ACCEPT_F5_W_IMPLEMENTATION_5546b6d3.md: Publish before fitting item 3', 'ACCEPT_F5_W_NEXT_PACKAGES_20261026_ROUND08.md: step 5', 'ACCEPT_F5_W_PLAN_SUCCESSOR_REQUAL.md', 'F5_COMMON_ENVIRONMENT_RNG_AMENDMENT_20261026.md', 'F5_BATCH_PLAN_ORDERING_ERRATUM_20261026.md'], 'verifier_script_sha256': file_sha(Path(__file__))}
    export_bytes(ARTIFACT / 'VERIFY_PLAN_SEAL.py', Path(__file__).read_bytes())
    export_json(ARTIFACT / 'PLAN_SEAL_VERIFICATION.json', report)
    lines = ['# F5 actual L4 plan seal verification', '', '**PASS_LABEL_BLIND_PLAN_SEAL_VERIFICATION.** No fit admission or outcome-look authority.', '', f'Output evidence was published first in `{PUBLICATION_COMMIT}` and verified against origin/main.', '', f'Actual plan payload SHA256: `{file_sha(plan_path)}`.', f'Internal plan digest: `{plan.sha256}`.', '', '| Check | Result |', '|---|---|']
    lines += [f'| {name} | PASS |' for name in CHECKS]
    lines += ['', 'All 2,000 batches have 128 doom indices followed by 128 ordinary indices; all 512,000 indices map to the original frozen status-only row projection. Typed dedicated and global RNG bytes are retained in the actual plan payload. No Mac replacement plan or label-dependent sampler replay occurred. Exact doom eligibility remains the reviewed L4 producer obligation.', '', '## Remaining gates', '']
    lines += [f'{i}. {gate}' for i, gate in enumerate(remaining, 1)]
    lines += ['', 'The prospective PREFIT_STAGE_CLAIM is unconsumed. This seal binds the returned bytes for independent review; it does not issue ACCEPT_F5_FIT. No label/outcome read, dispatch, Colab call, timing-worktree/package access, serving or release change occurred.', '']
    export_bytes(ARTIFACT / 'PLAN_SEAL_VERIFICATION.md', '\n'.join(lines).encode())
    for path in sorted(ARTIFACT.iterdir()):
        target = DEST / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(path.read_bytes())
    print(json.dumps({'decision': report['decision'], 'checks_passed': len(CHECKS), 'plan_payload_sha256': file_sha(plan_path), 'plan_seal': str(ARTIFACT / 'PLAN_SEAL_VERIFICATION.json'), 'plan_seal_sha256': file_sha(ARTIFACT / 'PLAN_SEAL_VERIFICATION.json'), 'rows': 512000, 'fit_authorized': False}))


if __name__ == '__main__':
    main()
