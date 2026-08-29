from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


_ACTIVE_TEST_LEASE = None


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(repo_root):
    from mambapose_opt.schema import CandidateSpec

    config = repo_root / 'configs/candidate.py'
    checkpoint = repo_root / 'work_dirs/reproduction/parent.pth'
    config.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    config.write_text('model = dict(type="Fixture")\n')
    checkpoint.write_bytes(b'parent checkpoint')
    inventory = repo_root / 'data/inventory.json'
    authority = repo_root / 'optimization/coco_val2017_authority.json'
    inventory.parent.mkdir(parents=True, exist_ok=True)
    authority.parent.mkdir(parents=True, exist_ok=True)
    authority.write_text(json.dumps({
        'schema_version': 1, 'dataset': 'coco', 'split': 'val2017',
        'annotation': {
            'path': 'data/coco/annotations/person_keypoints_val2017.json',
            'sha256': '1' * 64, 'image_count': 5000,
            'annotation_count': 11004,
            'inventory_asset_id': 'coco-annotations',
            'inventory_archive_sha256': '3' * 64,
        },
        'images': {
            'prefix': 'data/coco/val2017', 'image_count': 5000,
            'corpus_digest_algorithm': 'sha256-filename-size-content-v1',
            'corpus_sha256': '6' * 64,
            'inventory_asset_id': 'coco-val2017',
            'inventory_archive_sha256': '4' * 64,
        },
        'detections': {
            'path': ('data/coco/person_detection_results/'
                     'COCO_val2017_detections_AP_H_56_person.json'),
            'sha256': '2' * 64, 'record_count': 104125,
            'inventory_asset_id': 'coco-val-detections',
        },
    }))
    inventory.write_text(json.dumps({
        'schema_version': 1, 'assets': [
            {'id': 'coco-annotations', 'sha256': '3' * 64},
            {'id': 'coco-val2017', 'sha256': '4' * 64},
            {'id': 'coco-val-detections', 'sha256': '2' * 64},
        ],
    }))
    authority_value = json.loads(authority.read_text())
    authority_value['inventory'] = {
        'path': 'data/inventory.json', 'sha256': _sha256(inventory)}
    authority.write_text(json.dumps(authority_value))
    candidate = CandidateSpec.from_dict({
        'id': 'fixture',
        'route': 'accuracy-first',
        'kind': 'float',
        'config': 'configs/candidate.py',
        'checkpoint': 'work_dirs/reproduction/parent.pth',
        'checkpoint_sha256': _sha256(checkpoint),
        'seed': 0,
        'features': {},
    })
    manifest = repo_root / 'optimization/candidates.json'
    manifest.write_text(json.dumps({
        'schema_version': 1,
        'candidates': [{
            'id': candidate.id, 'route': candidate.route,
            'kind': candidate.kind, 'config': candidate.config.as_posix(),
            'checkpoint': candidate.checkpoint.as_posix(),
            'checkpoint_sha256': candidate.checkpoint_sha256,
            'seed': candidate.seed, 'features': dict(candidate.features),
        }],
    }))
    if not (repo_root / '.git').exists():
        (repo_root / '.gitignore').write_text('work_dirs/\n')
        subprocess.run(['git', 'init', '-q'], cwd=repo_root, check=True)
        subprocess.run(
            ['git', 'add', '.gitignore', 'configs/candidate.py',
             'optimization/candidates.json',
             'optimization/coco_val2017_authority.json',
             'data/inventory.json'], cwd=repo_root, check=True)
        subprocess.run(
            ['git', '-c', 'user.name=Fixture',
             '-c', 'user.email=fixture@example.com', 'commit', '-qm',
             'fixture source'], cwd=repo_root, check=True)
    return candidate


def _pwl_candidate(repo_root):
    from mambapose_opt.schema import CandidateSpec

    parent = _candidate(repo_root)
    return CandidateSpec.from_dict({
        'id': parent.id,
        'route': 'ssm-quant-pwl',
        'kind': 'pwl',
        'config': parent.config.as_posix(),
        'checkpoint': parent.checkpoint.as_posix(),
        'checkpoint_sha256': parent.checkpoint_sha256,
        'seed': parent.seed,
        'features': {'numeric_kind': 'pwl'},
    })


def _stub_pwl_controller_validation(
        tmp_path, monkeypatch, *, stage, authority):
    from types import SimpleNamespace

    from mambapose_opt import controller as controller_module
    from mambapose_opt import numeric_runtime
    from mambapose_opt.controller import OptimizationController

    candidate = _pwl_candidate(tmp_path)
    output = (
        tmp_path / 'work_dirs/optimization' / candidate.route /
        candidate.id / str(candidate.seed) / stage / f'{stage}.json')
    _write_generic_artifact(output, stage)
    runtime_config = output.parent.parent / 'convert/resolved-runtime.py'
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text(
        'model = dict(type="Fixture", authority_tag="canonical")\n',
        encoding='utf-8')
    runtime = {
        'config_path': runtime_config,
        'config_sha256': _sha256(runtime_config),
        'checkpoint_name': candidate.checkpoint.as_posix(),
        'checkpoint_sha256': candidate.checkpoint_sha256,
        'pwl_stage_a': {'path': 'smoke.json', 'sha256': '7' * 64},
    }
    source = {
        'authority_sha256': '8' * 64,
        'config_sha256': _sha256(tmp_path / candidate.config),
        'git_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=tmp_path,
            text=True).strip(),
    }
    monkeypatch.setattr(
        controller_module, 'resolve_artifact_source',
        lambda *args, **kwargs: (source, candidate, object()))
    monkeypatch.setattr(
        numeric_runtime, 'resolve_numeric_runtime',
        lambda *args, **kwargs: runtime)
    monkeypatch.setattr(
        controller_module, 'validate_evaluation_envelope',
        lambda *args, **kwargs: {
            'modes': {'flip': {'protocol': {'kind': 'evaluate'}}}})
    monkeypatch.setattr(
        controller_module, 'validate_latency_envelope',
        lambda *args, **kwargs: {
            'protocol': {'data': {'kind': 'latency'}}, 'gpu_lease': {}})
    monkeypatch.setattr(
        OptimizationController, '_match_latency_lease',
        lambda *args, **kwargs: None)
    monkeypatch.setattr(
        controller_module, 'validate_live_coco_observation',
        lambda *args, **kwargs: None)

    def authorize(*args, **kwargs):
        value = authority(*args, **kwargs)
        value.path = runtime_config.resolve()
        value.sha256 = _sha256(runtime_config)
        return value

    monkeypatch.setattr(
        'mambapose_opt.checkpoints.authorize_pwl_runtime_config', authorize)
    controller = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate,
        lambda *args: None, repository_root=tmp_path,
        manifest_path=tmp_path / 'optimization/candidates.json',
        stages=(stage,))
    return controller, output, runtime_config


@pytest.mark.parametrize('stage', ['evaluate', 'latency'])
def test_pwl_controller_reconstructs_runtime_config_without_direct_parse(
        tmp_path, monkeypatch, stage):
    from types import SimpleNamespace

    from mmengine.config import Config

    calls = []
    expected_config = Config(dict(
        model=dict(type='Fixture', authority_tag='authorized')))

    def authorize(*args, conversion_path, **kwargs):
        calls.append(('authorize', Path(conversion_path)))
        return SimpleNamespace(
            load_config=lambda: (
                calls.append(('load', None)) or expected_config),
            verify=lambda: calls.append(('verify', None)))

    controller, output, _runtime_config = _stub_pwl_controller_validation(
        tmp_path, monkeypatch, stage=stage, authority=authorize)
    monkeypatch.setattr(
        Config, 'fromfile',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('PWL controller used direct Config.fromfile')))

    schema = controller._artifact_schema(
        stage, output,
        expected_gpu_lease=(object() if stage == 'latency' else None),
        lease_validated_at=(
            datetime.now(timezone.utc) if stage == 'latency' else None))

    assert schema == 'optimization-stage-envelope-v1'
    assert calls[0] == (
        'authorize', output.parent.parent / 'convert/convert.json')
    assert [kind for kind, _ in calls].count('load') == 1
    assert [kind for kind, _ in calls].count('verify') >= 1


@pytest.mark.parametrize('stage', ['evaluate', 'latency'])
def test_pwl_controller_rejects_runtime_config_swap_after_live_observation(
        tmp_path, monkeypatch, stage):
    from types import SimpleNamespace

    from mmengine.config import Config
    from mambapose_opt import controller as controller_module
    from mambapose_opt.controller import ArtifactValidationError

    verifies = []
    expected_config = Config(dict(model=dict(type='Fixture')))

    def verify():
        verifies.append(None)
        if len(verifies) > 1:
            raise ValueError('PWL runtime config authority changed')

    authority = lambda *args, **kwargs: SimpleNamespace(
        load_config=lambda: (verify() or expected_config), verify=verify)
    controller, output, runtime_config = _stub_pwl_controller_validation(
        tmp_path, monkeypatch, stage=stage, authority=authority)
    monkeypatch.setattr(
        controller_module, 'validate_live_coco_observation',
        lambda *args, **kwargs: runtime_config.write_text(
            'model = dict(type="Substituted")\n', encoding='utf-8'))

    with pytest.raises(
            ArtifactValidationError,
            match='PWL runtime config authority changed'):
        controller._artifact_schema(
            stage, output,
            expected_gpu_lease=(object() if stage == 'latency' else None),
            lease_validated_at=(
                datetime.now(timezone.utc) if stage == 'latency' else None))

    assert len(verifies) >= 2


@pytest.mark.parametrize('stage', ['evaluate', 'latency'])
def test_pwl_controller_rejects_alternate_stage_output_before_authorizing(
        tmp_path, monkeypatch, stage):
    from mambapose_opt.controller import ArtifactValidationError

    def must_not_authorize(*args, **kwargs):
        raise AssertionError('alternate stage output reached Config authorizer')

    controller, output, _runtime_config = _stub_pwl_controller_validation(
        tmp_path, monkeypatch, stage=stage, authority=must_not_authorize)
    alternate = output.parent.parent / 'alternate' / output.name
    alternate.parent.mkdir()
    alternate.write_bytes(output.read_bytes())

    with pytest.raises(ArtifactValidationError, match='path is not canonical'):
        controller._artifact_schema(
            stage, alternate,
            expected_gpu_lease=(object() if stage == 'latency' else None),
            lease_validated_at=(
                datetime.now(timezone.utc) if stage == 'latency' else None))


def _outcome(stage, artifact, *, exit_code=0, valid=True, fingerprint='ok'):
    from mambapose_opt.controller import StageOutcome

    hashes = {str(artifact): _sha256(artifact)} if artifact.exists() else {}
    return StageOutcome(
        stage_id=f'fixture:{stage}',
        stage=stage,
        candidate_id='fixture',
        exit_code=exit_code,
        fingerprint=fingerprint,
        artifacts_valid=valid,
        artifacts=(artifact,),
        artifact_sha256=hashes,
    )


def test_pwl_smoke_controller_owns_one_outer_gpu_lease_and_precreates_dir(
        tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from mambapose_opt import controller as controller_module
    from mambapose_opt.controller import OptimizationController, StageOutcome
    from mambapose_opt.gpu_guard import GpuLease
    from mambapose_opt.schema import CandidateSpec

    parent = _candidate(tmp_path)
    candidate = CandidateSpec.from_dict({
        'id': parent.id, 'route': 'ssm-quant-pwl', 'kind': 'pwl',
        'config': parent.config.as_posix(),
        'checkpoint': parent.checkpoint.as_posix(),
        'checkpoint_sha256': parent.checkpoint_sha256,
        'seed': parent.seed, 'features': {'numeric_kind': 'pwl'},
    })
    calls = []
    lease = GpuLease(
        stage_id='fixture:smoke-stage-a', pid=os.getpid(),
        boot_id='11111111-1111-1111-1111-111111111111',
        timestamp='2026-08-28T00:00:00+00:00', device_index=0,
        allowed_pids=(os.getpid(),), lease_id='7' * 64)

    @contextmanager
    def owned(*args, **kwargs):
        calls.append((args, kwargs))
        yield lease

    def runner(_candidate, stage, stage_dir, attempt):
        assert stage_dir.is_dir()
        artifact = stage_dir / 'smoke.json'
        artifact.write_text('{}', encoding='utf-8')
        return StageOutcome(
            'fixture:smoke-stage-a', stage, 'fixture', 0, 'ok',
            artifacts_valid=True, artifacts=(artifact,),
            artifact_sha256={str(artifact): _sha256(artifact)},
            attempt=attempt)

    monkeypatch.setattr(controller_module, 'authorize_manifest_candidate',
                        lambda *args, **kwargs: SimpleNamespace(
                            candidate=candidate))
    monkeypatch.setattr(controller_module, 'exclusive_cuda_stage', owned)
    monkeypatch.setattr(
        OptimizationController, '_artifact_schema',
        lambda self, stage, path, **kwargs:
        'pwl-stage-a-full-model-smoke-v2')
    controller = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, runner,
        repository_root=tmp_path,
        manifest_path=tmp_path / 'optimization/candidates.json',
        stages=('smoke-stage-a',), now=lambda: datetime(
            2026, 8, 28, tzinfo=timezone.utc))

    outcome = controller.run_next()

    assert outcome.exit_code == 0
    assert outcome.gpu_lease == lease
    assert len(calls) == 1


def _write_generic_artifact(
        path, stage, candidate_id='fixture', *, device_index=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {}
    repo_root = next(
        (parent for parent in path.parents if (parent / '.git').exists()), None)
    source = None
    if repo_root is not None:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo_root, text=True).strip()
        source = {
            'git_commit': commit,
            'manifest_path': 'optimization/candidates.json',
            'manifest_sha256': _sha256(
                repo_root / 'optimization/candidates.json'),
            'config_path': 'configs/candidate.py',
            'config_sha256': _sha256(repo_root / 'configs/candidate.py'),
            'authority_path': 'optimization/coco_val2017_authority.json',
            'authority_sha256': _sha256(
                repo_root / 'optimization/coco_val2017_authority.json'),
        }
    if stage in {'evaluate', 'latency'}:
        provenance = {
            'checkpoint_sha256': hashlib.sha256(
                b'parent checkpoint').hexdigest(),
            'config_sha256': (
                'b' * 64 if stage == 'evaluate' else source['config_sha256']),
            'data_inventory_sha256': _sha256(
                repo_root / 'data/inventory.json'),
            'git_commit': source['git_commit'],
        }
        protocol = {
            'dataset': 'coco', 'split': 'val2017',
            'complete_split': True, 'batch_size': 1,
            'authority_path': 'optimization/coco_val2017_authority.json',
            'authority_sha256': source['authority_sha256'],
            'authority_image_count': 5000,
            'authority_annotation_count': 11004,
            'authority_detection_count': 104125,
            'inventory_authority_sha256': _sha256(
                repo_root / 'data/inventory.json'),
            'annotation_authority_sha256': '1' * 64,
            'detection_authority_sha256': '2' * 64,
            'image_corpus_digest_algorithm': (
                'sha256-filename-size-content-v1'),
            'image_corpus_authority_sha256': '6' * 64,
            'image_corpus_sha256': '6' * 64,
            'inventory_annotation_archive_sha256': '3' * 64,
            'inventory_image_archive_sha256': '4' * 64,
            'annotation_sha256': '1' * 64,
            'detection_sha256': '2' * 64,
            'inventory_detection_sha256': '2' * 64,
            'annotation_image_count': 5000,
            'annotation_record_count': 11004,
            'detection_record_count': 104125,
            'verified_image_count': 5000,
            'source_config': 'configs/candidate.py',
            'checkpoint': 'work_dirs/reproduction/parent.pth',
            'data_inventory': 'data/inventory.json',
            'inventory_projection': {
                'inventory_path': 'data/inventory.json',
                'inventory_sha256': _sha256(
                    repo_root / 'data/inventory.json'),
                'annotation_asset_id': 'coco-annotations',
                'annotation_declared_sha256': '3' * 64,
                'annotation_observed_archive_sha256': '3' * 64,
                'image_asset_id': 'coco-val2017',
                'image_declared_sha256': '4' * 64,
                'image_observed_archive_sha256': '4' * 64,
                'detection_asset_id': 'coco-val-detections',
                'detection_declared_sha256': '2' * 64,
                'detection_observed_sha256': '2' * 64,
            },
        }
        determinism = {
            'python_seed': 0, 'numpy_seed': 0, 'torch_seed': 0,
            'worker_count': 1,
            'workers': [{
                'worker_id': 0,
                'torch_seed_source': 'torch.initial_seed()',
                'python_seed_derivation': 'torch_seed % 2**32',
                'numpy_seed_derivation': 'torch_seed % 2**32',
            }],
            'persistent_workers': False,
            'order_hashes': {'0': 'e' * 64},
            'provenance': provenance,
        }
        row = {
            'metrics': {
                'unit': 'percentage_points', 'AP': 72.8, 'AP50': 89.7,
                'AP75': 80.5, 'APM': 69.4, 'APL': 79.2, 'AR': 78.2,
            },
            'provenance': provenance,
            'determinism': determinism,
            'protocol': protocol,
        }
        if stage == 'evaluate':
            result = {
                'route': 'accuracy-first', 'calibration_split': None,
                'modes': {'flip': row, 'no_flip': row}, 'source': source,
            }
        else:
            summary = {
                'median_ms': 1.0, 'p90_ms': 1.2, 'p95_ms': 1.3,
                'sample_count': 200,
            }
            data = dict(protocol)
            for key in ('batch_size', 'source_config', 'checkpoint',
                        'data_inventory'):
                data.pop(key)
            result = {
                'route': 'accuracy-first', 'provenance': provenance,
                'source': source,
                'protocol': {
                    'batch_size': 1, 'warmup': 50, 'iterations': 200,
                    'timer': 'torch.cuda.Event', 'synchronize': True,
                    'scope': 'full_topdown_model',
                    'lease_max_age_seconds': 300,
                    'lease_max_future_skew_seconds': 30,
                    'source_config': 'configs/candidate.py',
                    'checkpoint': 'work_dirs/reproduction/parent.pth',
                    'data_inventory': 'data/inventory.json', 'data': data,
                },
                'modes': {'flip': summary, 'no_flip': summary},
                'gpu_lease': (
                    {**asdict(_ACTIVE_TEST_LEASE),
                     'allowed_pids': list(_ACTIVE_TEST_LEASE.allowed_pids)}
                    if _ACTIVE_TEST_LEASE is not None else {
                        'stage_id': 'fixture:latency', 'pid': os.getpid(),
                        'boot_id': '11111111-1111-1111-1111-111111111111',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'device_index': device_index,
                        'allowed_pids': [os.getpid()],
                        'lease_id': '7' * 64,
                    }),
            }
    path.write_text(json.dumps({
        'schema_version': 1,
        'candidate_id': candidate_id,
        'stage': stage,
        'result': result,
    }))


def _profile_artifact_value(candidate):
    return {
        'schema_version': 1,
        'git_commit': 'a' * 40,
        'candidate': candidate.id,
        'config': candidate.config.as_posix(),
        'checkpoint': candidate.checkpoint.as_posix(),
        'checkpoint_sha256': candidate.checkpoint_sha256,
        'input_shapes': [1, 3, 256, 192],
        'output_shapes': [1, 17, 64, 48],
        'parameters': {
            'total': 1, 'trainable': 1,
            'bytes_by_dtype': {'torch.float32': 4},
            'by_prefix': {'model': 1},
        },
        'modules': [{
            'name': '',
            'kind': 'FixtureModel',
            'parameters': 1,
            'hazard': None,
        }],
    }


def _write_profile_artifact(path, candidate):
    path.write_text(json.dumps(_profile_artifact_value(candidate)))


def test_controller_refuses_single_mode_evaluation_as_complete(tmp_path):
    from mambapose_opt.controller import (
        ArtifactValidationError, OptimizationController)

    candidate = _candidate(tmp_path)
    campaign = tmp_path / 'work_dirs/optimization'
    controller = OptimizationController(
        campaign, candidate, lambda *args: None,
        repository_root=tmp_path)
    path = campaign / 'candidates/fixture/evaluate/evaluate.json'
    _write_generic_artifact(path, 'evaluate')
    payload = json.loads(path.read_text())
    del payload['result']['modes']['no_flip']
    path.write_text(json.dumps(payload))

    with pytest.raises(ArtifactValidationError, match='both.*modes'):
        controller._artifact_schema('evaluate', path)


def test_controller_refuses_empty_dual_mode_rows_as_complete(tmp_path):
    from mambapose_opt.controller import (
        ArtifactValidationError, OptimizationController)

    candidate = _candidate(tmp_path)
    campaign = tmp_path / 'work_dirs/optimization'
    controller = OptimizationController(
        campaign, candidate, lambda *args: None,
        repository_root=tmp_path)
    path = campaign / 'candidates/fixture/evaluate/evaluate.json'
    _write_generic_artifact(path, 'evaluate')
    payload = json.loads(path.read_text())
    payload['result']['modes'] = {
        'flip': {'metrics': {}, 'provenance': {},
                 'determinism': {}, 'protocol': {}},
        'no_flip': {'metrics': {}, 'provenance': {},
                    'determinism': {}, 'protocol': {}},
    }
    path.write_text(json.dumps(payload))

    with pytest.raises(ArtifactValidationError, match='metrics|provenance'):
        controller._artifact_schema('evaluate', path)


def test_controller_refuses_empty_latency_result_as_complete(tmp_path):
    from mambapose_opt.controller import (
        ArtifactValidationError, OptimizationController)

    candidate = _candidate(tmp_path)
    campaign = tmp_path / 'work_dirs/optimization'
    controller = OptimizationController(
        campaign, candidate, lambda *args: None,
        repository_root=tmp_path)
    path = campaign / 'candidates/fixture/latency/latency.json'
    _write_generic_artifact(path, 'latency')
    payload = json.loads(path.read_text())
    payload['result'] = {}
    path.write_text(json.dumps(payload))

    with pytest.raises(ArtifactValidationError, match='latency'):
        controller._artifact_schema('latency', path)


def test_controller_binds_evaluation_to_repository_authority_hash(tmp_path):
    from mambapose_opt.controller import (
        ArtifactValidationError, OptimizationController)

    candidate = _candidate(tmp_path)
    campaign = tmp_path / 'work_dirs/optimization'
    controller = OptimizationController(
        campaign, candidate, lambda *args: None,
        repository_root=tmp_path)
    path = campaign / 'candidates/fixture/evaluate/evaluate.json'
    _write_generic_artifact(path, 'evaluate')
    payload = json.loads(path.read_text())
    for row in payload['result']['modes'].values():
        row['protocol']['authority_sha256'] = '0' * 64
    path.write_text(json.dumps(payload))

    with pytest.raises(ArtifactValidationError, match='authority'):
        controller._artifact_schema('evaluate', path)


def test_controller_reobserves_live_assets_before_accepting_evaluation(
        tmp_path, monkeypatch):
    import mambapose_opt.controller as controller_module
    from mambapose_opt.controller import (
        ArtifactValidationError, OptimizationController)
    from mambapose_opt.evaluation import MetricError

    candidate = _candidate(tmp_path)
    campaign = tmp_path / 'work_dirs/optimization'
    controller = OptimizationController(
        campaign, candidate, lambda *args: None,
        repository_root=tmp_path)
    path = campaign / 'candidates/fixture/evaluate/evaluate.json'
    _write_generic_artifact(path, 'evaluate')
    monkeypatch.setattr(
        controller_module, 'validate_live_coco_observation',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            MetricError('live corpus changed')))

    with pytest.raises(ArtifactValidationError, match='live corpus changed'):
        controller._artifact_schema('evaluate', path)


@pytest.mark.parametrize('stage, tool', [
    ('evaluate', 'evaluate_candidate.py'),
    ('latency', 'measure_latency.py'),
])
def test_stage_runner_forwards_authoritative_manifest_exactly(
        tmp_path, monkeypatch, stage, tool):
    import tools.optimization.run_campaign as campaign_tool

    candidate = _candidate(tmp_path)
    manifest = tmp_path / 'alternate/candidates.json'
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({
        'schema_version': 1,
        'candidates': [{
            'id': 'fixture', 'route': 'accuracy-first', 'kind': 'float',
            'config': 'configs/alternate.py',
            'checkpoint': 'checkpoints/alternate.pth',
            'checkpoint_sha256': 'f' * 64, 'seed': 3, 'features': {},
        }],
    }))
    artifact = tmp_path / f'work_dirs/optimization/{stage}/{stage}.json'
    monkeypatch.setattr(campaign_tool, 'REPO_ROOT', tmp_path)
    runner = campaign_tool.SubprocessStageRunner(
        tmp_path / 'work_dirs/optimization', manifest, device_index=3)

    command = runner._command(candidate, stage, artifact)

    assert command == [
        str(tmp_path / '.venv/bin/python'),
        str(tmp_path / f'tools/optimization/{tool}'),
        'fixture', '--manifest', str(manifest),
        '--output', f'work_dirs/optimization/{stage}/{stage}.json',
    ]
    selected = (
        __import__(
            f'tools.optimization.{tool.removesuffix(".py")}',
            fromlist=['_candidate'])
        ._candidate(manifest, 'fixture'))
    assert selected.config.as_posix() == 'configs/alternate.py'
    assert selected.checkpoint.as_posix() == 'checkpoints/alternate.pth'


def _mock_lease(monkeypatch, entered):
    from mambapose_opt import controller
    from mambapose_opt.gpu_guard import GpuLease

    @contextmanager
    def lease(lock_path, device_index, allowed_pids, *, stage_id):
        global _ACTIVE_TEST_LEASE
        entered.append((Path(lock_path), device_index, stage_id, set(allowed_pids)))
        acquired = GpuLease(
            stage_id, os.getpid(),
            '11111111-1111-1111-1111-111111111111',
            datetime.now(timezone.utc).isoformat(), device_index,
            tuple(sorted(allowed_pids)), '7' * 64)
        _ACTIVE_TEST_LEASE = acquired
        try:
            yield acquired
        finally:
            _ACTIVE_TEST_LEASE = None

    monkeypatch.setattr(controller, 'exclusive_cuda_stage', lease)
    monkeypatch.setattr(
        controller, 'validate_live_coco_observation',
        lambda recorded, **unused: recorded)


def test_stage_completes_only_after_artifacts_validate(tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController
    from mambapose_repro.state import StateStore

    candidate = _candidate(tmp_path)
    entered = []
    _mock_lease(monkeypatch, entered)

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / 'metrics.json'
        _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact, valid=False)

    campaign = tmp_path / 'work_dirs/optimization'
    outcome = OptimizationController(
        campaign, candidate, runner, repository_root=tmp_path,
        stages=('evaluate',)).run_next()

    assert outcome.exit_code == 78
    run = StateStore(campaign).read()['runs']['fixture:evaluate']
    assert run['status'] == 'blocked'
    assert entered[0][2] == 'fixture:evaluate'


def test_contention_returns_75_and_retry_lineage_is_append_only(
        tmp_path, monkeypatch):
    from mambapose_opt import controller
    from mambapose_opt.controller import OptimizationController
    from mambapose_opt.gpu_guard import ExternalGpuContention
    from mambapose_repro.state import StateStore

    candidate = _candidate(tmp_path)

    @contextmanager
    def contended(*args, **kwargs):
        raise ExternalGpuContention('external pid=9001')
        yield

    monkeypatch.setattr(controller, 'exclusive_cuda_stage', contended)
    campaign = tmp_path / 'work_dirs/optimization'
    clock = [datetime(2026, 8, 27, tzinfo=timezone.utc)]
    instance = OptimizationController(
        campaign, candidate, lambda *args: None,
        repository_root=tmp_path, stages=('train',), now=lambda: clock[0])

    first = instance.run_next()
    clock[0] += timedelta(seconds=30)
    second = instance.run_next()

    assert first.exit_code == second.exit_code == 75
    run = StateStore(campaign).read()['runs']['fixture:train']
    assert run['status'] == 'retry_wait'
    assert [
        (item['attempt'], item['status'])
        for item in run['retry_lineage']
    ] == [
        (1, 'started'), (1, 'retry_wait'),
        (2, 'started'), (2, 'retry_wait'),
    ]
    terminal = [
        item for item in run['retry_lineage']
        if item['status'] == 'retry_wait'
    ]
    assert all(item['exit_code'] == 75 for item in terminal)
    assert [item['retry_delay_seconds'] for item in terminal] == [30, 120]
    assert run['retry_delay_seconds'] == 120
    events = [
        json.loads(line)
        for line in (campaign / 'events.jsonl').read_text().splitlines()
    ]
    waits = [event for event in events if event['status'] == 'retry_wait']
    assert [event['attempt'] for event in waits] == [1, 2]


def test_retry_deadline_survives_restart_without_consuming_an_attempt(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController, StageOutcome
    from mambapose_repro.state import StateStore

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])
    campaign = tmp_path / 'work_dirs/optimization'
    clock = [datetime(2026, 8, 27, 1, 2, 3, tzinfo=timezone.utc)]
    calls = []

    def runner(candidate, stage, stage_dir, attempt):
        calls.append(attempt)
        if attempt == 1:
            return StageOutcome(
                'fixture:evaluate', 'evaluate', 'fixture', 75,
                'transient-first')
        artifact = stage_dir / 'evaluate.json'
        _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact)

    first = OptimizationController(
        campaign, candidate, runner, repository_root=tmp_path,
        stages=('evaluate',), retry_delays=(30,), now=lambda: clock[0]
    ).run_next()
    stored_after_first = StateStore(campaign).read()
    deadline = '2026-08-27T01:02:33+00:00'

    assert first.exit_code == 75
    assert stored_after_first['runs']['fixture:evaluate'][
        'retry_not_before'] == deadline

    clock[0] += timedelta(seconds=29)
    waiting = OptimizationController(
        campaign, candidate, runner, repository_root=tmp_path,
        stages=('evaluate',), retry_delays=(30,), now=lambda: clock[0]
    ).run_next()

    assert waiting.exit_code == 75
    assert waiting.retry_not_before == deadline
    assert waiting.retry_remaining_seconds == pytest.approx(1.0)
    assert calls == [1]
    assert StateStore(campaign).read() == stored_after_first

    clock[0] += timedelta(seconds=1)
    resumed = OptimizationController(
        campaign, candidate, runner, repository_root=tmp_path,
        stages=('evaluate',), retry_delays=(30,), now=lambda: clock[0]
    ).run_next()

    assert resumed.exit_code == 0
    assert calls == [1, 2]
    lineage = StateStore(campaign).read()[
        'runs']['fixture:evaluate']['retry_lineage']
    assert [(item['attempt'], item['status']) for item in lineage] == [
        (1, 'started'), (1, 'retry_wait'),
        (2, 'started'), (2, 'complete'),
    ]


def test_invalid_config_or_checkpoint_hash_returns_78_without_runner(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController
    from mambapose_repro.orchestrator import failure_fingerprint

    candidate = _candidate(tmp_path)
    (tmp_path / candidate.checkpoint).write_bytes(b'tampered')
    entered = []
    _mock_lease(monkeypatch, entered)
    calls = []

    result = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate,
        lambda *args: calls.append(args), repository_root=tmp_path,
        stages=('profile',)).run_next()

    assert result.exit_code == 78
    assert 'checkpoint' in result.message
    assert result.fingerprint == failure_fingerprint(result.message)
    assert calls == []
    assert entered == []


def test_all_cuda_stages_share_one_lease_and_compare_is_cpu_only(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    entered = []
    _mock_lease(monkeypatch, entered)

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / f'{stage}.json'
        if stage == 'profile':
            _write_profile_artifact(artifact, candidate)
        else:
            _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact)

    instance = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, runner,
        repository_root=tmp_path, stages=('profile', 'compare'))

    assert instance.run_next().exit_code == 0
    assert instance.run_next().exit_code == 0
    assert [item[2] for item in entered] == ['fixture:profile']
    assert entered[0][0].name == 'gpu.lock'


def test_checkpoint_retention_keeps_best_and_two_valid_resume_points(tmp_path):
    from mambapose_opt.controller import retain_checkpoints

    paths = []
    for name in ('best_AP_epoch_4.pth', 'epoch_1.pth', 'epoch_2.pth',
                 'epoch_3.pth', 'epoch_4.pth', 'deployment.pth'):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths.append(path)

    kept = retain_checkpoints(
        tmp_path,
        validator=lambda path: path.name not in {'epoch_3.pth'},
    )

    assert {path.name for path in kept} == {
        'best_AP_epoch_4.pth', 'epoch_4.pth', 'epoch_2.pth'}
    assert {path.name for path in tmp_path.glob('*.pth')} == {
        'best_AP_epoch_4.pth', 'epoch_4.pth', 'epoch_2.pth',
        'deployment.pth'}


def test_retention_requires_sufficient_valid_set_before_deleting(tmp_path):
    from mambapose_opt.controller import (
        CheckpointRetentionError,
        retain_checkpoints,
    )

    for name in ('best_AP_epoch_2.pth', 'epoch_1.pth', 'broken.pth'):
        (tmp_path / name).write_bytes(name.encode())
    before = {path.name: path.read_bytes() for path in tmp_path.glob('*.pth')}

    with pytest.raises(CheckpointRetentionError, match='two valid resume'):
        retain_checkpoints(tmp_path, validator=lambda path: True)

    assert {
        path.name: path.read_bytes() for path in tmp_path.glob('*.pth')
    } == before


def test_retention_validator_exception_does_not_delete_anything(tmp_path):
    from mambapose_opt.controller import CheckpointRetentionError, retain_checkpoints

    for name in ('best_AP_epoch_3.pth', 'epoch_1.pth', 'epoch_2.pth'):
        (tmp_path / name).write_bytes(name.encode())

    with pytest.raises(CheckpointRetentionError, match='validation failed'):
        retain_checkpoints(
            tmp_path,
            validator=lambda path: (_ for _ in ()).throw(ValueError('bad')),
        )

    assert len(tuple(tmp_path.glob('*.pth'))) == 3


def test_default_checkpoint_contract_accepts_real_best_and_resume_shapes(
        tmp_path):
    import torch

    from mambapose_opt.controller import OptimizationController

    best = tmp_path / 'best_AP_epoch_3.pth'
    resume_a = tmp_path / 'epoch_2.pth'
    resume_b = tmp_path / 'epoch_3.pth'
    torch.save({
        'state_dict': {'weight': torch.ones(1)},
        'meta': {'epoch': 3},
    }, best)
    for epoch, path in ((2, resume_a), (3, resume_b)):
        torch.save({
            'state_dict': {'weight': torch.ones(1)},
            'meta': {'epoch': epoch},
            'optimizer': {'state': {}},
            'param_schedulers': [{}],
        }, path)

    assert OptimizationController._checkpoint_is_valid(best) is True
    assert OptimizationController._checkpoint_is_valid(resume_a) is True
    assert OptimizationController._checkpoint_is_valid(resume_b) is True


def test_completed_train_stage_applies_checkpoint_retention(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    entered = []
    _mock_lease(monkeypatch, entered)

    def runner(candidate, stage, stage_dir, attempt):
        for name in (
                'best_AP_epoch_4.pth', 'epoch_1.pth', 'epoch_2.pth',
                'epoch_3.pth', 'epoch_4.pth'):
            (stage_dir / name).write_bytes(name.encode())
        artifact = stage_dir / 'train.json'
        _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact)

    instance = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, runner,
        repository_root=tmp_path, stages=('train',),
        checkpoint_validator=lambda path: path.name != 'epoch_3.pth')

    result = instance.run_next()

    checkpoint_dir = (
        tmp_path / 'work_dirs/optimization/accuracy-first/fixture/0/train')
    assert result.exit_code == 0
    assert {path.name for path in checkpoint_dir.glob('*.pth')} == {
        'best_AP_epoch_4.pth', 'epoch_4.pth', 'epoch_2.pth'}


def test_train_with_insufficient_checkpoints_is_blocked_without_deletion(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController
    from mambapose_repro.state import StateStore

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])

    def runner(candidate, stage, stage_dir, attempt):
        (stage_dir / 'best_AP_epoch_1.pth').write_bytes(b'best')
        (stage_dir / 'epoch_1.pth').write_bytes(b'resume')
        artifact = stage_dir / 'train.json'
        _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact)

    campaign = tmp_path / 'work_dirs/optimization'
    result = OptimizationController(
        campaign, candidate, runner, repository_root=tmp_path,
        stages=('train',), checkpoint_validator=lambda path: True).run_next()
    stage_dir = campaign / 'accuracy-first/fixture/0/train'

    assert result.exit_code == 78
    assert StateStore(campaign).read()['runs']['fixture:train']['status'] == 'blocked'
    assert {path.name for path in stage_dir.glob('*.pth')} == {
        'best_AP_epoch_1.pth', 'epoch_1.pth'}


def test_not_json_artifact_is_blocked_with_exit_78(tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / 'evaluate.json'
        artifact.write_text('not-json')
        return _outcome(stage, artifact)

    result = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, runner,
        repository_root=tmp_path, stages=('evaluate',)).run_next()

    assert result.exit_code == 78
    assert result.artifacts_valid is False


def test_generic_artifact_identity_and_stage_are_mandatory(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / 'evaluate.json'
        _write_generic_artifact(
            artifact, 'latency', candidate_id='different-candidate')
        return _outcome(stage, artifact)

    result = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, runner,
        repository_root=tmp_path, stages=('evaluate',)).run_next()

    assert result.exit_code == 78


def test_existing_profile_schema_is_accepted(tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / 'profile.json'
        _write_profile_artifact(artifact, candidate)
        return _outcome(stage, artifact)

    result = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, runner,
        repository_root=tmp_path, stages=('profile',)).run_next()

    assert result.exit_code == 0


def test_formal_profile_v2_binds_runtime_and_controller_cuda_device(tmp_path):
    from mambapose_opt.controller import (
        ArtifactValidationError, OptimizationController)

    candidate = _candidate(tmp_path)
    campaign = tmp_path / 'work_dirs/optimization'
    controller = OptimizationController(
        campaign, candidate, lambda *_args: None,
        repository_root=tmp_path, device_index=3)
    path = campaign / 'accuracy-first/fixture/0/profile/profile.json'
    path.parent.mkdir(parents=True)
    value = _profile_artifact_value(candidate)
    value.update({
        'schema_version': 2,
        'device': {
            'logical': 'cuda:0', 'physical_index': 3, 'kind': 'cuda'},
        'parent': {
            'config': candidate.config.as_posix(),
            'checkpoint': candidate.checkpoint.as_posix(),
            'checkpoint_sha256': candidate.checkpoint_sha256,
        },
        'runtime': {
            'config': {
                'path': candidate.config.as_posix(),
                'sha256': _sha256(tmp_path / candidate.config),
            },
            'checkpoint': {
                'path': candidate.checkpoint.as_posix(),
                'sha256': candidate.checkpoint_sha256,
            },
        },
    })
    path.write_text(json.dumps(value))

    assert controller._artifact_schema('profile', path) == (
        'optimization-profile-v2')

    value['device']['physical_index'] = 2
    path.write_text(json.dumps(value))
    with pytest.raises(ArtifactValidationError, match='device'):
        controller._artifact_schema('profile', path)


@pytest.mark.parametrize(
    'mutate',
    [
        lambda value: value.update({'unexpected': True}),
        lambda value: value.pop('output_shapes'),
        lambda value: value.update({'git_commit': 'not-a-full-commit'}),
        lambda value: value.update({'input_shapes': [1, 3, 0, 192]}),
        lambda value: value.update({'input_shapes': []}),
        lambda value: value.update({'output_shapes': None}),
        lambda value: value.update({'output_shapes': []}),
        lambda value: value.update({'parameters': {}}),
        lambda value: value['parameters'].update({'total': -1}),
        lambda value: value['parameters'].update({
            'total': 1, 'trainable': 2}),
        lambda value: value['parameters'].update({'by_prefix': {}}),
        lambda value: value.update({'modules': []}),
        lambda value: value.update({'modules': [{
            'name': '', 'kind': 'FixtureModel', 'parameters': -1,
            'hazard': None,
        }]}),
        lambda value: value.update({'modules': [{
            'name': '', 'kind': 'FixtureModel', 'parameters': 1,
            'hazard': 7,
        }]}),
    ],
    ids=[
        'extra-field', 'missing-field', 'invalid-commit', 'nonpositive-input',
        'empty-input', 'null-output', 'empty-output', 'empty-parameters',
        'negative-total', 'trainable-exceeds-total', 'empty-parameter-map',
        'empty-modules', 'negative-module-parameters', 'untyped-hazard',
    ],
)
def test_profile_v1_rejects_reviewer_malformed_structures(
        tmp_path, monkeypatch, mutate):
    from mambapose_opt.controller import OptimizationController
    from mambapose_repro.state import StateStore

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / 'profile.json'
        value = _profile_artifact_value(candidate)
        mutate(value)
        artifact.write_text(json.dumps(value))
        return _outcome(stage, artifact)

    campaign = tmp_path / 'work_dirs/optimization'
    result = OptimizationController(
        campaign, candidate, runner, repository_root=tmp_path,
        stages=('profile',)).run_next()

    assert result.exit_code == 78
    assert StateStore(campaign).read()[
        'runs']['fixture:profile']['status'] == 'blocked'


def test_complete_stage_evidence_is_persisted_and_revalidated(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController
    from mambapose_repro.state import StateStore

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])
    calls = []

    def runner(candidate, stage, stage_dir, attempt):
        calls.append(attempt)
        artifact = stage_dir / 'evaluate.json'
        _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact)

    campaign = tmp_path / 'work_dirs/optimization'
    first = OptimizationController(
        campaign, candidate, runner, repository_root=tmp_path,
        stages=('evaluate',)).run_next()
    run = StateStore(campaign).read()['runs']['fixture:evaluate']

    assert first.exit_code == 0
    assert run['artifact_evidence'] == [{
        'path': 'accuracy-first/fixture/0/evaluate/evaluate.json',
        'sha256': _sha256(
            campaign / 'accuracy-first/fixture/0/evaluate/evaluate.json'),
        'schema': 'optimization-stage-envelope-v1',
    }]

    artifact = campaign / run['artifact_evidence'][0]['path']
    artifact.write_text('tampered')
    second = OptimizationController(
        campaign, candidate, runner, repository_root=tmp_path,
        stages=('evaluate',)).run_next()

    assert second.stage == 'evaluate'
    assert calls == [1, 2]


def test_isolated_campaign_roots_collide_on_canonical_gpu_lock(
        tmp_path, monkeypatch):
    from mambapose_opt import gpu_guard
    from mambapose_opt.controller import OptimizationController

    route_a = tmp_path / 'route-a'
    route_b = tmp_path / 'route-b'
    route_a.mkdir()
    route_b.mkdir()
    candidate = _candidate(route_a)
    second_candidate = _candidate(route_b)
    monkeypatch.setattr(gpu_guard, 'query_compute_processes', lambda _: ())
    import mambapose_opt.controller as controller_module
    monkeypatch.setattr(
        controller_module, 'validate_live_coco_observation',
        lambda recorded, **unused: recorded)
    canonical_lock = tmp_path / 'shared/work_dirs/optimization/gpu.lock'
    nested_results = []

    def second_runner(*args):
        raise AssertionError('contended runner must not start')

    second = OptimizationController(
        route_b / 'work_dirs/optimization', second_candidate,
        second_runner, repository_root=route_b, stages=('evaluate',),
        gpu_lock_path=canonical_lock,
        shared_lock_root=tmp_path / 'shared')

    def first_runner(candidate, stage, stage_dir, attempt):
        nested_results.append(second.run_next())
        artifact = stage_dir / 'evaluate.json'
        _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact)

    first = OptimizationController(
        route_a / 'work_dirs/optimization', candidate,
        first_runner, repository_root=route_a, stages=('evaluate',),
        gpu_lock_path=canonical_lock,
        shared_lock_root=tmp_path / 'shared')

    assert first.run_next().exit_code == 0
    assert nested_results[0].exit_code == 75
    assert canonical_lock.is_file()


def test_device_one_is_used_for_admission_and_runner_environment(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController
    from tools.optimization.run_campaign import SubprocessStageRunner

    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '0')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    monkeypatch.setenv('MAMBAPOSE_TEST_SENTINEL', 'preserved')
    candidate = _candidate(tmp_path)
    entered = []
    _mock_lease(monkeypatch, entered)

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / 'latency.json'
        _write_generic_artifact(artifact, stage, device_index=1)
        return _outcome(stage, artifact)

    result = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, runner,
        repository_root=tmp_path, stages=('latency',), device_index=1,
        gpu_lock_path=(
            tmp_path / 'shared/work_dirs/optimization/gpu.lock'),
        shared_lock_root=tmp_path / 'shared').run_next()

    subprocess_runner = SubprocessStageRunner(
        tmp_path / 'work_dirs/optimization',
        tmp_path / 'optimization/candidates.json',
        device_index=1,
    )
    assert result.exit_code == 0
    assert entered[0][1] == 1
    environment = subprocess_runner.environment()
    assert environment['CUDA_VISIBLE_DEVICES'] == '1'
    assert environment['PYTHONDONTWRITEBYTECODE'] == '1'
    assert environment['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert environment['MAMBAPOSE_TEST_SENTINEL'] == 'preserved'


def test_controller_rejects_latency_nonce_not_from_its_acquisition(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / 'latency.json'
        _write_generic_artifact(artifact, stage)
        value = json.loads(artifact.read_text())
        value['result']['gpu_lease']['lease_id'] = '8' * 64
        artifact.write_text(json.dumps(value))
        return _outcome(stage, artifact)

    result = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, runner,
        repository_root=tmp_path, stages=('latency',),
        gpu_lock_path=tmp_path / 'shared/work_dirs/optimization/gpu.lock',
        shared_lock_root=tmp_path / 'shared').run_next()

    assert result.exit_code == 78
    assert 'lease' in result.message and 'lease_id' in result.message


def test_completed_latency_revalidates_persisted_controller_lease(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / 'latency.json'
        _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact)

    controller = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, runner,
        repository_root=tmp_path, stages=('latency',),
        gpu_lock_path=tmp_path / 'shared/work_dirs/optimization/gpu.lock',
        shared_lock_root=tmp_path / 'shared')
    assert controller.run_next().exit_code == 0
    run = controller.store.read()['runs']['fixture:latency']
    assert datetime.fromisoformat(run['lease_validated_at']).tzinfo is not None
    assert controller._completed_evidence_valid('latency', run)

    counterfeit = json.loads(json.dumps(run))
    counterfeit['gpu_lease']['lease_id'] = '8' * 64
    assert not controller._completed_evidence_valid('latency', counterfeit)


def test_old_completed_latency_remains_valid_against_persisted_validation_time(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / 'latency.json'
        _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact)

    campaign = tmp_path / 'work_dirs/optimization'
    controller = OptimizationController(
        campaign, candidate, runner, repository_root=tmp_path,
        stages=('latency',),
        gpu_lock_path=tmp_path / 'shared/work_dirs/optimization/gpu.lock',
        shared_lock_root=tmp_path / 'shared')
    assert controller.run_next().exit_code == 0
    run = controller.store.read()['runs']['fixture:latency']
    old_restart = OptimizationController(
        campaign, candidate, runner, repository_root=tmp_path,
        stages=('latency',),
        gpu_lock_path=tmp_path / 'shared/work_dirs/optimization/gpu.lock',
        shared_lock_root=tmp_path / 'shared',
        now=lambda: datetime(2036, 8, 27, tzinfo=timezone.utc))

    assert old_restart._completed_evidence_valid('latency', run)


def test_restart_rejects_future_latency_timestamp_even_with_updated_hash(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / 'latency.json'
        _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact)

    campaign = tmp_path / 'work_dirs/optimization'
    controller = OptimizationController(
        campaign, candidate, runner, repository_root=tmp_path,
        stages=('latency',),
        gpu_lock_path=tmp_path / 'shared/work_dirs/optimization/gpu.lock',
        shared_lock_root=tmp_path / 'shared')
    assert controller.run_next().exit_code == 0
    run = json.loads(json.dumps(
        controller.store.read()['runs']['fixture:latency']))
    artifact = campaign / run['artifact_evidence'][0]['path']
    value = json.loads(artifact.read_text())
    value['result']['gpu_lease']['timestamp'] = (
        '2999-01-01T00:00:00+00:00')
    artifact.write_text(json.dumps(value))
    run['artifact_evidence'][0]['sha256'] = _sha256(artifact)

    assert not controller._completed_evidence_valid('latency', run)


@pytest.mark.parametrize('validated_at', [
    'not-a-time',
    '2000-01-01T00:00:00',
    '1999-01-01T00:00:00+00:00',
])
def test_restart_rejects_malformed_or_replayed_lease_validation_time(
        tmp_path, monkeypatch, validated_at):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / 'latency.json'
        _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact)

    campaign = tmp_path / 'work_dirs/optimization'
    controller = OptimizationController(
        campaign, candidate, runner, repository_root=tmp_path,
        stages=('latency',),
        gpu_lock_path=tmp_path / 'shared/work_dirs/optimization/gpu.lock',
        shared_lock_root=tmp_path / 'shared')
    assert controller.run_next().exit_code == 0
    run = json.loads(json.dumps(
        controller.store.read()['runs']['fixture:latency']))
    run['lease_validated_at'] = validated_at

    assert not controller._completed_evidence_valid('latency', run)


@pytest.mark.parametrize('stage', ['calibrate', 'evaluate', 'latency'])
def test_every_non_profile_cuda_stage_uses_the_shared_lease(
        tmp_path, monkeypatch, stage):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    entered = []
    _mock_lease(monkeypatch, entered)

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / f'{stage}.json'
        _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact)

    result = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, runner,
        repository_root=tmp_path, stages=(stage,),
        gpu_lock_path=(
            tmp_path / 'shared/work_dirs/optimization/gpu.lock'),
        shared_lock_root=tmp_path / 'shared').run_next()

    assert result.exit_code == 0
    assert [record[2] for record in entered] == [f'fixture:{stage}']


def test_compare_is_serialized_by_controller_wide_single_writer_lock(
        tmp_path):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    campaign = tmp_path / 'work_dirs/optimization'
    nested = []

    def must_not_run(*args):
        raise AssertionError('concurrent compare runner must not start')

    second = OptimizationController(
        campaign, candidate, must_not_run, repository_root=tmp_path,
        stages=('compare',))

    def first_runner(candidate, stage, stage_dir, attempt):
        nested.append(second.run_next())
        artifact = stage_dir / 'compare.json'
        _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact)

    first = OptimizationController(
        campaign, candidate, first_runner, repository_root=tmp_path,
        stages=('compare',))

    assert first.run_next().exit_code == 0
    assert nested[0].exit_code == 75


def test_crashed_attempt_remains_in_lineage_when_next_process_resumes(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController
    from mambapose_repro.state import StateStore

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])
    campaign = tmp_path / 'work_dirs/optimization'

    def crash(*args):
        raise KeyboardInterrupt('simulated process loss')

    with pytest.raises(KeyboardInterrupt, match='simulated process loss'):
        OptimizationController(
            campaign, candidate, crash, repository_root=tmp_path,
            stages=('evaluate',)).run_next()

    def resume(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / 'evaluate.json'
        _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact)

    result = OptimizationController(
        campaign, candidate, resume, repository_root=tmp_path,
        stages=('evaluate',)).run_next()
    lineage = StateStore(campaign).read()[
        'runs']['fixture:evaluate']['retry_lineage']

    assert result.exit_code == 0
    assert [(item['attempt'], item['status']) for item in lineage] == [
        (1, 'started'),
        (2, 'started'),
        (2, 'complete'),
    ]


def test_controller_writes_immutable_expected_run_plan(tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])
    campaign = tmp_path / 'work_dirs/optimization'

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / 'evaluate.json'
        _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact)

    result = OptimizationController(
        campaign, candidate, runner, repository_root=tmp_path,
        stages=('evaluate',),
        expected_run_ids=('fixture:evaluate', 'other:evaluate')).run_next()

    assert result.exit_code == 0
    assert json.loads((campaign / 'campaign-plan.json').read_text()) == {
        'schema_version': 1,
        'run_ids': ['fixture:evaluate', 'other:evaluate'],
    }

    incompatible = OptimizationController(
        campaign, candidate, runner, repository_root=tmp_path,
        stages=('evaluate',),
        expected_run_ids=('fixture:evaluate',)).run_next()
    assert incompatible.exit_code == 78
    assert 'campaign plan mismatch' in incompatible.message


def test_artifact_hash_permission_error_is_normalized_to_blocked_78(
        tmp_path, monkeypatch):
    from mambapose_opt import controller
    from mambapose_opt.controller import OptimizationController
    from mambapose_repro.state import StateStore

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])
    real_sha256 = controller._sha256

    def denied(path):
        if Path(path).name == 'evaluate.json':
            raise PermissionError('artifact denied')
        return real_sha256(path)

    monkeypatch.setattr(controller, '_sha256', denied)

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / 'evaluate.json'
        _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact)

    campaign = tmp_path / 'work_dirs/optimization'
    result = OptimizationController(
        campaign, candidate, runner, repository_root=tmp_path,
        stages=('evaluate',)).run_next()

    assert result.exit_code == 78
    run = StateStore(campaign).read()['runs']['fixture:evaluate']
    assert run['status'] == 'blocked'


def test_additional_validator_value_error_is_normalized_to_blocked_78(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / 'evaluate.json'
        _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact)

    def invalid(*args):
        raise ValueError('validator failed')

    result = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, runner,
        repository_root=tmp_path, stages=('evaluate',),
        artifact_validator=invalid).run_next()

    assert result.exit_code == 78
    assert 'validator failed' in result.message


def test_additional_validator_runtime_error_is_normalized_to_blocked_78(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / 'evaluate.json'
        _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact)

    def invalid(*args):
        raise RuntimeError('integrity validator crashed')

    result = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, runner,
        repository_root=tmp_path, stages=('evaluate',),
        artifact_validator=invalid).run_next()

    assert result.exit_code == 78
    assert 'integrity validator crashed' in result.message


def test_checkpoint_validator_value_error_is_normalized_to_blocked_78(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])

    def runner(candidate, stage, stage_dir, attempt):
        for name in ('best_AP_epoch_2.pth', 'epoch_1.pth', 'epoch_2.pth'):
            (stage_dir / name).write_bytes(b'checkpoint')
        artifact = stage_dir / 'train.json'
        _write_generic_artifact(artifact, stage)
        return _outcome(stage, artifact)

    result = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, runner,
        repository_root=tmp_path, stages=('train',),
        checkpoint_validator=lambda path: (_ for _ in ()).throw(
            ValueError('checkpoint validator failed'))).run_next()

    assert result.exit_code == 78
    assert 'checkpoint validator failed' in result.message


@pytest.mark.parametrize(
    ('campaign_root', 'lock_path'),
    [
        ('outside', 'work_dirs/optimization/gpu.lock'),
        ('work_dirs/optimization', 'arbitrary/gpu.lock'),
    ],
)
def test_controller_rejects_out_of_contract_output_roots(
        tmp_path, campaign_root, lock_path):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)

    with pytest.raises(ValueError, match='optimization'):
        OptimizationController(
            tmp_path / campaign_root, candidate, lambda *args: None,
            repository_root=tmp_path, stages=('compare',),
            gpu_lock_path=tmp_path / lock_path)


def test_controller_lock_cannot_escape_campaign_root(tmp_path):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)

    with pytest.raises(ValueError, match='controller lock'):
        OptimizationController(
            tmp_path / 'work_dirs/optimization', candidate,
            lambda *args: None, repository_root=tmp_path,
            stages=('compare',),
            controller_lock_path=tmp_path / 'outside/controller.lock')


def test_controller_rejects_suffix_matching_lock_outside_trusted_anchor(
        tmp_path):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    trusted = tmp_path / 'trusted-checkout'
    external = tmp_path / 'external/work_dirs/optimization/gpu.lock'

    with pytest.raises(ValueError, match='canonical'):
        OptimizationController(
            tmp_path / 'work_dirs/optimization', candidate,
            lambda *args: None, repository_root=tmp_path,
            stages=('compare',), gpu_lock_path=external,
            shared_lock_root=trusted)


def test_run_cli_rejects_noncanonical_lock_override_before_controller(
        tmp_path, monkeypatch, capsys):
    from tools.optimization import run_campaign

    candidate = _candidate(tmp_path)
    monkeypatch.setattr(
        run_campaign, 'load_candidate_manifest', lambda path: (candidate,))
    monkeypatch.setattr(
        run_campaign, '_canonical_checkout_root',
        lambda: tmp_path / 'canonical', raising=False)

    class MustNotConstruct:
        def __init__(self, *args, **kwargs):
            raise AssertionError('controller must not be constructed')

    monkeypatch.setattr(run_campaign, 'OptimizationController', MustNotConstruct)
    monkeypatch.setattr(sys, 'argv', [
        'run_campaign.py', '--run', '--gpu-lock-path',
        str(tmp_path / 'external/work_dirs/optimization/gpu.lock'),
    ])

    assert run_campaign.main() == 78
    assert 'canonical GPU lock' in capsys.readouterr().err


def test_run_cli_normalizes_invalid_campaign_path_to_78_without_traceback(
        tmp_path, monkeypatch, capsys):
    from tools.optimization import run_campaign

    candidate = _candidate(tmp_path)
    monkeypatch.setattr(
        run_campaign, 'load_candidate_manifest', lambda path: (candidate,))
    monkeypatch.setattr(
        run_campaign, '_canonical_checkout_root',
        lambda: tmp_path / 'canonical', raising=False)
    monkeypatch.setattr(sys, 'argv', [
        'run_campaign.py', '--run', '--campaign-root',
        str(tmp_path / 'external/work_dirs/optimization'),
    ])

    assert run_campaign.main() == 78
    captured = capsys.readouterr()
    assert 'optimization campaign root' in captured.err
    assert 'Traceback' not in captured.err


def test_run_cli_normalizes_runner_path_value_error_to_78(
        tmp_path, monkeypatch, capsys):
    from tools.optimization import run_campaign

    candidate = _candidate(tmp_path)
    monkeypatch.setattr(
        run_campaign, 'load_candidate_manifest', lambda path: (candidate,))
    monkeypatch.setattr(
        run_campaign, '_canonical_checkout_root',
        lambda: tmp_path / 'canonical', raising=False)

    class InvalidRunner:
        def __init__(self, *args, **kwargs):
            raise ValueError('runner output path escapes repository')

    monkeypatch.setattr(run_campaign, 'SubprocessStageRunner', InvalidRunner)
    monkeypatch.setattr(sys, 'argv', ['run_campaign.py', '--run'])

    assert run_campaign.main() == 78
    captured = capsys.readouterr()
    assert 'runner output path escapes repository' in captured.err
    assert 'Traceback' not in captured.err


def test_retry_budget_persists_across_controller_process_restarts(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController, StageOutcome
    from mambapose_repro.state import StateStore

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])
    campaign = tmp_path / 'work_dirs/optimization'
    clock = [datetime(2026, 8, 27, tzinfo=timezone.utc)]

    def transient(candidate, stage, stage_dir, attempt):
        return StageOutcome(
            'fixture:evaluate', 'evaluate', 'fixture', 75,
            f'transient-{attempt}')

    results = []
    for _ in range(3):
        results.append(OptimizationController(
            campaign, candidate, transient, repository_root=tmp_path,
            stages=('evaluate',), max_attempts=3,
            now=lambda: clock[0]).run_next())
        clock[0] += timedelta(minutes=10)

    assert [result.exit_code for result in results] == [75, 75, 78]
    run = StateStore(campaign).read()['runs']['fixture:evaluate']
    assert run['status'] == 'exhausted'
    assert run['attempt'] == 3


def test_reboot_style_running_state_at_budget_is_exhausted_without_runner(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController
    from mambapose_repro.state import StateStore

    candidate = _candidate(tmp_path)
    _mock_lease(monkeypatch, [])
    campaign = tmp_path / 'work_dirs/optimization'
    store = StateStore(campaign)
    store.transition(
        'fixture:evaluate', 'running', attempt=3,
        retry_lineage=[{
            'attempt': 3,
            'status': 'started',
            'timestamp': 'before-reboot',
        }])
    calls = []

    result = OptimizationController(
        campaign, candidate, lambda *args: calls.append(args),
        repository_root=tmp_path, stages=('evaluate',),
        max_attempts=3).run_next()

    assert result.exit_code == 78
    assert calls == []
    assert store.read()['runs']['fixture:evaluate']['status'] == 'exhausted'
