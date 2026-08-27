from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from mmengine.config import ConfigDict
import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def _train_fixture(tmp_path: Path):
    from mambapose_opt.distill_smoke import validate_coco_train_assets_at_root

    root = tmp_path / 'repo'
    images = {
        'train2017/000000000001.jpg': b'first-jpeg',
        'train2017/000000000002.jpg': b'second-jpeg',
    }
    for name, content in images.items():
        target = root / 'data/coco' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    annotation_member = 'annotations/person_keypoints_train2017.json'
    annotation = b'{"images": [{"id": 1}, {"id": 2}], "annotations": []}'
    annotation_path = root / 'data/coco' / annotation_member
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_bytes(annotation)

    image_archive = root / 'work_dirs/reproduction/downloads/train2017.zip'
    annotation_archive = (
        root / 'work_dirs/reproduction/downloads/annotations_trainval2017.zip')
    _write_zip(image_archive, images)
    _write_zip(annotation_archive, {annotation_member: annotation})

    inventory_path = root / 'data/inventory.json'
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps({
        'schema_version': 1,
        'assets': [
            {
                'id': 'coco-train2017',
                'path': 'work_dirs/reproduction/downloads/train2017.zip',
                'sha256': _sha256(image_archive),
                'required_paths': ['data/coco/train2017'],
            },
            {
                'id': 'coco-annotations',
                'path': (
                    'work_dirs/reproduction/downloads/'
                    'annotations_trainval2017.zip'),
                'sha256': _sha256(annotation_archive),
                'required_paths': [
                    'data/coco/annotations/person_keypoints_train2017.json'],
            },
        ],
    }, sort_keys=True))

    authority = ConfigDict(
        inventory_path='data/inventory.json',
        inventory_sha256=_sha256(inventory_path),
        image_archive_path=(
            'work_dirs/reproduction/downloads/train2017.zip'),
        image_archive_sha256=_sha256(image_archive),
        image_inventory_asset_id='coco-train2017',
        image_prefix='train2017/',
        image_count=2,
        annotation_archive_path=(
            'work_dirs/reproduction/downloads/annotations_trainval2017.zip'),
        annotation_archive_sha256=_sha256(annotation_archive),
        annotation_inventory_asset_id='coco-annotations',
        annotation_path=(
            'data/coco/annotations/person_keypoints_train2017.json'),
        annotation_member=annotation_member,
    )
    dataset = ConfigDict(
        type='CocoDataset', data_root='data/coco/', data_mode='topdown',
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=ConfigDict(img='train2017/'),
        pipeline=[ConfigDict(type='LoadImage'), ConfigDict(type='PackPoseInputs')])
    return root, authority, dataset, validate_coco_train_assets_at_root


def test_smoke_dataloader_clones_packed_pipeline_at_batch_one():
    from mambapose_opt.distill_smoke import smoke_dataloader_config

    source = ConfigDict(
        batch_size=128,
        num_workers=4,
        persistent_workers=True,
        sampler=ConfigDict(type='DefaultSampler', shuffle=True),
        dataset=ConfigDict(
            type='CocoDataset', data_root='data/coco/', data_mode='topdown',
            ann_file='annotations/person_keypoints_train2017.json',
            data_prefix=ConfigDict(img='train2017/'),
            pipeline=[
                ConfigDict(type='LoadImage'),
                ConfigDict(type='TopdownAffine', input_size=(192, 256)),
                ConfigDict(type='GenerateTarget'),
                ConfigDict(type='PackPoseInputs'),
            ]))

    smoke = smoke_dataloader_config(source)

    assert source.batch_size == 128
    assert source.num_workers == 4
    assert source.persistent_workers is True
    assert source.sampler.shuffle is True
    assert smoke.batch_size == 1
    assert smoke.num_workers == 0
    assert smoke.persistent_workers is False
    assert smoke.sampler == dict(type='DefaultSampler', shuffle=False)
    assert smoke.dataset.pipeline == source.dataset.pipeline
    assert smoke.dataset.pipeline[-1].type == 'PackPoseInputs'


def test_train_asset_validation_binds_archives_and_extracted_corpus(tmp_path):
    root, authority, dataset, validate = _train_fixture(tmp_path)

    binding = validate(authority, dataset, root)

    assert binding.image_count == 2
    assert binding.annotation_sha256 == _sha256(
        root / authority.annotation_path)
    assert binding.image_archive_sha256 == authority.image_archive_sha256
    assert len(binding.corpus_sha256) == 64


def test_train_asset_validation_rejects_mutated_extracted_image(tmp_path):
    root, authority, dataset, validate = _train_fixture(tmp_path)
    (root / 'data/coco/train2017/000000000002.jpg').write_bytes(b'mutated')

    with pytest.raises(ValueError, match='extracted train2017 corpus'):
        validate(authority, dataset, root)


def _valid_artifact(tmp_path: Path):
    from mambapose_opt.distill_smoke import canonical_json_sha256

    root = tmp_path / 'repo'
    run = root / 'work_dirs/optimization/accuracy-first/smoke-fixture'
    run.mkdir(parents=True)
    files = {
        'configs/optimization/accuracy_first/distill_s_v1_from_b.py': b'config',
        'work_dirs/reproduction/runs/coco-b/teacher.pth': b'teacher',
        'work_dirs/reproduction/runs/coco-s-v1/student.pth': b'student',
        'data/inventory.json': b'inventory',
        'work_dirs/reproduction/downloads/train2017.zip': b'images-archive',
        'work_dirs/reproduction/downloads/annotations.zip': b'ann-archive',
        'data/coco/annotations/person_keypoints_train2017.json': json.dumps({
            'images': [{
                'id': 1,
                'file_name': '000000000001.jpg',
            }],
            'annotations': [],
        }, sort_keys=True).encode(),
        'data/coco/train2017/000000000001.jpg': b'batch-image',
        'work_dirs/optimization/accuracy-first/smoke-fixture/distiller.pth':
            b'full-distiller',
        'work_dirs/optimization/accuracy-first/smoke-fixture/student.pth':
            b'exported-student',
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
    subprocess.run([
        'git', 'add',
        'configs/optimization/accuracy_first/distill_s_v1_from_b.py',
    ], cwd=root, check=True)
    subprocess.run([
        'git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.test',
        'commit', '-qm', 'fixture config',
    ], cwd=root, check=True)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()

    lease = {
        'stage_id': 'distill-smoke:distill-s-v1-from-coco-b',
        'pid': 42,
        'boot_id': '12345678-1234-5678-1234-567812345678',
        'timestamp': '2026-08-27T12:00:00+00:00',
        'device_index': 0,
        'allowed_pids': [42],
        'lease_id': '1' * 64,
    }
    file_binding = lambda relative: {
        'path': relative,
        'sha256': _sha256(root / relative),
    }
    value = {
        'schema_version': 1,
        'kind': 'mambapose-real-batch-distillation-smoke',
        'claims': {
            'coco_ap': False,
            'latency': False,
            'training_batch_vram': False,
        },
        'execution': {
            'python_seed': 0,
            'numpy_seed': 0,
            'torch_seed': 0,
            'deterministic_algorithms': True,
            'python_dont_write_bytecode': True,
            'vram_scope': 'batch-1-smoke-not-training-batch',
        },
        'source': {
            'git_commit': commit,
            'config': file_binding(
                'configs/optimization/accuracy_first/'
                'distill_s_v1_from_b.py'),
            'resolved_config_sha256': 'b' * 64,
        },
        'inputs': {
            'teacher_checkpoint': file_binding(
                'work_dirs/reproduction/runs/coco-b/teacher.pth'),
            'student_checkpoint': file_binding(
                'work_dirs/reproduction/runs/coco-s-v1/student.pth'),
        },
        'data': {
            'dataset': 'coco',
            'split': 'train2017',
            'pipeline_terminal': 'PackPoseInputs',
            'original_batch_size': 128,
            'smoke_batch_size': 1,
            'smoke_num_workers': 0,
            'smoke_persistent_workers': False,
            'inventory': file_binding('data/inventory.json'),
            'image_archive': file_binding(
                'work_dirs/reproduction/downloads/train2017.zip'),
            'annotation_archive': file_binding(
                'work_dirs/reproduction/downloads/annotations.zip'),
            'annotation': file_binding(
                'data/coco/annotations/person_keypoints_train2017.json'),
            'corpus_digest_algorithm': (
                'sha256-zip-member-and-extracted-content-v1'),
            'corpus_sha256': 'c' * 64,
            'corpus_image_count': 118287,
            'dataloader_override_sha256': 'd' * 64,
            'batch': {
                'size': 1,
                'image_ids': [1],
                'image_paths': ['data/coco/train2017/000000000001.jpg'],
                'image_sha256': [_sha256(
                    root / 'data/coco/train2017/000000000001.jpg')],
                'packed_sha256': 'f' * 64,
            },
        },
        'gpu': {
            'device_index': 0,
            'lease': lease,
            'lease_sha256': canonical_json_sha256(lease),
        },
        'checks': {
            'losses': {
                'heatmap_distill_mse': 0.5,
                'loss_heatmap_distill': 0.5,
                'loss_kpt': 1.0,
            },
            'teacher_frozen': True,
            'teacher_gradient_tensors': 0,
            'student_gradient_tensors': 10,
            'student_gradient_elements': 100,
            'student_gradients_finite': True,
            'peak_allocated_bytes': 1000,
            'peak_reserved_bytes': 2000,
            'heatmap_shape': [1, 17, 64, 48],
            'reference_heatmaps_sha256': '1' * 64,
            'full_restore_heatmaps_sha256': '1' * 64,
            'export_restore_heatmaps_sha256': '1' * 64,
            'full_restore_exact': True,
            'export_restore_exact': True,
        },
        'artifacts': {
            'distiller_checkpoint': file_binding(
                'work_dirs/optimization/accuracy-first/'
                'smoke-fixture/distiller.pth'),
            'student_checkpoint': file_binding(
                'work_dirs/optimization/accuracy-first/'
                'smoke-fixture/student.pth'),
        },
    }
    artifact = run / 'smoke.json'
    artifact.write_text(json.dumps(value, sort_keys=True))
    return root, artifact, value


def test_smoke_artifact_strict_schema_and_bound_files_round_trip(tmp_path):
    from mambapose_opt.distill_smoke import load_smoke_artifact

    root, artifact, value = _valid_artifact(tmp_path)

    assert load_smoke_artifact(artifact, repository_root=root) == value


@pytest.mark.parametrize('tamper', [
    'unknown-field', 'checkpoint-content', 'batch-id', 'artifact-path',
    'source-commit',
])
def test_smoke_artifact_rejects_schema_or_file_tampering(tmp_path, tamper):
    from mambapose_opt.distill_smoke import load_smoke_artifact

    root, artifact, value = _valid_artifact(tmp_path)
    if tamper == 'unknown-field':
        value['AP'] = 72.8
        artifact.write_text(json.dumps(value))
    elif tamper == 'checkpoint-content':
        (artifact.parent / 'student.pth').write_bytes(b'tampered')
    elif tamper == 'batch-id':
        value['data']['batch']['image_ids'] = [2]
        artifact.write_text(json.dumps(value))
    elif tamper == 'artifact-path':
        value['artifacts']['student_checkpoint']['path'] = (
            'work_dirs/optimization/accuracy-first/another-run/student.pth')
        artifact.write_text(json.dumps(value))
    else:
        value['source']['git_commit'] = 'a' * 40
        artifact.write_text(json.dumps(value))

    with pytest.raises(ValueError, match=(
            'invalid fields|sha256 mismatch|batch image identity|artifact path|'
            'tracked|clean commit')):
        load_smoke_artifact(artifact, repository_root=root)


def test_existing_output_refuses_partial_artifacts_before_preflight(
        tmp_path, monkeypatch):
    import mambapose_opt.distill_smoke as smoke

    output = tmp_path / 'work_dirs/optimization/accuracy-first/existing'
    output.mkdir(parents=True)
    (output / 'partial.pth').write_bytes(b'partial')
    monkeypatch.setattr(
        smoke, 'build_smoke_preflight',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('preflight must not run for an existing output')))

    with pytest.raises(FileExistsError, match='refusing to overwrite'):
        smoke.run_distill_smoke(
            repository_root=tmp_path,
            config_path=Path(
                'configs/optimization/accuracy_first/'
                'distill_s_v1_from_b.py'),
            output_relative=Path(
                'work_dirs/optimization/accuracy-first/existing'),
            device_index=0)
    assert (output / 'partial.pth').read_bytes() == b'partial'


def test_invalid_preflight_never_enters_gpu_or_creates_partial_output(
        tmp_path, monkeypatch):
    import mambapose_opt.distill_smoke as smoke

    def reject(*args, **kwargs):
        raise RuntimeError('dirty executable source')

    @contextmanager
    def forbidden_lease(*args, **kwargs):
        raise AssertionError('GPU lease entered before preflight completed')
        yield

    monkeypatch.setattr(smoke, 'build_smoke_preflight', reject)
    monkeypatch.setattr(smoke, 'exclusive_cuda_stage', forbidden_lease)
    monkeypatch.setattr(
        smoke, '_execute_smoke',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('CUDA runtime entered after invalid preflight')))
    relative = Path('work_dirs/optimization/accuracy-first/new-smoke')

    with pytest.raises(RuntimeError, match='dirty executable source'):
        smoke.run_distill_smoke(
            repository_root=tmp_path,
            config_path=Path(
                'configs/optimization/accuracy_first/'
                'distill_s_v1_from_b.py'),
            output_relative=relative,
            device_index=0)
    assert not (tmp_path / relative).exists()


@pytest.mark.parametrize('failure', ['dirty-source', 'checkpoint-hash'])
def test_real_preflight_rejects_before_data_validation_or_gpu(
        tmp_path, monkeypatch, failure):
    import mambapose_opt.distill_smoke as smoke

    root = tmp_path / 'repo'
    root.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
    (root / '.gitignore').write_text('data/\nwork_dirs/\n')
    (root / 'data').mkdir()
    config = root / (
        'configs/optimization/accuracy_first/distill_s_v1_from_b.py')
    config.parent.mkdir(parents=True)
    teacher = root / 'work_dirs/reproduction/runs/coco-b/teacher.pth'
    student = root / 'work_dirs/reproduction/runs/coco-s-v1/student.pth'
    teacher.parent.mkdir(parents=True)
    student.parent.mkdir(parents=True)
    teacher.write_bytes(b'teacher')
    student.write_bytes(b'student')
    config.write_text(
        "experiment_id = 'fixture'\n"
        "model = dict(type='MambaPoseHeatmapDistiller', "
        "teacher_checkpoint='work_dirs/reproduction/runs/coco-b/teacher.pth', "
        "teacher_checkpoint_sha256='0' * 64, "
        "student_checkpoint=("
        "'work_dirs/reproduction/runs/coco-s-v1/student.pth'), "
        f"student_checkpoint_sha256='{_sha256(student)}')\n"
        "train_dataloader = dict(batch_size=128, dataset=dict())\n")
    subprocess.run(['git', 'add', '.gitignore', str(config.relative_to(root))],
                   cwd=root, check=True)
    subprocess.run([
        'git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.test',
        'commit', '-qm', 'fixture source',
    ], cwd=root, check=True)
    if failure == 'dirty-source':
        (root / 'unexpected_executable.py').write_text('raise SystemExit\n')

    monkeypatch.setattr(
        smoke, 'validate_coco_train_assets_at_root',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('data validation occurred after invalid preflight')))

    @contextmanager
    def forbidden_lease(*args, **kwargs):
        raise AssertionError('GPU lease entered after invalid preflight')
        yield

    monkeypatch.setattr(smoke, 'exclusive_cuda_stage', forbidden_lease)
    output = Path('work_dirs/optimization/accuracy-first/fixture')
    expected = 'clean worktree' if failure == 'dirty-source' else (
        'teacher checkpoint sha256 mismatch')
    with pytest.raises((RuntimeError, ValueError), match=expected):
        smoke.run_distill_smoke(
            repository_root=root,
            config_path=config.relative_to(root),
            output_relative=output,
            device_index=0)
    assert not (root / output).exists()


def test_canonical_gpu_lock_is_shared_with_primary_checkout():
    from mambapose_opt.gpu_guard import canonical_gpu_lock_path

    root = Path(__file__).resolve().parents[2]
    common = subprocess.check_output(
        ['git', 'rev-parse', '--git-common-dir'], cwd=root, text=True).strip()
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = root / common_path

    assert canonical_gpu_lock_path(root) == (
        common_path.resolve().parent / 'work_dirs/optimization/gpu.lock')


def test_smoke_cli_rejects_output_outside_optimization_before_execution():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            'tools/optimization/smoke_distiller.py',
            '--config',
            'configs/optimization/accuracy_first/distill_s_v1_from_b.py',
            '--output-root', '/tmp/not-an-optimization-artifact',
            '--device-index', '0',
        ],
        cwd=root,
        env={
            'PATH': os.environ.get('PATH', ''),
            'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONNOUSERSITE': '1',
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert 'work_dirs/optimization' in result.stderr
