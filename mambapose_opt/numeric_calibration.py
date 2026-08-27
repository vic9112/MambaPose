"""Production calibration contracts for the full S-V1 Route 3 baseline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping
import zipfile

from torch import nn


class CalibrationContractError(ValueError):
    """Raised before calibration when source, data, or hooks are ambiguous."""


@dataclass(frozen=True)
class CalibrationTargets:
    ss2d_boundaries: tuple[str, ...]
    vmamba_in_proj: tuple[str, ...]
    vmamba_out_proj: tuple[str, ...]
    attention_qkv: tuple[str, ...]
    pif_boundaries: tuple[str, ...]
    heatmap_projection: tuple[str, ...]
    transition_parameters: tuple[str, ...]
    functional_observers: tuple[str, ...]
    unsupported_internals: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, relative: Path, label: str) -> Path:
    if relative.is_absolute():
        raise CalibrationContractError(f'{label} must be repository-relative')
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise CalibrationContractError(f'{label} escapes repository root') from error
    if not path.is_file() and label != 'image prefix':
        raise CalibrationContractError(f'{label} is missing: {relative}')
    if label == 'image prefix' and not path.is_dir():
        raise CalibrationContractError(f'{label} is missing: {relative}')
    return path


def _verified_archive(
        root: Path, inventory: Mapping[str, Any], asset_id: str) -> tuple[Path, str]:
    assets = inventory.get('assets')
    if not isinstance(assets, list):
        raise CalibrationContractError('dataset inventory assets are invalid')
    matches = [item for item in assets if isinstance(item, Mapping)
               and item.get('id') == asset_id]
    if len(matches) != 1:
        raise CalibrationContractError(
            f'dataset inventory requires exactly one {asset_id} asset')
    item = matches[0]
    relative = item.get('path')
    expected = item.get('sha256')
    if not isinstance(relative, str) or not isinstance(expected, str) \
            or not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise CalibrationContractError(
            f'dataset inventory {asset_id} binding is invalid')
    archive = _inside(root, Path(relative), f'{asset_id} archive')
    actual = _sha256(archive)
    if actual != expected:
        raise CalibrationContractError(
            f'{asset_id} archive content disagrees with inventory sha256')
    return archive, actual


def _compare_zip_member(
        archive: zipfile.ZipFile, member: zipfile.ZipInfo, extracted: Path,
        *, label: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member, 'r') as source, extracted.open('rb') as target:
        while True:
            source_chunk = source.read(1024 * 1024)
            target_chunk = target.read(1024 * 1024)
            if source_chunk != target_chunk:
                raise CalibrationContractError(
                    f'{label} extracted content disagrees with official archive')
            if not source_chunk:
                break
            digest.update(source_chunk)
    return digest.hexdigest()


def _verified_dataset_authority(
        root: Path, images: Path, annotation: Path) -> dict[str, Any]:
    inventory_path = _inside(root, Path('data/inventory.json'),
                             'dataset inventory')
    try:
        inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationContractError('dataset inventory is not valid JSON') from error
    if not isinstance(inventory, Mapping):
        raise CalibrationContractError('dataset inventory root is invalid')
    train_archive, train_archive_sha256 = _verified_archive(
        root, inventory, 'coco-train2017')
    annotation_archive, annotation_archive_sha256 = _verified_archive(
        root, inventory, 'coco-annotations')

    extracted = {
        path.name: path for path in images.iterdir()
        if path.is_file() and path.suffix.lower() == '.jpg'
    }
    content_aggregate = hashlib.sha256()
    order_aggregate = hashlib.sha256()
    try:
        with zipfile.ZipFile(train_archive) as archive:
            members = [
                member for member in archive.infolist()
                if not member.is_dir()
                and member.filename.startswith('train2017/')
                and member.filename.lower().endswith('.jpg')
            ]
            names = [Path(member.filename).name for member in members]
            if not members or len(names) != len(set(names)):
                raise CalibrationContractError(
                    'official train2017 archive image membership is invalid')
            if set(names) != set(extracted):
                raise CalibrationContractError(
                    'extracted train2017 image membership/content is incomplete')
            for member, name in zip(members, names):
                member_sha256 = _compare_zip_member(
                    archive, member, extracted[name], label=member.filename)
                encoded = member.filename.encode('utf-8')
                order_aggregate.update(encoded)
                order_aggregate.update(b'\0')
                content_aggregate.update(encoded)
                content_aggregate.update(b'\0')
                content_aggregate.update(member_sha256.encode('ascii'))
                content_aggregate.update(b'\0')
    except (OSError, zipfile.BadZipFile) as error:
        raise CalibrationContractError(
            'official train2017 archive cannot be verified') from error

    annotation_member_name = (
        'annotations/person_keypoints_train2017.json')
    try:
        with zipfile.ZipFile(annotation_archive) as archive:
            members = [member for member in archive.infolist()
                       if member.filename == annotation_member_name
                       and not member.is_dir()]
            if len(members) != 1:
                raise CalibrationContractError(
                    'official annotation archive member is missing or duplicated')
            annotation_member_sha256 = _compare_zip_member(
                archive, members[0], annotation,
                label=annotation_member_name)
    except (OSError, zipfile.BadZipFile) as error:
        raise CalibrationContractError(
            'official annotation archive cannot be verified') from error

    return {
        'inventory': 'data/inventory.json',
        'inventory_sha256': _sha256(inventory_path),
        'train_archive': train_archive.relative_to(root).as_posix(),
        'train_archive_sha256': train_archive_sha256,
        'image_count': len(extracted),
        'image_content_algorithm': 'sha256-zip-member-bytes-v1',
        'image_content_aggregate_sha256': content_aggregate.hexdigest(),
        'image_order_algorithm': 'sha256-zip-central-directory-order-v1',
        'image_order_sha256': order_aggregate.hexdigest(),
        'annotation_archive': annotation_archive.relative_to(root).as_posix(),
        'annotation_archive_sha256': annotation_archive_sha256,
        'annotation_member': annotation_member_name,
        'annotation_member_sha256': annotation_member_sha256,
    }


def calibration_identity(
        *, repository_root: Path, candidate_id: str, config: Path,
        checkpoint: Path, expected_checkpoint_sha256: str, policy: Path,
        split: str, annotation: Path, image_prefix: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if candidate_id != 'full-s-v1':
        raise CalibrationContractError('calibration source must be full-s-v1')
    if config.as_posix() != 'configs/reproduction/coco_s_v1.py':
        raise CalibrationContractError('calibration config must be full S-V1')
    if split != 'train2017':
        raise CalibrationContractError('numeric calibration requires train2017')
    if annotation.as_posix() != (
            'data/coco/annotations/person_keypoints_train2017.json'):
        raise CalibrationContractError(
            'numeric calibration requires official train2017 annotations')
    if image_prefix.as_posix() != 'data/coco/train2017':
        raise CalibrationContractError(
            'numeric calibration requires official train2017 images')
    config_path = _inside(root, config, 'config')
    checkpoint_path = _inside(root, checkpoint, 'checkpoint')
    policy_path = _inside(root, policy, 'policy')
    annotation_path = _inside(root, annotation, 'annotation')
    images = _inside(root, image_prefix, 'image prefix')
    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise CalibrationContractError('checkpoint sha256 disagrees with manifest')
    dataset_authority = _verified_dataset_authority(
        root, images, annotation_path)
    return {
        'candidate_id': candidate_id,
        'config': config.as_posix(),
        'config_sha256': _sha256(config_path),
        'checkpoint': checkpoint.as_posix(),
        'checkpoint_sha256': checkpoint_sha256,
        'policy': policy.as_posix(),
        'policy_sha256': _sha256(policy_path),
        'split': split,
        'dataset': {
            'annotation': annotation.as_posix(),
            'annotation_sha256': _sha256(annotation_path),
            'image_prefix': image_prefix.as_posix(),
            **dataset_authority,
        },
    }


def discover_calibration_targets(model: nn.Module) -> CalibrationTargets:
    modules = dict(model.named_modules())
    ss2d = tuple(
        name for name, module in modules.items()
        if name and type(module).__name__ == 'SS2D')
    if not ss2d:
        raise CalibrationContractError('required SS2D boundaries are missing')
    in_proj = tuple(f'{name}.in_proj' for name in ss2d)
    out_proj = tuple(f'{name}.out_proj' for name in ss2d)
    if any(name not in modules for name in in_proj + out_proj):
        raise CalibrationContractError('required VMamba in/out projections are missing')

    attention = tuple(
        f'{name}.to_qkv' for name, module in modules.items()
        if name and type(module).__name__ == 'Attention'
        and hasattr(module, 'to_qkv'))
    if len(attention) != 6 or len(set(attention)) != 6:
        raise CalibrationContractError(
            'calibration requires exactly six Transformer Q/K/V projections')
    pif = tuple(
        name for name, module in modules.items()
        if name and type(module).__name__ == 'PoseInteraction')
    if len(pif) != 1:
        raise CalibrationContractError(
            'calibration requires exactly one PIF boundary')
    explicit_heatmap = tuple(
        name for name, module in modules.items()
        if name.endswith('heatmap_projection') and isinstance(module, nn.Linear))
    head_linears = tuple(
        name for name, module in modules.items()
        if '.tokenpose.mlp_head.' in name and isinstance(module, nn.Linear))
    heatmap = explicit_heatmap or head_linears[-1:]
    if len(heatmap) != 1:
        raise CalibrationContractError(
            'calibration requires one unambiguous heatmap projection')

    parameter_names = set(dict(model.named_parameters()))
    transition_suffixes = (
        'A_logs', 'Ds', 'dt_projs_bias', 'dt_projs_weight', 'x_proj_weight')
    transition = tuple(
        f'{name}.{suffix}' for name in ss2d for suffix in transition_suffixes)
    missing_parameters = tuple(
        name for name in transition if name not in parameter_names)
    if missing_parameters:
        raise CalibrationContractError(
            f'required accessible transition parameters are missing: '
            f'{missing_parameters}')
    functional_observers = tuple(
        name for name in ss2d
        if callable(getattr(modules[name], 'set_numeric_observer', None)))
    if functional_observers != ss2d:
        missing = tuple(name for name in ss2d
                        if name not in functional_observers)
        raise CalibrationContractError(
            f'required SS2D numeric observers are missing: {missing}')
    unsupported = tuple(
        f'{name}.selective_scan_internal_state:opaque_cuda_kernel'
        for name in ss2d)
    return CalibrationTargets(
        ss2d_boundaries=ss2d,
        vmamba_in_proj=in_proj,
        vmamba_out_proj=out_proj,
        attention_qkv=attention,
        pif_boundaries=pif,
        heatmap_projection=heatmap,
        transition_parameters=transition,
        functional_observers=functional_observers,
        unsupported_internals=unsupported,
    )


def validate_calibration_artifact(
        value: Mapping[str, Any], *,
        expected_candidate_id: str | None = None) -> Mapping[str, Any]:
    required = {'schema_version', 'candidate_id', 'stage', 'identity',
                'protocol', 'hooks'}
    if not isinstance(value, Mapping) or set(value) != required:
        raise CalibrationContractError(
            'calibration artifact fields do not match schema v1')
    if value['schema_version'] != 1 or value['stage'] != 'calibrate' \
            or not isinstance(value['candidate_id'], str) \
            or not value['candidate_id']:
        raise CalibrationContractError('calibration artifact identity is invalid')
    if (expected_candidate_id is not None
            and value['candidate_id'] != expected_candidate_id):
        raise CalibrationContractError(
            'calibration artifact target candidate mismatch')
    identity = value['identity']
    if not isinstance(identity, Mapping) or identity.get('split') != 'train2017' \
            or identity.get('candidate_id') != 'full-s-v1':
        raise CalibrationContractError('calibration identity must bind train2017')
    protocol = value['protocol']
    if not isinstance(protocol, Mapping):
        raise CalibrationContractError('calibration protocol is invalid')
    expected = {
        'model_mode': 'eval', 'grad_enabled': False, 'shuffle': False,
        'worker_count': 0,
    }
    for key, expected_value in expected.items():
        if protocol.get(key) != expected_value:
            raise CalibrationContractError(
                f'calibration protocol requires {key}={expected_value!r}')
    if not isinstance(protocol.get('sample_count'), int) \
            or protocol['sample_count'] <= 0:
        raise CalibrationContractError('calibration sample_count must be positive')
    if not isinstance(protocol.get('sample_order_sha256'), str) or not re.fullmatch(
            r'[0-9a-f]{64}', protocol['sample_order_sha256']):
        raise CalibrationContractError('calibration sample order hash is invalid')
    hooks = value['hooks']
    if not isinstance(hooks, Mapping) or not isinstance(
            hooks.get('records'), Mapping) or not hooks['records']:
        raise CalibrationContractError('calibration hook records are missing')
    required_records = hooks.get('required_records')
    if not isinstance(required_records, list) or not required_records \
            or any(not isinstance(name, str) or not name
                   for name in required_records):
        raise CalibrationContractError(
            'calibration required hook records are invalid')
    required_set = set(required_records)
    if len(required_records) != len(required_set):
        raise CalibrationContractError(
            'calibration required hook records are duplicated')
    missing_records = tuple(
        name for name in required_records if name not in hooks['records'])
    if missing_records:
        raise CalibrationContractError(
            f'calibration required hook records are missing: {missing_records}')
    unexpected_records = tuple(
        name for name in hooks['records'] if name not in required_set)
    if unexpected_records:
        raise CalibrationContractError(
            f'calibration hook records are unexpected: {unexpected_records}')
    if not isinstance(hooks.get('unsupported_internals'), list):
        raise CalibrationContractError(
            'calibration unsupported internals must be explicit')
    return value
