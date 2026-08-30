"""Production calibration contracts for the full S-V1 Route 3 baseline."""

from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import math
import stat
import threading
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


@dataclass(frozen=True)
class _DatasetMemoEntry:
    authority: Mapping[str, Any]
    seal: tuple[Any, ...]


_DATASET_MEMO_MAX = 8
_DATASET_MEMO: OrderedDict[tuple[Any, ...], _DatasetMemoEntry] = OrderedDict()
_DATASET_MEMO_LOCK = threading.RLock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, relative: Path, label: str) -> Path:
    if relative.is_absolute():
        raise CalibrationContractError(f'{label} must be repository-relative')
    if any(part in {'.', '..'} for part in relative.parts):
        raise CalibrationContractError(f'{label} path is unsafe')
    lexical = root / relative
    cursor = root
    for index, part in enumerate(relative.parts):
        cursor = cursor / part
        if cursor.is_symlink() and not _approved_asset_link(
                root, relative, cursor, index):
            raise CalibrationContractError(
                f'{label} path uses an unapproved symlink')
    path = lexical.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        from .evaluation import MetricError, resolve_project_asset_root
        try:
            asset_root = resolve_project_asset_root(root)
        except MetricError as error:
            raise CalibrationContractError(
                f'{label} escapes approved shared assets') from error
        if relative.parts[:1] == ('data',):
            expected = asset_root / relative
        elif relative.parts[:2] == ('work_dirs', 'reproduction'):
            expected = asset_root / relative
        else:
            raise CalibrationContractError(
                f'{label} escapes repository root')
        try:
            if path != expected.resolve(strict=True):
                raise CalibrationContractError(
                    f'{label} does not use the approved shared asset root')
        except OSError as error:
            raise CalibrationContractError(f'{label} is missing') from error
    if not path.is_file() and label != 'image prefix':
        raise CalibrationContractError(f'{label} is missing: {relative}')
    if label == 'image prefix' and not path.is_dir():
        raise CalibrationContractError(f'{label} is missing: {relative}')
    return path


def _approved_asset_link(
        root: Path, relative: Path, link: Path, index: int) -> bool:
    prefix = tuple(relative.parts[:index + 1])
    if prefix not in {('data',), ('work_dirs', 'reproduction')}:
        return False
    from .evaluation import MetricError, resolve_project_asset_root
    try:
        asset_root = resolve_project_asset_root(root)
        expected = asset_root.joinpath(*prefix)
        raw_target = Path(os.readlink(link))
        lexical_target = (
            raw_target if raw_target.is_absolute() else link.parent / raw_target)
        return (
            lexical_target.absolute() == expected.absolute()
            and link.resolve(strict=True) == expected.resolve(strict=True))
    except (MetricError, OSError):
        return False


def _verified_archive(
        root: Path, inventory: Mapping[str, Any], asset_id: str
        ) -> tuple[Path, str, str]:
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
    return archive, actual, relative


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


def _dataset_stat_stamp(path: Path) -> tuple[Any, ...]:
    try:
        value = path.lstat()
    except OSError as error:
        return ('error', type(error).__name__, getattr(error, 'errno', None))
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns)


def _dataset_component_stamp(path: Path) -> tuple[Any, ...]:
    stamp = _dataset_stat_stamp(path)
    if len(stamp) == 6 and stat.S_ISDIR(stamp[2]):
        return stamp[:3]
    return stamp


def _dataset_tree_seal(
        path: Path, *, recursive: bool = False) -> tuple[Any, ...]:
    digest = hashlib.sha256()
    count = 0

    def record(value: tuple[Any, ...]) -> None:
        nonlocal count
        encoded = repr(value).encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'big'))
        digest.update(encoded)
        count += 1

    pending = [(path, Path())]
    while pending:
        directory, prefix = pending.pop()
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as error:
            record((
                prefix.as_posix(), 'error', type(error).__name__,
                getattr(error, 'errno', None)))
            continue
        for entry in entries:
            relative = prefix / entry.name
            record((relative.as_posix(), _dataset_stat_stamp(entry)))
            if recursive and entry.is_dir() and not entry.is_symlink():
                pending.append((entry, relative))
    return (count, digest.hexdigest())


def _dataset_relative_path_seal(
        root: Path, relative: Path) -> tuple[Any, ...]:
    if (relative.is_absolute() or not relative.parts
            or any(part in {'', '.', '..'} for part in relative.parts)):
        return (str(relative), ('unsafe',))
    current = root
    components = []
    for part in relative.parts:
        current = current / part
        components.append((part, _dataset_component_stamp(current)))
    try:
        resolved = current.resolve(strict=True)
    except OSError as error:
        target = ('error', type(error).__name__, getattr(error, 'errno', None))
    else:
        target = (str(resolved), _dataset_stat_stamp(resolved))
    return (relative.as_posix(), tuple(components), target)


def _calibration_archive_relatives(root: Path) -> tuple[Any, ...]:
    try:
        inventory = json.loads(
            (root / 'data/inventory.json').read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        return (('invalid', type(error).__name__),)
    assets = inventory.get('assets') if isinstance(inventory, Mapping) else None
    result = []
    for asset_id in ('coco-train2017', 'coco-annotations'):
        matches = [
            item for item in assets or ()
            if isinstance(item, Mapping) and item.get('id') == asset_id]
        value = matches[0].get('path') if len(matches) == 1 else None
        if not isinstance(value, str):
            result.append((asset_id, ('invalid',)))
        else:
            result.append((
                asset_id,
                _dataset_relative_path_seal(root, Path(value))))
    return tuple(result)


def _dataset_state_seal(
        root: Path, images: Path, annotation: Path) -> tuple[Any, ...]:
    """Cheaply detect input changes; this seal is never data authority."""
    return (
        ('pid', os.getpid()),
        ('inventory', _dataset_relative_path_seal(
            root, Path('data/inventory.json'))),
        ('images-root', _dataset_stat_stamp(images)),
        ('images', _dataset_tree_seal(images)),
        ('annotation', _dataset_stat_stamp(annotation)),
        ('archives', _calibration_archive_relatives(root)),
    )


def _verified_dataset_authority_uncached(
        root: Path, images: Path, annotation: Path) -> dict[str, Any]:
    inventory_path = _inside(root, Path('data/inventory.json'),
                             'dataset inventory')
    try:
        inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationContractError('dataset inventory is not valid JSON') from error
    if not isinstance(inventory, Mapping):
        raise CalibrationContractError('dataset inventory root is invalid')
    train_archive, train_archive_sha256, train_archive_relative = _verified_archive(
        root, inventory, 'coco-train2017')
    annotation_archive, annotation_archive_sha256, annotation_archive_relative = _verified_archive(
        root, inventory, 'coco-annotations')

    image_entries = tuple(images.iterdir())
    if any(path.is_symlink() for path in image_entries):
        raise CalibrationContractError(
            'extracted train2017 images must not use symlinks')
    extracted = {
        path.name: path for path in image_entries
        if path.is_file() and path.suffix.lower() == '.jpg'}
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

    if (_sha256(train_archive) != train_archive_sha256
            or _sha256(annotation_archive) != annotation_archive_sha256):
        raise CalibrationContractError(
            'official archive content changed during dataset verification')

    return {
        'inventory': 'data/inventory.json',
        'inventory_sha256': _sha256(inventory_path),
        'train_archive': train_archive_relative,
        'train_archive_sha256': train_archive_sha256,
        'image_count': len(extracted),
        'image_content_algorithm': 'sha256-zip-member-bytes-v1',
        'image_content_aggregate_sha256': content_aggregate.hexdigest(),
        'image_order_algorithm': 'sha256-zip-central-directory-order-v1',
        'image_order_sha256': order_aggregate.hexdigest(),
        'annotation_archive': annotation_archive_relative,
        'annotation_archive_sha256': annotation_archive_sha256,
        'annotation_member': annotation_member_name,
        'annotation_member_sha256': annotation_member_sha256,
    }


def _clone_dataset_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    """Isolate every memo consumer from retained dataset authority."""
    return copy.deepcopy(dict(value))


def _verified_dataset_authority(
        root: Path, images: Path, annotation: Path) -> dict[str, Any]:
    key = (os.getpid(), root, images, annotation)
    with _DATASET_MEMO_LOCK:
        before = _dataset_state_seal(root, images, annotation)
        entry = _DATASET_MEMO.get(key)
        if entry is not None and entry.seal == before:
            value = _clone_dataset_authority(entry.authority)
            after = _dataset_state_seal(root, images, annotation)
            if after == before:
                _DATASET_MEMO.move_to_end(key)
                return value
            _DATASET_MEMO.pop(key, None)
            before = after
        value = _verified_dataset_authority_uncached(
            root, images, annotation)
        after = _dataset_state_seal(root, images, annotation)
        if after != before:
            raise CalibrationContractError(
                'dataset inputs changed during authority verification')
        retained = _clone_dataset_authority(value)
        _DATASET_MEMO[key] = _DatasetMemoEntry(retained, after)
        _DATASET_MEMO.move_to_end(key)
        while len(_DATASET_MEMO) > _DATASET_MEMO_MAX:
            _DATASET_MEMO.popitem(last=False)
        return _clone_dataset_authority(value)


def calibration_identity(
        *, repository_root: Path, candidate_id: str, config: Path,
        checkpoint: Path, expected_checkpoint_sha256: str, policy: Path,
        split: str, annotation: Path, image_prefix: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    expected_configs = {
        'full-s-v1': 'configs/reproduction/coco_s_v1.py',
        'no-pif-s-v1': (
            'configs/reproduction/ablations/coco_s_v1_no_pif.py'),
    }
    if candidate_id not in expected_configs:
        raise CalibrationContractError(
            'calibration source must be an admitted S-V1 parent')
    if config.as_posix() != expected_configs[candidate_id]:
        raise CalibrationContractError(
            'calibration config must match its admitted S-V1 parent')
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
    base_required = {'schema_version', 'candidate_id', 'stage', 'source',
                     'identity', 'protocol', 'hooks'}
    if not isinstance(value, Mapping):
        raise CalibrationContractError(
            'calibration artifact fields do not match a supported schema')
    schema_version = value.get('schema_version')
    if (isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version not in (1, 2, 3)):
        raise CalibrationContractError(
            'calibration artifact schema version is unsupported')
    required = base_required | ({'pwl_fit'} if schema_version == 3 else set())
    if set(value) != required:
        raise CalibrationContractError(
            'calibration artifact fields do not match a supported schema')
    if value['stage'] != 'calibrate' \
            or not isinstance(value['candidate_id'], str) \
            or not value['candidate_id']:
        raise CalibrationContractError('calibration artifact identity is invalid')
    if (expected_candidate_id is not None
            and value['candidate_id'] != expected_candidate_id):
        raise CalibrationContractError(
            'calibration artifact target candidate mismatch')
    identity = value['identity']
    identity_fields = {
        'candidate_id', 'config', 'config_sha256', 'checkpoint',
        'checkpoint_sha256', 'policy', 'policy_sha256', 'split', 'dataset',
        'git_commit'}
    if (not isinstance(identity, Mapping) or set(identity) != identity_fields
            or identity.get('split') != 'train2017'
            or (identity.get('candidate_id'), identity.get('config')) not in {
                ('full-s-v1', 'configs/reproduction/coco_s_v1.py'),
                ('no-pif-s-v1',
                 'configs/reproduction/ablations/coco_s_v1_no_pif.py')}
            or not isinstance(identity.get('git_commit'), str)
            or not re.fullmatch(r'[0-9a-f]{40}', identity['git_commit'])
            or any(not isinstance(identity.get(field), str)
                   or not re.fullmatch(r'[0-9a-f]{64}', identity[field])
                   for field in (
                       'config_sha256', 'checkpoint_sha256',
                       'policy_sha256'))):
        raise CalibrationContractError('calibration identity must bind train2017')
    dataset = identity['dataset']
    dataset_fields = {
        'annotation', 'annotation_sha256', 'image_prefix', 'inventory',
        'inventory_sha256', 'train_archive', 'train_archive_sha256',
        'image_count', 'image_content_algorithm',
        'image_content_aggregate_sha256', 'image_order_algorithm',
        'image_order_sha256', 'annotation_archive',
        'annotation_archive_sha256', 'annotation_member',
        'annotation_member_sha256'}
    if (not isinstance(dataset, Mapping) or set(dataset) != dataset_fields
            or dataset.get('annotation') !=
            'data/coco/annotations/person_keypoints_train2017.json'
            or dataset.get('image_prefix') != 'data/coco/train2017'
            or dataset.get('inventory') != 'data/inventory.json'
            or dataset.get('image_count') != 118287
            or dataset.get('image_content_algorithm') !=
            'sha256-zip-member-bytes-v1'
            or dataset.get('image_order_algorithm') !=
            'sha256-zip-central-directory-order-v1'
            or dataset.get('annotation_member') !=
            'annotations/person_keypoints_train2017.json'
            or any(not isinstance(dataset.get(field), str)
                   or not re.fullmatch(r'[0-9a-f]{64}', dataset[field])
                   for field in (
                       'annotation_sha256', 'inventory_sha256',
                       'train_archive_sha256',
                       'image_content_aggregate_sha256',
                       'image_order_sha256', 'annotation_archive_sha256',
                       'annotation_member_sha256'))):
        raise CalibrationContractError(
            'calibration dataset authority is incomplete')
    protocol = value['protocol']
    legacy_protocol_fields = {
        'model_mode', 'grad_enabled', 'shuffle', 'worker_count',
        'sample_count', 'sample_order_sha256'}
    deterministic_protocol_fields = (
        legacy_protocol_fields | {'root_determinism'})
    protocol_fields = set(protocol) if isinstance(protocol, Mapping) else set()
    if not isinstance(protocol, Mapping):
        raise CalibrationContractError('calibration protocol is invalid')
    if schema_version == 1 and protocol_fields != legacy_protocol_fields:
        raise CalibrationContractError(
            'calibration schema v1 requires the exact legacy protocol')
    if schema_version in (2, 3) \
            and protocol_fields != deterministic_protocol_fields:
        raise CalibrationContractError(
            f'calibration schema v{schema_version} requires root '
            'determinism protocol')
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
    if schema_version in (2, 3):
        root_determinism = protocol['root_determinism']
        seed_fields = {
            'seed', 'python_seed', 'numpy_seed', 'torch_seed',
            'torch_cuda_seed'}
        expected_root_fields = seed_fields | {
            'torch_deterministic_algorithms', 'cudnn_benchmark',
            'cudnn_deterministic'}
        seed = root_determinism.get('seed') \
            if isinstance(root_determinism, Mapping) else None
        if (not isinstance(root_determinism, Mapping)
                or set(root_determinism) != expected_root_fields
                or isinstance(seed, bool) or not isinstance(seed, int)
                or not 0 <= seed < 2**32
                or any(root_determinism.get(field) != seed
                       for field in seed_fields)
                or root_determinism.get(
                    'torch_deterministic_algorithms') is not True
                or root_determinism.get('cudnn_benchmark') is not False
                or root_determinism.get('cudnn_deterministic') is not True):
            raise CalibrationContractError(
                'calibration root determinism contract is invalid')
    hooks = value['hooks']
    if (not isinstance(hooks, Mapping)
            or set(hooks) not in (
                {'records', 'required_records', 'unsupported_internals'},
                {'records', 'required_records', 'unsupported_internals',
                 'activation_scales'})
            or not isinstance(hooks.get('records'), Mapping)
            or not hooks['records']):
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
    record_fields = {
        'granularity', 'sample_count', 'zero_count', 'underflow_count',
        'overflow_count', 'max_abs', 'range', 'percentiles', 'algorithm',
        'histogram_bins', 'histogram_domain', 'percentile_bound_valid',
        'relative_error_bound', 'outlier_ratio_above_p99_bin', 'token_ids',
        'observed_shape'}
    for name, record in hooks['records'].items():
        numeric = record.get('range') if isinstance(record, Mapping) else None
        if (not isinstance(record, Mapping) or set(record) != record_fields
                or record.get('granularity') not in {
                    'tensor', 'channel', 'token'}
                or not isinstance(record.get('sample_count'), int)
                or isinstance(record.get('sample_count'), bool)
                or record['sample_count'] <= 0
                or any(not isinstance(record.get(field), int)
                       or isinstance(record.get(field), bool)
                       or record[field] < 0 for field in (
                           'zero_count', 'underflow_count', 'overflow_count'))
                or not isinstance(numeric, list) or len(numeric) != 2
                or any(not isinstance(item, (int, float))
                       or isinstance(item, bool) or not math.isfinite(item)
                       for item in numeric)
                or record.get('algorithm') != 'fixed-log2-histogram-v1'
                or record.get('histogram_bins') != 256
                or record.get('histogram_domain') != [2 ** -32, 2 ** 32]
                or not isinstance(record.get('percentile_bound_valid'), bool)):
            raise CalibrationContractError(
                f'calibration hook record schema is invalid: {name}')
        bounded = record['percentile_bound_valid']
        if bounded != (
                record['underflow_count'] == 0 and record['overflow_count'] == 0):
            raise CalibrationContractError(
                f'calibration hook percentile bound is inconsistent: {name}')
        if (numeric[0] > numeric[1]
                or record['zero_count'] + record['underflow_count']
                + record['overflow_count'] > record['sample_count']):
            raise CalibrationContractError(
                f'calibration hook count/range is inconsistent: {name}')
        max_abs = record.get('max_abs')
        flat_max = _finite_values(max_abs)
        observed_shape = record.get('observed_shape')
        if (not flat_max or any(item < 0 for item in flat_max)
                or not isinstance(observed_shape, list)):
            raise CalibrationContractError(
                f'calibration hook max_abs/shape is invalid: {name}')
        if record['granularity'] == 'tensor' and (
                observed_shape != [] or len(flat_max) != 1
                or not math.isclose(
                    flat_max[0], max(abs(numeric[0]), abs(numeric[1])),
                    rel_tol=1e-6, abs_tol=0.0)):
            raise CalibrationContractError(
                f'calibration tensor range is inconsistent: {name}')
        percentiles = record.get('percentiles')
        expected_percentiles = {'0.5', '0.9', '0.99', '0.999'}
        if not isinstance(percentiles, Mapping) \
                or set(percentiles) != expected_percentiles:
            raise CalibrationContractError(
                f'calibration percentiles are invalid: {name}')
        percentile_values = tuple(percentiles[key]
                                  for key in ('0.5', '0.9', '0.99', '0.999'))
        relative_bound = record.get('relative_error_bound')
        outlier = record.get('outlier_ratio_above_p99_bin')
        if bounded:
            if (any(not isinstance(item, (int, float))
                    or isinstance(item, bool) or not math.isfinite(item)
                    for item in percentile_values)
                    or list(percentile_values) != sorted(percentile_values)
                    or not math.isclose(
                        float(relative_bound), 2 ** 0.25 - 1,
                        rel_tol=1e-12, abs_tol=0.0)
                    or not isinstance(outlier, (int, float))
                    or isinstance(outlier, bool) or not 0 <= outlier <= 1):
                raise CalibrationContractError(
                    f'calibration bounded statistics are invalid: {name}')
        elif (any(item is not None for item in percentile_values)
              or relative_bound is not None or outlier is not None):
            raise CalibrationContractError(
                f'calibration unbounded statistics must be invalidated: {name}')
    if not isinstance(hooks.get('unsupported_internals'), list):
        raise CalibrationContractError(
            'calibration unsupported internals must be explicit')
    if 'activation_scales' in hooks:
        scales = hooks['activation_scales']
        if not isinstance(scales, Mapping):
            raise CalibrationContractError(
                'calibration activation scales must be a role mapping')
        for role, record in scales.items():
            if (not isinstance(role, str) or not role
                    or not isinstance(record, Mapping)
                    or set(record) != {
                        'source_record', 'granularity', 'scale'}
                    or record.get('granularity') not in {'tensor', 'channel'}
                    or record.get('source_record') not in hooks['records']):
                raise CalibrationContractError(
                    f'calibration activation scale is invalid for {role!r}')
            validate_activation_scale_record(
                role, record, hooks['records'][record['source_record']])
    if schema_version == 3:
        try:
            from .pwl_artifacts import validate_pwl_fit_report
            validate_pwl_fit_report(
                value['pwl_fit'],
                expected_candidate_id=value['candidate_id'])
        except ValueError as error:
            raise CalibrationContractError(
                f'PWL fit artifact is invalid: {error}') from error
    return value


def validate_calibration_provenance(
        value: Mapping[str, Any], *, expected_candidate,
        repository_root: Path, manifest_path: Path) -> Mapping[str, Any]:
    """Reconstruct the manifest/source/dataset contract for production use."""
    validate_calibration_artifact(
        value, expected_candidate_id=expected_candidate.id)
    from .numeric_source import validate_numeric_source_binding
    try:
        source = validate_numeric_source_binding(
            value['source'], repository_root=repository_root,
            candidate=expected_candidate, manifest_path=manifest_path)
    except ValueError as error:
        raise CalibrationContractError(
            f'calibration source binding is invalid: {error}') from error
    expected_policy_path = expected_candidate.config.as_posix()
    if source['policy_path'] != expected_policy_path:
        raise CalibrationContractError(
            'calibration source policy must equal target candidate config')
    identity = value['identity']
    if (value['schema_version'] in (2, 3)
            and value['protocol']['root_determinism']['seed']
            != expected_candidate.seed):
        raise CalibrationContractError(
            'calibration root determinism seed disagrees with candidate')
    if identity['git_commit'] != source['git_commit']:
        raise CalibrationContractError(
            'calibration identity commit disagrees with source binding')
    from .checkpoints import authorize_tracked_config
    config = authorize_tracked_config(
        repository_root, manifest_path, expected_candidate).load_config()
    numeric = config.get('numeric_optimization', {})
    calibration_policy = numeric.get('calibration', {}) \
        if isinstance(numeric, Mapping) else None
    expected_schema_version = calibration_policy.get(
        'artifact_schema_version', 1) \
        if isinstance(calibration_policy, Mapping) else None
    if (isinstance(expected_schema_version, bool)
            or not isinstance(expected_schema_version, int)
            or expected_schema_version not in (1, 2, 3)):
        raise CalibrationContractError(
            'calibration source policy schema version is invalid')
    if value['schema_version'] != expected_schema_version:
        raise CalibrationContractError(
            'calibration artifact version disagrees with source policy')
    if (identity['policy'] != source['policy_path']
            or identity['policy_sha256'] != source['policy_sha256']):
        raise CalibrationContractError(
            'calibration policy identity disagrees with source binding')
    try:
        authority = json.loads((
            repository_root / source['authority_path']).read_text(
                encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationContractError(
            'calibration train authority is unreadable') from error
    authority_fields = {
        'schema_version', 'dataset', 'split', 'image_count', 'image_prefix',
        'annotation_path', 'annotation_archive_member',
        'image_archive_asset_id', 'image_archive_sha256',
        'annotation_archive_asset_id', 'annotation_archive_sha256'}
    dataset = identity['dataset']
    if (not isinstance(authority, Mapping) or set(authority) != authority_fields
            or authority.get('schema_version') != 1
            or authority.get('dataset') != 'coco'
            or authority.get('split') != 'train2017'
            or authority.get('image_count') != dataset['image_count']
            or authority.get('image_prefix') != dataset['image_prefix']
            or authority.get('annotation_path') != dataset['annotation']
            or authority.get('annotation_archive_member') !=
            dataset['annotation_member']
            or authority.get('image_archive_asset_id') != 'coco-train2017'
            or authority.get('image_archive_sha256') !=
            dataset['train_archive_sha256']
            or authority.get('annotation_archive_asset_id') !=
            'coco-annotations'
            or authority.get('annotation_archive_sha256') !=
            dataset['annotation_archive_sha256']):
        raise CalibrationContractError(
            'calibration identity disagrees with tracked train authority')
    from .schema import load_candidate_manifest
    source_candidate_id = expected_candidate.features.get(
        'calibration_source_candidate', 'full-s-v1')
    baselines = tuple(
        item for item in load_candidate_manifest(manifest_path)
        if item.id == source_candidate_id)
    if len(baselines) != 1:
        raise CalibrationContractError(
            'calibration requires exactly one manifest source parent')
    baseline = baselines[0]
    expected_identity = calibration_identity(
        repository_root=repository_root, candidate_id=baseline.id,
        config=baseline.config, checkpoint=baseline.checkpoint,
        expected_checkpoint_sha256=baseline.checkpoint_sha256,
        policy=Path(source['policy_path']), split='train2017',
        annotation=Path(
            'data/coco/annotations/person_keypoints_train2017.json'),
        image_prefix=Path('data/coco/train2017'))
    expected_identity['git_commit'] = source['git_commit']
    if identity != expected_identity:
        raise CalibrationContractError(
            'calibration identity disagrees with canonical production inputs')
    kind = expected_candidate.features.get('numeric_kind')
    if kind in {'pwl', 'pwl-combined'}:
        if value['schema_version'] != 3:
            raise CalibrationContractError(
                'PWL calibration requires measured fit schema v3')
        try:
            from .pwl_artifacts import validate_pwl_fit_report
            pwl_policy = numeric.get('pwl')
            if not isinstance(pwl_policy, Mapping):
                raise CalibrationContractError('PWL calibration policy is missing')
            validate_pwl_fit_report(
                value['pwl_fit'], expected_candidate_id=expected_candidate.id,
                expected_policy={
                    name: pwl_policy[name] for name in (
                        'enabled_function', 'source', 'roles', 'domain',
                        'segments', 'grid_points', 'saturation', 'qat_form',
                        'selection_policy')})
        except ValueError as error:
            raise CalibrationContractError(
                f'PWL fit policy binding is invalid: {error}') from error
    elif value['schema_version'] == 3:
        raise CalibrationContractError(
            'PWL calibration schema is forbidden for non-PWL candidate')
    policy = numeric.get('quant_policy', {})
    observers = policy.get('activation_observers', {})
    if kind == 'w8a8':
        scales = value['hooks'].get('activation_scales')
        if (not isinstance(observers, Mapping) or not observers
                or not isinstance(scales, Mapping)
                or set(scales) != set(observers)
                or any(scales[role]['source_record'] != source_record
                       for role, source_record in observers.items())):
            raise CalibrationContractError(
                'W8A8 calibration scales do not exactly cover policy roles')
    elif value['hooks'].get('activation_scales'):
        raise CalibrationContractError(
            'non-W8A8 calibration must not publish activation scales')
    return value


def _finite_values(value: object) -> list[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool) \
            and math.isfinite(value):
        return [float(value)]
    if isinstance(value, list):
        result: list[float] = []
        for item in value:
            child = _finite_values(item)
            if not child:
                return []
            result.extend(child)
        return result
    return []


def validate_activation_scale_record(
        role: str, value: Mapping[str, Any],
        source_record: Mapping[str, Any]) -> None:
    if source_record.get('overflow_count') != 0:
        raise CalibrationContractError(
            f'activation scale source has histogram overflow for {role}')
    granularity = value['granularity']
    if granularity != source_record.get('granularity'):
        raise CalibrationContractError(
            f'activation scale granularity disagrees for {role}')
    scale = value['scale']
    if granularity == 'tensor':
        numeric_range = source_record['range']
        maximum = max(abs(float(item)) for item in numeric_range)
        expected = maximum / 127.0
        if (maximum <= 0
                or not isinstance(scale, (int, float))
                or isinstance(scale, bool)
                or not math.isfinite(scale) or scale <= 0
                or not math.isclose(scale, expected, rel_tol=1e-12,
                                    abs_tol=0.0)):
            raise CalibrationContractError(
                f'activation scale requires a positive measured maximum and '
                f'must be derived from measured range for {role}')
        return
    maximum = source_record.get('max_abs')
    shape = source_record.get('observed_shape')
    if (not isinstance(scale, list) or not isinstance(maximum, list)
            or not maximum or shape != [len(maximum)]
            or len(scale) != len(maximum)):
        raise CalibrationContractError(
            f'activation scale width disagrees for {role}')
    expected = [float(item) / 127.0 if float(item) else 1.0
                for item in maximum]
    if any(not isinstance(item, (int, float)) or isinstance(item, bool)
           or not math.isfinite(item) or item <= 0
           or not math.isclose(item, wanted, rel_tol=1e-12, abs_tol=0.0)
           for item, wanted in zip(scale, expected)):
        raise CalibrationContractError(
            f'activation scale is not derived from measured channels for {role}')
