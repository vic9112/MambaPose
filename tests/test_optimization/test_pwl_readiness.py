import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import zipfile

import pytest
import torch
import torch.nn.functional as F


def test_pwl_transitive_validators_do_not_parse_mutable_config_paths():
    from mambapose_opt.numeric_calibration import (
        validate_calibration_provenance)
    from mambapose_opt.numeric_runtime import (
        validate_numeric_convert_artifact)
    from mambapose_opt.pwl_selection import _canonical_envelopes

    for function in (
            validate_calibration_provenance,
            validate_numeric_convert_artifact,
            _canonical_envelopes):
        assert 'Config.fromfile' not in inspect.getsource(function)


def _policy(function_name='silu'):
    source = 'module' if function_name in {'silu', 'gelu'} \
        else 'ss2d-transition'
    return {
        'enabled_function': function_name,
        'source': source,
        'roles': ('block.act',),
        'domain': (-2.0, 2.0),
        'segments': 4,
        'grid_points': 129,
        'saturation': ('clamp' if function_name == 'exp'
                       else 'continuous-asymptotic-tail-v1'),
        'qat_form': 'differentiable',
        'selection_policy': 'observed-range-max-then-mean-v1',
    }


def _fit(function_name='silu', values=None, *, domain=None, segments=None,
         grid_points=None):
    from mambapose_opt.pwl_artifacts import fit_pwl_observations

    policy = _policy(function_name)
    if domain is not None:
        policy['domain'] = domain
    if segments is not None:
        policy['segments'] = segments
    if grid_points is not None:
        policy['grid_points'] = grid_points
    return fit_pwl_observations(
        candidate_id=f'pwl-{function_name}-s-v1',
        policy=policy,
        observations={
            'block.act': [
                torch.tensor(
                    values if values is not None else [-1.0, 0.0, 1.0],
                    dtype=torch.float64)
            ]
        })


def _write_reference(root: Path, artifact: dict, relative: str):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'pwl_fit': artifact}), encoding='utf-8')
    return path, {
        'path': relative,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _selection_calibrations():
    common_source = {
        'git_commit': '1' * 40,
        'manifest_path': 'optimization/candidates.json',
        'manifest_sha256': '2' * 64,
        'checkpoint_path': 'work_dirs/reproduction/full.pth',
        'checkpoint_sha256': '3' * 64,
        'authority_path': 'optimization/coco_train2017_authority.json',
        'authority_sha256': '4' * 64,
    }
    common_identity = {
        'config': 'configs/reproduction/coco_s_v1.py',
        'config_sha256': '5' * 64,
        'checkpoint': 'work_dirs/reproduction/full.pth',
        'checkpoint_sha256': '3' * 64,
        'dataset': {'authority': 'same-train2017'},
        'git_commit': '1' * 40,
    }
    common_protocol = {
        'model_mode': 'eval', 'grad_enabled': False, 'shuffle': False,
        'worker_count': 0, 'sample_count': 512,
        'sample_order_sha256': '6' * 64,
        'root_determinism': {'seed': 0},
    }
    result = {}
    checksum_digits = {'silu': 'a', 'gelu': 'b', 'softplus': 'c', 'exp': 'd'}
    for function in ('silu', 'gelu', 'softplus', 'exp'):
        candidate_id = f'pwl-{function}-s-v1'
        policy = _policy(function)
        fit = _fit(function, values=[-1.0, 0.0, 1.0])
        result[candidate_id] = {
            'reference': {
                'path': (f'work_dirs/optimization/{candidate_id}/0/'
                         'calibrate/calibrate.json'),
                'sha256': checksum_digits[function] * 64,
            },
            'policy': policy,
            'calibration': {
                'schema_version': 3, 'candidate_id': candidate_id,
                'stage': 'calibrate',
                'source': {**common_source,
                           'candidate_id': candidate_id,
                           'candidate_row_sha256': '7' * 64,
                           'config_path': f'configs/{function}.py',
                           'config_sha256': function[-1] * 64,
                           'policy_path': f'configs/{function}.py',
                           'policy_sha256': function[-1] * 64},
                'identity': {**common_identity,
                             'candidate_id': 'full-s-v1',
                             'policy': f'configs/{function}.py',
                             'policy_sha256': function[-1] * 64,
                             'split': 'train2017'},
                'protocol': dict(common_protocol),
                'hooks': {}, 'pwl_fit': fit,
            },
        }
    return result


def test_pwl_fit_records_exact_role_ranges_errors_and_saturation():
    fit = _fit(values=[-3.0, -1.0, 0.0, 2.5])

    assert fit['candidate_id'] == 'pwl-silu-s-v1'
    assert fit['input_roles'] == [{
        'operation_role': 'block.act',
        'exact_input_role': 'block.act.input',
    }]
    assert fit['observed_range'] == [-3.0, 2.5]
    assert fit['in_domain_error']['samples'] == 129
    assert fit['observed_range_error']['samples'] == 129
    assert fit['in_domain_error']['max'] >= fit['in_domain_error']['mean'] >= 0
    assert fit['observed_range_error']['max'] >= (
        fit['observed_range_error']['mean']) >= 0
    assert fit['observed_samples_error']['samples'] == 4
    assert fit['schema_version'] == 2
    assert fit['domain_coverage'] == {
        'below': 1, 'above': 1, 'total': 4, 'ratio': 0.5,
        'handling': 'continuous-asymptotic-tail-v1'}
    role = fit['role_observations'][0]
    assert role['operation_role'] == 'block.act'
    assert role['exact_input_role'] == 'block.act.input'
    assert role['observed_range'] == [-3.0, 2.5]
    assert role['domain_coverage']['ratio'] == 0.5
    assert role['domain_coverage']['handling'] == (
        'continuous-asymptotic-tail-v1')


def test_pwl_fit_records_validator_recomputable_exact_role_tail_percentiles():
    from mambapose_opt.pwl_artifacts import validate_pwl_fit_report

    fit = _fit(values=[0.0, 0.25, 0.5, 1.0, 2.0])
    tail = fit['role_observations'][0]['tail_statistics']

    assert tail['algorithm'] == 'fixed-log2-absolute-histogram-v1'
    assert set(tail['absolute_percentiles']) == {'0.9', '0.99', '0.999'}
    assert len(tail['histogram']) == tail['histogram_bins'] == 256
    assert sum(tail['histogram']) + tail['zero_count'] == tail['sample_count']
    assert list(tail['absolute_percentiles'].values()) == sorted(
        tail['absolute_percentiles'].values())
    assert validate_pwl_fit_report(fit, expected_policy=_policy()) == fit

    forged = json.loads(json.dumps(fit))
    forged['role_observations'][0]['tail_statistics'][
        'absolute_percentiles']['0.99'] *= 0.5
    with pytest.raises(ValueError, match='tail'):
        validate_pwl_fit_report(forged, expected_policy=_policy())


def test_pwl_artifact_ranking_is_measured_not_function_name_order():
    from mambapose_opt.pwl_artifacts import rank_pwl_fit_artifacts

    silu = _fit('silu', values=[-2.0, -1.5])
    gelu = _fit('gelu', values=[-2.0, -1.5])

    assert [item['candidate_id'] for item in rank_pwl_fit_artifacts(
            [silu, gelu])] == ['pwl-gelu-s-v1', 'pwl-silu-s-v1']


def test_pwl_fit_rejects_forged_role_range_and_recomputed_error():
    from mambapose_opt.pwl_artifacts import (
        PWLArtifactError, validate_pwl_fit_report)

    fit = _fit()
    invalid_range = json.loads(json.dumps(fit))
    invalid_range['role_observations'][0]['observed_range'][0] = float('nan')
    with pytest.raises(PWLArtifactError, match='role observation'):
        validate_pwl_fit_report(invalid_range)

    forged_error = json.loads(json.dumps(fit))
    forged_error['observed_range_error']['max'] += 1.0
    with pytest.raises(PWLArtifactError, match='recomputed'):
        validate_pwl_fit_report(forged_error)


def test_pwl_fit_rejects_compound_forged_coefficients_and_all_metrics():
    """Break caught: self-consistent noncanonical coefficients must not pass."""
    from mambapose_opt.pwl_artifacts import (
        PWLArtifactError, validate_pwl_fit_report)
    from mmpose.models.utils.hardware_friendly.pwl import fit_pwl

    values = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    fit = _fit(values=values.tolist())
    forged_approximation = fit_pwl(
        F.gelu, (-2.0, 2.0), 4, 129, function_name='silu')
    forged = json.loads(json.dumps(fit))
    forged['coefficients'] = {
        'breakpoints': forged_approximation.breakpoints.tolist(),
        'slopes': forged_approximation.slopes.tolist(),
        'intercepts': forged_approximation.intercepts.tolist(),
    }

    def metric(bounds, samples):
        grid = torch.linspace(*bounds, samples, dtype=torch.float64)
        error = (forged_approximation(grid) - F.silu(grid)).abs()
        return {
            'max': float(error.max()), 'mean': float(error.mean()),
            'samples': int(error.numel()),
        }

    observed_error = (forged_approximation(values) - F.silu(values)).abs()
    sample_metric = {
        'max': float(observed_error.max()),
        'mean': float(observed_error.mean()), 'samples': 3,
    }
    forged['in_domain_error'] = metric((-2.0, 2.0), 129)
    forged['observed_range_error'] = metric((-1.0, 1.0), 129)
    forged['observed_samples_error'] = sample_metric
    forged['role_observations'][0]['observed_range_error'] = metric(
        (-1.0, 1.0), 129)
    forged['role_observations'][0]['observed_samples_error'] = sample_metric

    with pytest.raises(PWLArtifactError, match='canonical'):
        validate_pwl_fit_report(
            forged, expected_candidate_id='pwl-silu-s-v1',
            expected_policy=_policy())


def test_tail_covered_out_of_domain_fit_is_admitted_with_full_range_error():
    """Break caught: tail-covered samples must not be classified as clamps."""
    from mambapose_opt.pwl_artifacts import (
        build_pwl_installation_manifest,
        rank_pwl_fit_artifacts, validate_pwl_fit_report)

    admitted = _fit(
        values=[-37.16, -6.0, 0.0, 6.0, 38.29],
        domain=(-6.0, 6.0), segments=16, grid_points=4097)

    assert admitted['admission'] == {'decision': 'passed', 'reasons': []}
    assert admitted['domain_coverage'] == {
        'below': 1, 'above': 1, 'total': 5, 'ratio': 0.4,
        'handling': 'continuous-asymptotic-tail-v1'}
    assert admitted['observed_range_error']['max'] < 0.033
    assert admitted['observed_samples_error']['max'] < 0.016
    assert validate_pwl_fit_report(
        admitted, expected_policy={
            **_policy(), 'domain': (-6.0, 6.0), 'segments': 16,
            'grid_points': 4097}) == admitted
    assert rank_pwl_fit_artifacts([admitted]) == (admitted,)
    installation = build_pwl_installation_manifest(
        candidate_id='pwl-silu-s-v1', fit=admitted,
        fit_reference={'path': 'work_dirs/optimization/fit.json',
                       'sha256': 'a' * 64})
    assert installation['report']['out_of_domain_ratio'] == 0.4


def test_tail_coverage_cannot_be_forged_into_clamp_or_rejection():
    from mambapose_opt.pwl_artifacts import (
        PWLArtifactError, validate_pwl_fit_report)

    forged = _fit(values=[-20.0, 0.0, 20.0])
    forged['domain_coverage']['handling'] = 'clamp'

    with pytest.raises(PWLArtifactError, match='coverage|policy'):
        validate_pwl_fit_report(forged, expected_policy=_policy())

    forged = _fit(values=[-20.0, 0.0, 20.0])
    forged['admission'] = {
        'decision': 'rejected',
        'reasons': ['clamp-count-nonzero', 'observed-range-outside-domain']}
    with pytest.raises(PWLArtifactError, match='admission'):
        validate_pwl_fit_report(forged, expected_policy=_policy())


def test_exp_out_of_domain_fit_retains_clamp_rejection():
    """Break caught: asymptotic tails must never be applied to exp."""
    from mambapose_opt.pwl_artifacts import (
        PWLArtifactError, rank_pwl_fit_artifacts, validate_pwl_fit_report)

    rejected = _fit('exp', values=[-20.0, 0.0, 20.0])

    assert rejected['domain_coverage'] == {
        'below': 1, 'above': 1, 'total': 3, 'ratio': 2 / 3,
        'handling': 'clamp'}
    assert rejected['admission'] == {
        'decision': 'rejected',
        'reasons': ['clamp-count-nonzero', 'observed-range-outside-domain']}
    assert validate_pwl_fit_report(
        rejected, expected_policy=_policy('exp')) == rejected
    with pytest.raises(PWLArtifactError, match='not admitted'):
        rank_pwl_fit_artifacts([rejected])


def test_fit_schema_v1_is_rejected_instead_of_reinterpreted():
    from mambapose_opt.pwl_artifacts import (
        PWLArtifactError, validate_pwl_fit_report)

    legacy = _fit()
    legacy['schema_version'] = 1

    with pytest.raises(PWLArtifactError, match='schema version'):
        validate_pwl_fit_report(legacy, expected_policy=_policy())


def test_selection_schema_v1_is_explicitly_rejected(tmp_path):
    from mambapose_opt.pwl_selection import (
        PWLSelectionError, validate_pwl_selection_artifact)

    with pytest.raises(PWLSelectionError, match='schema version'):
        validate_pwl_selection_artifact(
            {'schema_version': 1}, repository_root=tmp_path,
            manifest_path=tmp_path / 'optimization/candidates.json')


def test_exp_fit_retains_exact_export_time_constant_folding_comparator():
    fit = _fit('exp', values=[-1.0, 0.0, 1.0])

    assert fit['exact_comparator'] == {
        'kind': 'exact-export-time-constant-folding',
        'applicable_source': 'static-parameter',
        'runtime_nonlinear_operations': 0,
        'max_error': 0.0,
        'mean_error': 0.0,
        'preferred_over_pwl_when_exportable': True,
    }


def test_four_candidate_selection_excludes_exportable_exp_and_is_measured():
    """Break caught: exp metadata must control the production decision."""
    from mambapose_opt.pwl_selection import build_pwl_selection_record

    selection = build_pwl_selection_record(
        manifest_reference={
            'path': 'optimization/candidates.json', 'sha256': '2' * 64},
        calibrations=_selection_calibrations())

    assert selection['artifact_kind'] == 'pwl-four-candidate-selection'
    assert [row['candidate_id'] for row in selection['candidates']] == [
        'pwl-silu-s-v1', 'pwl-gelu-s-v1',
        'pwl-softplus-s-v1', 'pwl-exp-s-v1']
    exp = next(row for row in selection['candidates']
               if row['candidate_id'] == 'pwl-exp-s-v1')
    silu = next(row for row in selection['candidates']
                if row['candidate_id'] == 'pwl-silu-s-v1')
    assert selection['schema_version'] == 2
    assert 'clamp' not in silu
    assert silu['domain_coverage']['handling'] == (
        'continuous-asymptotic-tail-v1')
    assert exp['selection_status'] == 'excluded-exact-constant-fold'
    assert exp['domain_coverage']['handling'] == 'clamp'
    assert 'pwl-exp-s-v1' not in selection['ranking']
    assert selection['selected_candidate_id'] == selection['ranking'][0]
    assert selection['selected_candidate_id'] != 'pwl-exp-s-v1'


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        (lambda values: values.pop('pwl-gelu-s-v1'), 'exactly four'),
        (lambda values: values['pwl-gelu-s-v1']['calibration']['source']
         .__setitem__('git_commit', '9' * 40), 'source authority'),
        (lambda values: values['pwl-gelu-s-v1']['calibration']['identity']
         .__setitem__('checkpoint_sha256', '9' * 64), 'source authority'),
        (lambda values: values['pwl-gelu-s-v1']['calibration']['protocol']
         .__setitem__('sample_order_sha256', '9' * 64), 'protocol'),
        (lambda values: values['pwl-gelu-s-v1']['calibration']['protocol']
         .__setitem__('sample_count', 511), 'protocol'),
    ],
)
def test_four_candidate_selection_requires_cross_artifact_authority(
        mutation, message):
    from mambapose_opt.pwl_selection import (
        PWLSelectionError, build_pwl_selection_record)

    values = _selection_calibrations()
    mutation(values)

    with pytest.raises(PWLSelectionError, match=message):
        build_pwl_selection_record(
            manifest_reference={
                'path': 'optimization/candidates.json', 'sha256': '2' * 64},
            calibrations=values)


def test_selection_reference_is_relative_hash_bound_and_rejects_aliases(
        tmp_path):
    from mambapose_opt.pwl_selection import (
        PWLSelectionError, load_pwl_selection_reference)

    from mambapose_opt.pwl_selection import (
        _load_selection_file, build_pwl_selection_record)

    artifact = build_selection = build_pwl_selection_record(
        manifest_reference={
            'path': 'optimization/candidates.json', 'sha256': '2' * 64},
        calibrations=_selection_calibrations())
    path = tmp_path / (
        'work_dirs/optimization/ssm-quant-pwl/pwl-selection/selection.json')
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(artifact), encoding='utf-8')
    reference = {
        'path': path.relative_to(tmp_path).as_posix(),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert _load_selection_file(
        reference, repository_root=tmp_path) == build_selection

    traversal = dict(reference, path=(
        'work_dirs/optimization/ssm-quant-pwl/alias/../'
        'pwl-selection/selection.json'))
    with pytest.raises(PWLSelectionError, match='unsafe'):
        load_pwl_selection_reference(
            traversal, repository_root=tmp_path,
            manifest_path=Path('optimization/candidates.json'))
    alias = path.with_name('selection-alias.json')
    alias.symlink_to(path)
    with pytest.raises(PWLSelectionError, match='symlink'):
        load_pwl_selection_reference(
            {'path': alias.relative_to(tmp_path).as_posix(),
             'sha256': reference['sha256']}, repository_root=tmp_path,
            manifest_path=Path('optimization/candidates.json'))


def test_exp_pwl_cannot_be_installed_even_with_an_admitted_fit():
    from mambapose_opt.numeric_conversion import (
        NumericBindingError, install_pwl_fit)
    from mambapose_opt.pwl_artifacts import PWLInstallationReport

    fit = _fit('exp', values=[-1.0, 0.0, 1.0])
    report = PWLInstallationReport(
        function_name='exp', source='ss2d-transition', roles=('block.act',),
        input_roles=('block.act.transition_exp_input',), domain=(-2.0, 2.0),
        segments=4, in_domain_max_error=fit['in_domain_error']['max'],
        in_domain_mean_error=fit['in_domain_error']['mean'],
        observed_range=(-1.0, 1.0),
        observed_range_max_error=fit['observed_range_error']['max'],
        observed_range_mean_error=fit['observed_range_error']['mean'],
        out_of_domain_ratio=0.0,
        fit_artifact_path='work_dirs/optimization/fit.json',
        fit_artifact_sha256='a' * 64,
        exact_comparator=fit['exact_comparator'])

    with pytest.raises(NumericBindingError, match='constant fold'):
        install_pwl_fit(torch.nn.Module(), fit=fit, expected_report=report)


def test_pwl_fit_reference_is_relative_hash_bound_and_rejects_symlinks(
        tmp_path):
    from mambapose_opt.pwl_artifacts import (
        PWLArtifactError, load_pwl_fit_reference)

    fit = _fit()
    path, reference = _write_reference(
        tmp_path, fit,
        'work_dirs/optimization/pwl-silu-s-v1/calibrate/calibrate.json')

    loaded = load_pwl_fit_reference(
        reference, repository_root=tmp_path,
        expected_candidate_id='pwl-silu-s-v1', expected_policy=_policy())
    assert loaded == fit

    absolute = dict(reference, path=str(path))
    with pytest.raises(PWLArtifactError, match='repository-relative'):
        load_pwl_fit_reference(
            absolute, repository_root=tmp_path,
            expected_candidate_id='pwl-silu-s-v1', expected_policy=_policy())
    traversal = dict(reference, path='work_dirs/optimization/../fit.json')
    with pytest.raises(PWLArtifactError, match='unsafe'):
        load_pwl_fit_reference(
            traversal, repository_root=tmp_path,
            expected_candidate_id='pwl-silu-s-v1', expected_policy=_policy())

    symlink = tmp_path / 'work_dirs/optimization/pwl-link.json'
    symlink.symlink_to(path)
    linked = {
        'path': 'work_dirs/optimization/pwl-link.json',
        'sha256': reference['sha256'],
    }
    with pytest.raises(PWLArtifactError, match='symlink'):
        load_pwl_fit_reference(
            linked, repository_root=tmp_path,
            expected_candidate_id='pwl-silu-s-v1', expected_policy=_policy())


def test_pwl_fit_reference_rejects_alternate_policy_downgrade(tmp_path):
    from mambapose_opt.pwl_artifacts import (
        PWLArtifactError, load_pwl_fit_reference)

    fit = _fit('silu')
    _, reference = _write_reference(
        tmp_path, fit,
        'work_dirs/optimization/pwl-silu-s-v1/calibrate/calibrate.json')

    with pytest.raises(PWLArtifactError, match='policy'):
        load_pwl_fit_reference(
            reference, repository_root=tmp_path,
            expected_candidate_id='pwl-silu-s-v1',
            expected_policy=_policy('gelu'))


def test_serialized_pwl_installation_binds_fit_and_operation_manifest():
    from dataclasses import asdict

    from mambapose_opt.pwl_artifacts import (
        build_pwl_installation_manifest, validate_pwl_installation_manifest)

    fit = _fit()
    fit_reference = {
        'path': ('work_dirs/optimization/pwl-silu-s-v1/'
                 'calibrate/calibrate.json'),
        'sha256': 'a' * 64,
    }
    manifest = build_pwl_installation_manifest(
        candidate_id='pwl-silu-s-v1', fit=fit,
        fit_reference=fit_reference)
    validated = validate_pwl_installation_manifest(
        manifest, expected_candidate_id='pwl-silu-s-v1',
        expected_fit_reference=fit_reference, expected_fit=fit)

    assert validated['report'] == asdict(validated['report_object'])
    assert validated['operation_manifest'] == {
        'function': 'silu',
        'source': 'module',
        'operation_roles': ['block.act'],
        'exact_input_roles': ['block.act.input'],
        'segments_per_role': 4,
        'domain_handling': {
            'kind': 'continuous-asymptotic-tail-v1',
            'left': 'constant-endpoint',
            'right': 'identity-plus-endpoint-offset'},
        'hardware_latency_claimed': False,
    }
    assert manifest['schema_version'] == 2
    forged = json.loads(json.dumps(manifest))
    forged['report']['observed_range_max_error'] += 1.0
    with pytest.raises(ValueError, match='measured fit'):
        validate_pwl_installation_manifest(
            forged, expected_candidate_id='pwl-silu-s-v1',
            expected_fit_reference=fit_reference, expected_fit=fit)

    downgraded = json.loads(json.dumps(manifest))
    downgraded['report']['saturation'] = 'clamp'
    downgraded['operation_manifest']['domain_handling'] = {'kind': 'clamp'}
    with pytest.raises(ValueError, match='domain handling'):
        validate_pwl_installation_manifest(
            downgraded, expected_candidate_id='pwl-silu-s-v1',
            expected_fit_reference=fit_reference, expected_fit=fit)

    legacy = json.loads(json.dumps(manifest))
    legacy['schema_version'] = 1
    with pytest.raises(ValueError, match='identity'):
        validate_pwl_installation_manifest(
            legacy, expected_candidate_id='pwl-silu-s-v1',
            expected_fit_reference=fit_reference, expected_fit=fit)


@pytest.mark.parametrize(
    'missing_field',
    ['qat_form', 'hardware_latency_claimed', 'exact_comparator'])
def test_installation_report_rejects_missing_defaulted_fields(missing_field):
    """Break caught: serialized reports must not consume dataclass defaults."""
    from mambapose_opt.pwl_artifacts import (
        build_pwl_installation_manifest, validate_pwl_installation_manifest)

    fit = _fit()
    fit_reference = {
        'path': 'work_dirs/optimization/pwl-silu-s-v1/calibrate/calibrate.json',
        'sha256': 'a' * 64}
    manifest = build_pwl_installation_manifest(
        candidate_id='pwl-silu-s-v1', fit=fit,
        fit_reference=fit_reference)
    manifest['report'].pop(missing_field)

    with pytest.raises(ValueError, match='installation report is invalid'):
        validate_pwl_installation_manifest(
            manifest, expected_candidate_id='pwl-silu-s-v1',
            expected_fit_reference=fit_reference, expected_fit=fit)


def test_installation_validation_requires_measured_fit_authority():
    """Break caught: a self-described exp report must not replace SiLU fit."""
    from mambapose_opt.pwl_artifacts import (
        build_pwl_installation_manifest, validate_pwl_installation_manifest)

    fit = _fit()
    fit_reference = {
        'path': 'work_dirs/optimization/pwl-silu-s-v1/calibrate/calibrate.json',
        'sha256': 'a' * 64}
    forged = build_pwl_installation_manifest(
        candidate_id='pwl-silu-s-v1', fit=fit,
        fit_reference=fit_reference)
    forged['report'].update({
        'function_name': 'exp', 'source': 'ss2d-transition',
        'input_roles': ['block.act.transition_exp_input'],
        'saturation': 'clamp',
        'exact_comparator': {
            'kind': 'exact-export-time-constant-folding',
            'applicable_source': 'static-parameter',
            'runtime_nonlinear_operations': 0,
            'max_error': 0.0, 'mean_error': 0.0,
            'preferred_over_pwl_when_exportable': True}})
    forged['operation_manifest'].update({
        'function': 'exp', 'source': 'ss2d-transition',
        'exact_input_roles': ['block.act.transition_exp_input'],
        'domain_handling': {'kind': 'clamp'}})

    with pytest.raises(ValueError, match='measured fit authority'):
        validate_pwl_installation_manifest(
            forged, expected_candidate_id='pwl-silu-s-v1',
            expected_fit_reference=fit_reference, expected_fit=None)


def test_calibration_hook_session_observes_declared_module_exact_inputs():
    from mambapose_opt.numeric_calibration import CalibrationTargets
    from tools.optimization.calibrate_numeric import _HookSession

    model = torch.nn.ModuleDict({
        'block': torch.nn.ModuleDict({'act': torch.nn.SiLU()})})
    empty = CalibrationTargets(
        ss2d_boundaries=(), vmamba_in_proj=(), vmamba_out_proj=(),
        attention_qkv=(), pif_boundaries=(), heatmap_projection=(),
        transition_parameters=(), functional_observers=(),
        unsupported_internals=())

    with _HookSession(model, empty, pwl_policy=_policy()) as session:
        model['block']['act'](torch.tensor([-3.0, 0.0, 1.0]))
        fit = session.pwl_report(candidate_id='pwl-silu-s-v1')

    assert fit['input_roles'][0]['exact_input_role'] == 'block.act.input'
    assert fit['observed_range'] == [-3.0, 1.0]


def test_calibration_hook_session_observes_ss2d_pre_nonlinear_exact_input():
    from mambapose_opt.numeric_calibration import CalibrationTargets
    from tools.optimization.calibrate_numeric import _HookSession

    class FunctionalHost(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.callback = None

        def set_numeric_observer(self, callback):
            self.callback = callback

    model = torch.nn.ModuleDict({'block': FunctionalHost()})
    targets = CalibrationTargets(
        ss2d_boundaries=(), vmamba_in_proj=(), vmamba_out_proj=(),
        attention_qkv=(), pif_boundaries=(), heatmap_projection=(),
        transition_parameters=(), functional_observers=('block',),
        unsupported_internals=())
    policy = _policy('softplus')
    policy['roles'] = ('block',)

    with _HookSession(model, targets, pwl_policy=policy) as session:
        model['block'].callback(
            'transition_softplus_input', torch.tensor([-7.0, 2.0]))
        fit = session.pwl_report(candidate_id='pwl-softplus-s-v1')

    assert fit['input_roles'][0]['exact_input_role'] == (
        'block.transition_softplus_input')
    assert fit['observed_range'] == [-7.0, 2.0]


def test_numeric_runtime_installs_only_hash_bound_fit_and_manifest(
        tmp_path, monkeypatch):
    from mambapose_opt import numeric_conversion
    from mambapose_opt.numeric_conversion import (
        NumericBindingError, NumericRuntimeHook)
    from mambapose_opt.pwl_artifacts import build_pwl_installation_manifest
    from mmpose.models.utils.hardware_friendly import (
        PiecewiseLinearApproximation)

    fit = _fit()
    fit_path, fit_reference = _write_reference(
        tmp_path, fit,
        'work_dirs/optimization/pwl-silu-s-v1/calibrate/calibrate.json')
    installation = build_pwl_installation_manifest(
        candidate_id='pwl-silu-s-v1', fit=fit,
        fit_reference=fit_reference)
    install_path = (
        tmp_path / 'work_dirs/optimization/pwl-silu-s-v1/'
        'convert/pwl-installation.json')
    install_path.parent.mkdir(parents=True, exist_ok=True)
    install_path.write_text(json.dumps(installation), encoding='utf-8')
    install_reference = {
        'path': install_path.relative_to(tmp_path).as_posix(),
        'sha256': hashlib.sha256(install_path.read_bytes()).hexdigest(),
    }
    selection_path = (
        tmp_path / 'work_dirs/optimization/ssm-quant-pwl/'
        'pwl-selection/selection.json')
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text('{}', encoding='utf-8')
    selection_reference = {
        'path': selection_path.relative_to(tmp_path).as_posix(),
        'sha256': hashlib.sha256(selection_path.read_bytes()).hexdigest(),
    }
    policy = _policy()
    policy.update({
        'candidate_id': 'pwl-silu-s-v1',
        'fit_artifact': fit_reference,
        'installation_manifest': install_reference,
        'selection_artifact': selection_reference,
    })
    monkeypatch.setattr(numeric_conversion, 'REPOSITORY_ROOT', tmp_path)
    monkeypatch.setattr(
        'mambapose_opt.pwl_selection.load_pwl_selection_reference',
        lambda *_args, **_kwargs: {
            'selected_candidate_id': 'pwl-silu-s-v1'})
    model = torch.nn.ModuleDict({
        'block': torch.nn.ModuleDict({'act': torch.nn.SiLU()})})

    report = NumericRuntimeHook.apply_to_model(
        model, {'candidate_kind': 'pwl', 'pwl': policy})

    assert report.fit_artifact_sha256 == fit_reference['sha256']
    assert isinstance(model['block']['act'], PiecewiseLinearApproximation)
    fit_path.write_text(json.dumps({'pwl_fit': _fit('gelu')}), encoding='utf-8')
    second = torch.nn.ModuleDict({
        'block': torch.nn.ModuleDict({'act': torch.nn.SiLU()})})
    with pytest.raises(NumericBindingError, match='hash changed'):
        NumericRuntimeHook.apply_to_model(
            second, {'candidate_kind': 'pwl', 'pwl': policy})


@pytest.mark.parametrize('name', ['silu', 'gelu', 'softplus', 'exp'])
def test_pwl_configs_declare_artifact_ranking_and_global_campaign(name):
    from mmengine.config import Config

    from mambapose_opt.numeric_conversion import numeric_stage_plan

    config = Config.fromfile(f'configs/optimization/numeric/pwl_{name}.py')
    numeric = config.numeric_optimization
    assert tuple(numeric.stage_order) == (
        'calibrate', 'pwl-selection', 'convert', 'smoke-stage-a',
        'profile', 'evaluate', 'latency')
    assert numeric.pwl.selection_policy == (
        'observed-range-max-then-mean-v1')
    assert numeric.calibration.artifact_schema_version == 3
    assert numeric_stage_plan('pwl', conditional=True) == tuple(
        numeric.stage_order)


@pytest.mark.parametrize('raw', [
    'work_dirs/./optimization/x.json',
    'work_dirs//optimization/x.json',
    'work_dirs/optimization/x.json/',
])
def test_pwl_public_references_reject_lexical_aliases(raw, tmp_path):
    from mambapose_opt.pwl_artifacts import _strict_reference
    from mambapose_opt.pwl_selection import (
        _reference, validate_pwl_selection_artifact)
    from mambapose_opt.pwl_smoke import (
        _binding, validate_pwl_stage_a_artifact)
    from mambapose_opt.numeric_runtime import _file

    value = {'path': raw, 'sha256': 'a' * 64}
    with pytest.raises(ValueError, match='invalid|unsafe|canonical'):
        _reference(value, label='selection')
    with pytest.raises(ValueError, match='invalid|unsafe|canonical'):
        _strict_reference(value, repository_root=Path.cwd(), label='fit')
    with pytest.raises(ValueError, match='invalid|unsafe|canonical'):
        _binding(value, label='smoke')
    with pytest.raises(ValueError, match='invalid|unsafe|canonical'):
        _file(tmp_path, value, 'numeric export')
    with pytest.raises(ValueError, match='invalid|unsafe|canonical'):
        validate_pwl_selection_artifact(
            {}, repository_root=tmp_path,
            manifest_path='optimization/./candidates.json')
    with pytest.raises(ValueError, match='invalid|unsafe|canonical'):
        validate_pwl_stage_a_artifact(
            'work_dirs/./optimization/x/smoke-stage-a/smoke.json',
            repository_root=tmp_path,
            manifest_path='optimization/candidates.json')


def test_pwl_cli_rejects_lexical_aliases_before_resolution():
    import argparse
    from tools.optimization.select_pwl_candidate import _calibrations
    from tools.optimization.smoke_pwl import _output_root

    with pytest.raises(ValueError, match='canonical|relative|unsafe'):
        _calibrations([
            'pwl-silu-s-v1=work_dirs/./optimization/a.json',
            'pwl-gelu-s-v1=work_dirs/optimization/b.json',
            'pwl-softplus-s-v1=work_dirs/optimization/c.json',
            'pwl-exp-s-v1=work_dirs/optimization/d.json'])
    with pytest.raises(
            argparse.ArgumentTypeError,
            match='canonical|relative|unsafe|smoke-stage-a'):
        _output_root(
            'work_dirs/./optimization/ssm-quant-pwl/pwl-silu-s-v1/0/'
            'smoke-stage-a')


def test_pwl_convert_requires_fit_artifact_before_model_load(
        tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tools.optimization import convert_numeric

    candidate = SimpleNamespace(
        id='pwl-silu-s-v1', route='ssm-quant-pwl',
        features={'numeric_kind': 'pwl'}, config=Path('configs/pwl.py'),
        checkpoint=Path('checkpoint.pth'), checkpoint_sha256='a' * 64)
    entered = []
    monkeypatch.setattr(
        convert_numeric, 'authorize_manifest_candidate',
        lambda *_args: SimpleNamespace(candidate=candidate))
    monkeypatch.setattr(
        'mmpose.apis.init_model',
        lambda *_args, **_kwargs: entered.append(True))

    with pytest.raises(ValueError, match='PWL.*calibration|fit artifact'):
        convert_numeric.convert(
            candidate, stage='convert',
            output=tmp_path / 'work_dirs/optimization/pwl/convert/convert.json',
            manifest_path=tmp_path / 'optimization/candidates.json')
    assert entered == []


def test_pwl_convert_requires_hash_bound_four_candidate_selection_before_load(
        tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tools.optimization import convert_numeric

    candidate = SimpleNamespace(
        id='pwl-silu-s-v1', route='ssm-quant-pwl',
        features={'numeric_kind': 'pwl'}, config=Path('configs/pwl.py'),
        checkpoint=Path('checkpoint.pth'), checkpoint_sha256='a' * 64)
    calibration = tmp_path / 'work_dirs/optimization/calibrate.json'
    calibration.parent.mkdir(parents=True)
    calibration.write_text('{}', encoding='utf-8')
    entered = []
    monkeypatch.setattr(
        convert_numeric, 'authorize_manifest_candidate',
        lambda *_args: SimpleNamespace(candidate=candidate))
    monkeypatch.setattr(
        'mmpose.apis.init_model',
        lambda *_args, **_kwargs: entered.append(True))

    with pytest.raises(ValueError, match='selection'):
        convert_numeric.convert(
            candidate, stage='convert',
            output=tmp_path / 'work_dirs/optimization/pwl/convert.json',
            manifest_path=tmp_path / 'optimization/candidates.json',
            calibration_artifact=calibration)
    assert entered == []


def test_pwl_campaign_convert_consumes_canonical_calibrate_output(
        tmp_path, monkeypatch):
    from mambapose_opt.schema import CandidateSpec
    from tools.optimization import run_campaign
    from tools.optimization.run_campaign import SubprocessStageRunner

    candidate = CandidateSpec.from_dict({
        'id': 'pwl-silu-s-v1', 'route': 'ssm-quant-pwl', 'kind': 'pwl',
        'config': 'configs/optimization/numeric/pwl_silu.py',
        'checkpoint': 'checkpoint.pth', 'checkpoint_sha256': 'a' * 64,
        'seed': 0, 'features': {
            'numeric_kind': 'pwl', 'conditional': True, 'auto_run': False},
    })
    output = (
        tmp_path / 'work_dirs/optimization/pwl-silu-s-v1/'
        'convert/convert.json')
    monkeypatch.setattr(run_campaign, 'REPO_ROOT', tmp_path)
    runner = SubprocessStageRunner(
        tmp_path / 'work_dirs/optimization',
        tmp_path / 'optimization/candidates.json')

    command = runner._command(candidate, 'convert', output)

    index = command.index('--calibration-artifact')
    assert command[index + 1].endswith(
        'pwl-silu-s-v1/calibrate/calibrate.json')
    selection_index = command.index('--selection-artifact')
    assert command[selection_index + 1].endswith(
        'ssm-quant-pwl/pwl-selection/selection.json')


def test_pwl_runtime_resolver_fails_closed_before_convert(
        tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mambapose_opt.numeric_runtime import (
        NumericRuntimeError, resolve_numeric_runtime)
    from mambapose_opt.schema import CandidateSpec

    checkpoint = tmp_path / 'checkpoint.pth'
    checkpoint.write_bytes(b'checkpoint')
    candidate = CandidateSpec.from_dict({
        'id': 'pwl-silu-s-v1', 'route': 'ssm-quant-pwl', 'kind': 'pwl',
        'config': 'configs/pwl.py', 'checkpoint': 'checkpoint.pth',
        'checkpoint_sha256': hashlib.sha256(b'checkpoint').hexdigest(),
        'seed': 0, 'features': {
            'numeric_kind': 'pwl', 'conditional': True, 'auto_run': False},
    })
    monkeypatch.setattr(
        'mambapose_opt.numeric_runtime.authorize_manifest_candidate',
        lambda *_args: SimpleNamespace(
            candidate=candidate, checkpoint_path=checkpoint))

    with pytest.raises(NumericRuntimeError, match='completed convert'):
        resolve_numeric_runtime(
            candidate, repository_root=tmp_path,
            manifest_path=tmp_path / 'optimization/candidates.json',
            downstream_output=(
                tmp_path / 'work_dirs/optimization/pwl-silu-s-v1/'
                'profile/profile.json'))


def test_pwl_producer_controller_and_all_downstream_share_install_provenance(
        tmp_path, monkeypatch):
    from torch import nn

    from mambapose_opt.controller import OptimizationController
    from mambapose_opt.numeric_runtime import (
        resolve_numeric_runtime, validate_numeric_convert_artifact)
    from mambapose_opt.schema import load_candidate_manifest
    from tools.optimization import convert_numeric

    (tmp_path / 'configs').mkdir()
    (tmp_path / 'optimization').mkdir()
    (tmp_path / 'work_dirs/reproduction').mkdir(parents=True)
    (tmp_path / 'work_dirs/optimization').mkdir(parents=True)
    (tmp_path / 'data/coco/train2017').mkdir(parents=True)
    (tmp_path / 'data/coco/annotations').mkdir(parents=True)
    image = tmp_path / 'data/coco/train2017/000000000001.jpg'
    annotation = (
        tmp_path / 'data/coco/annotations/person_keypoints_train2017.json')
    image.write_bytes(b'image')
    annotation.write_bytes(b'{}')
    train_archive = (
        tmp_path / 'work_dirs/reproduction/downloads/train2017.zip')
    annotation_archive = (
        tmp_path / 'work_dirs/reproduction/downloads/annotations.zip')
    train_archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(train_archive, 'w') as archive:
        archive.writestr('train2017/000000000001.jpg', b'image')
    with zipfile.ZipFile(annotation_archive, 'w') as archive:
        archive.writestr(
            'annotations/person_keypoints_train2017.json', b'{}')
    (tmp_path / 'data/inventory.json').write_text(json.dumps({
        'schema_version': 1,
        'assets': [
            {'id': 'coco-train2017',
             'path': train_archive.relative_to(tmp_path).as_posix(),
             'sha256': hashlib.sha256(train_archive.read_bytes()).hexdigest()},
            {'id': 'coco-annotations',
             'path': annotation_archive.relative_to(tmp_path).as_posix(),
             'sha256': hashlib.sha256(
                 annotation_archive.read_bytes()).hexdigest()},
        ],
    }), encoding='utf-8')
    (tmp_path / '.gitignore').write_text(
        'data/\nwork_dirs/\n', encoding='utf-8')
    checkpoint = tmp_path / 'work_dirs/reproduction/checkpoint.pth'
    checkpoint.write_bytes(b'checkpoint')
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = tmp_path / 'configs/pwl.py'
    config.write_text(
        "model = dict(type='Fixture', test_cfg=dict(flip_test=False))\n"
        "train_dataloader = dict(batch_size=1, num_workers=0, "
        "persistent_workers=False)\n"
        "val_dataloader = dict(batch_size=1, num_workers=0, "
        "persistent_workers=False)\n"
        "test_dataloader = dict(batch_size=1, num_workers=0, "
        "persistent_workers=False)\n"
        "numeric_optimization = dict(\n"
        "    candidate_kind='pwl',\n"
        "    calibration=dict(artifact_schema_version=3),\n"
        "    pwl=dict(\n"
        "        candidate_id='pwl-silu-s-v1', enabled_function='silu',\n"
        "        source='module', roles=('layer',), domain=(-2.0, 2.0),\n"
        "        segments=4, grid_points=129,\n"
        "        saturation='continuous-asymptotic-tail-v1',\n"
        "        qat_form='differentiable',\n"
        "        selection_policy='observed-range-max-then-mean-v1'))\n",
        encoding='utf-8')
    manifest = tmp_path / 'optimization/candidates.json'
    manifest.write_text(json.dumps({
        'schema_version': 1,
        'candidates': [{
            'id': 'pwl-silu-s-v1', 'route': 'ssm-quant-pwl',
            'kind': 'pwl', 'config': 'configs/pwl.py',
            'checkpoint': 'work_dirs/reproduction/checkpoint.pth',
            'checkpoint_sha256': checkpoint_sha, 'seed': 0,
            'features': {'numeric_kind': 'pwl', 'conditional': True,
                         'auto_run': False},
        }],
    }), encoding='utf-8')
    (tmp_path / 'optimization/coco_train2017_authority.json').write_text(
        '{}\n', encoding='utf-8')
    (tmp_path / 'optimization/coco_val2017_authority.json').write_text(
        '{}\n', encoding='utf-8')
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'test@example.com'],
        cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'config', 'user.name', 'Test'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'commit', '-qm', 'pwl fixture'], cwd=tmp_path, check=True)
    candidate = load_candidate_manifest(manifest)[0]
    root = tmp_path / 'work_dirs/optimization/pwl-silu-s-v1'
    calibration_path = root / 'calibrate/calibrate.json'
    calibration_path.parent.mkdir(parents=True)
    from mambapose_opt.pwl_artifacts import fit_pwl_observations
    fixture_policy = _policy()
    fixture_policy['roles'] = ('layer',)
    fixture_fit = fit_pwl_observations(
        candidate_id=candidate.id, policy=fixture_policy,
        observations={'layer': [torch.tensor([-1.0, 0.0, 1.0])]})
    calibration_path.write_text(json.dumps({
        'schema_version': 3, 'candidate_id': candidate.id,
        'stage': 'calibrate',
        'source': {
            'authority_path': 'optimization/coco_train2017_authority.json'},
        'identity': {},
        'protocol': {}, 'hooks': {}, 'pwl_fit': fixture_fit,
    }), encoding='utf-8')
    output = root / 'convert/convert.json'
    output.parent.mkdir(parents=True)
    model = nn.Module()
    model.layer = nn.SiLU()
    monkeypatch.setattr(convert_numeric, 'REPOSITORY_ROOT', tmp_path)
    monkeypatch.setattr(
        convert_numeric, 'build_manifest_authorized_model',
        lambda *_args, **_kwargs: model)
    import mambapose_opt.numeric_calibration as numeric_calibration

    def validate_fixture_calibration(value, **_kwargs):
        numeric_calibration._verified_dataset_authority(
            tmp_path, image.parent, annotation)
        return value

    monkeypatch.setattr(
        convert_numeric, 'validate_calibration_provenance',
        validate_fixture_calibration)
    monkeypatch.setattr(
        numeric_calibration, 'validate_calibration_provenance',
        validate_fixture_calibration)
    selection_path = (
        tmp_path / 'work_dirs/optimization/ssm-quant-pwl/'
        'pwl-selection/selection.json')
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    other_calibration = (
        tmp_path / 'work_dirs/optimization/ssm-quant-pwl/'
        'pwl-gelu-s-v1/0/calibrate/calibrate.json')
    other_calibration.parent.mkdir(parents=True)
    other_calibration.write_text('{}', encoding='utf-8')
    selection_path.write_text(json.dumps({
        'candidates': [{
            'calibration_artifact': {
                'path': other_calibration.relative_to(tmp_path).as_posix(),
                'sha256': hashlib.sha256(
                    other_calibration.read_bytes()).hexdigest(),
            },
        }],
    }), encoding='utf-8')
    monkeypatch.setattr(
        convert_numeric, 'load_pwl_selection_reference',
        lambda *_args, **_kwargs: {
            'decision': 'selected',
            'selected_candidate_id': 'pwl-silu-s-v1'})
    monkeypatch.setattr(
        'mambapose_opt.pwl_selection.load_pwl_selection_reference',
        lambda *_args, **_kwargs: {
            'decision': 'selected',
            'selected_candidate_id': 'pwl-silu-s-v1'})
    monkeypatch.setattr(
        'mambapose_opt.pwl_smoke.pwl_stage_a_binding',
        lambda *_args, **_kwargs: {
            'path': ('work_dirs/optimization/ssm-quant-pwl/'
                     'pwl-silu-s-v1/0/smoke-stage-a/smoke.json'),
            'sha256': '9' * 64})

    produced = convert_numeric.convert(
        candidate, stage='convert', output=output, manifest_path=manifest,
        calibration_artifact=calibration_path,
        selection_artifact=selection_path)
    output.write_text(json.dumps(produced), encoding='utf-8')
    round_tripped = json.loads(output.read_text(encoding='utf-8'))

    assert validate_numeric_convert_artifact(
        round_tripped, candidate=candidate, repository_root=tmp_path,
        manifest_path=manifest, artifact_path=output) == round_tripped
    controller = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate,
        lambda *_args: None, repository_root=tmp_path,
        manifest_path=manifest, stages=('convert',))
    assert controller._artifact_schema('convert', output) == (
        'optimization-stage-envelope-v1')
    runtimes = [resolve_numeric_runtime(
        candidate, repository_root=tmp_path, manifest_path=manifest,
        downstream_output=root / stage / f'{stage}.json')
        for stage in ('profile', 'evaluate', 'latency')]
    assert len({item['config_sha256'] for item in runtimes}) == 1
    assert len({item['pwl_installation']['sha256'] for item in runtimes}) == 1
    assert len({item['pwl_stage_a']['sha256'] for item in runtimes}) == 1
    assert all(item['pwl_installation'] == produced['result']['installation']
               for item in runtimes)

    from mambapose_opt.checkpoints import (
        authorize_pwl_runtime_config,
        load_materialized_config_authority,
        materialize_evaluation_config_authority)
    import mambapose_opt.checkpoints as checkpoints
    original_runtime_snapshot = checkpoints._pwl_runtime_config_snapshot
    runtime_snapshot_calls = 0

    def counted_runtime_snapshot(*args, **kwargs):
        nonlocal runtime_snapshot_calls
        runtime_snapshot_calls += 1
        return original_runtime_snapshot(*args, **kwargs)

    monkeypatch.setattr(
        checkpoints, '_pwl_runtime_config_snapshot',
        counted_runtime_snapshot)
    authority = authorize_pwl_runtime_config(
        tmp_path, manifest, candidate, conversion_path=output)
    assert authority.path == output.parent / 'resolved-runtime.py'
    assert authority.sha256 == produced['result']['runtime_config']['sha256']
    assert authority.load_config().numeric_optimization.pwl.candidate_id == (
        candidate.id)
    authority.verify()
    repeated_authority = authorize_pwl_runtime_config(
        tmp_path, manifest, candidate, conversion_path=output)
    assert repeated_authority.load_config().numeric_optimization.pwl.candidate_id \
        == candidate.id
    assert runtime_snapshot_calls == 1
    health = tmp_path / 'work_dirs/reproduction/monitor/status.json'
    health.parent.mkdir(parents=True)
    health.write_text('{"status":"alive"}\n', encoding='utf-8')
    assert authority.load_config().numeric_optimization.pwl.candidate_id == (
        candidate.id)
    assert runtime_snapshot_calls == 1
    entry = checkpoints._CONFIG_MEMO[
        checkpoints._memo_key(authority)]
    scoped_files = {path.as_posix() for path in entry.scope.files}
    assert {
        output.relative_to(tmp_path).as_posix(),
        calibration_path.relative_to(tmp_path).as_posix(),
        selection_path.relative_to(tmp_path).as_posix(),
        other_calibration.relative_to(tmp_path).as_posix(),
        'optimization/coco_train2017_authority.json',
        'data/inventory.json',
        annotation.relative_to(tmp_path).as_posix(),
        train_archive.relative_to(tmp_path).as_posix(),
        annotation_archive.relative_to(tmp_path).as_posix(),
    } <= scoped_files
    assert {path.as_posix() for path in entry.scope.trees} == {
        image.parent.relative_to(tmp_path).as_posix()}
    assert 'work_dirs/reproduction/monitor/status.json' not in scoped_files
    for dependency in (selection_path, calibration_path, other_calibration,
                       annotation, train_archive, annotation_archive):
        original_stat = dependency.stat()
        replacement = dependency.with_name(f'{dependency.name}.replacement')
        replacement.write_bytes(dependency.read_bytes())
        os.utime(
            replacement,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        os.replace(replacement, dependency)
        assert authority.load_config().numeric_optimization.pwl.candidate_id == (
            candidate.id)
    assert runtime_snapshot_calls == 7

    image_stat = image.stat()
    image.write_bytes(b'MUTAT')
    os.utime(image, ns=(image_stat.st_atime_ns, image_stat.st_mtime_ns))
    assert image.stat().st_ctime_ns != image_stat.st_ctime_ns
    with pytest.raises(ValueError, match='content'):
        authority.load_config()
    assert runtime_snapshot_calls == 8
    image.write_bytes(b'image')
    assert authority.load_config().numeric_optimization.pwl.candidate_id == (
        candidate.id)
    assert runtime_snapshot_calls == 9

    image_stat = image.stat()
    replacement_image = image.with_name('replacement.jpg')
    replacement_image.write_bytes(image.read_bytes())
    os.utime(
        replacement_image,
        ns=(image_stat.st_atime_ns, image_stat.st_mtime_ns))
    os.replace(replacement_image, image)
    assert image.stat().st_ino != image_stat.st_ino
    assert authority.load_config().numeric_optimization.pwl.candidate_id == (
        candidate.id)
    assert runtime_snapshot_calls == 10

    extra_image = image.with_name('000000000002.jpg')
    extra_image.write_bytes(b'extra')
    with pytest.raises(ValueError, match='membership'):
        authority.load_config()
    assert runtime_snapshot_calls == 11
    extra_image.unlink()
    assert authority.load_config().numeric_optimization.pwl.candidate_id == (
        candidate.id)
    assert runtime_snapshot_calls == 12

    image_bytes = image.read_bytes()
    image.unlink()
    with pytest.raises(ValueError, match='membership'):
        authority.load_config()
    assert runtime_snapshot_calls == 13
    image.write_bytes(image_bytes)
    assert authority.load_config().numeric_optimization.pwl.candidate_id == (
        candidate.id)
    assert runtime_snapshot_calls == 14

    image_target = tmp_path / 'data/alternate-image.jpg'
    image_target.write_bytes(b'image')
    original_clone = checkpoints._clone_config_snapshot
    armed = True

    def symlink_during_hit(snapshot):
        nonlocal armed
        result = original_clone(snapshot)
        if armed:
            armed = False
            image.unlink()
            image.symlink_to(image_target)
        return result

    monkeypatch.setattr(
        checkpoints, '_clone_config_snapshot', symlink_during_hit)
    with pytest.raises(ValueError, match='symlink'):
        authority.load_config()
    assert runtime_snapshot_calls == 15
    image.unlink()
    image.write_bytes(b'image')
    monkeypatch.setattr(
        checkpoints, '_clone_config_snapshot', original_clone)

    materialized_path = output.parent.parent / 'evaluate/resolved-flip.py'
    authority_path = output.parent.parent / (
        'evaluate/resolved-flip.config-authority.json')
    materialized = materialize_evaluation_config_authority(
        authority, flip_test=True, config_path=materialized_path,
        authority_path=authority_path)
    loaded = load_materialized_config_authority(
        tmp_path, manifest, candidate, authority_path)
    assert loaded.sha256 == materialized.sha256
    assert loaded.load_config().model.test_cfg.flip_test is True

    original = materialized_path.read_bytes()
    materialized_path.write_text(
        "model = dict(type='Fixture', test_cfg=dict(flip_test=False))\n",
        encoding='utf-8')
    with pytest.raises(ValueError, match='materialized|authority|changed'):
        loaded.verify()
    materialized_path.write_bytes(original)
    loaded.verify()
