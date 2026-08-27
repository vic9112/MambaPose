import pytest
import torch
import torch.nn.functional as F
from mmengine.config import Config
from torch import nn


def _install_measured_pwl(model, *, function_name, source, roles):
    from mambapose_opt.numeric_conversion import install_pwl_fit
    from mambapose_opt.pwl_artifacts import (
        build_pwl_installation_manifest, fit_pwl_observations,
        validate_pwl_installation_manifest)

    policy = {
        'enabled_function': function_name, 'source': source, 'roles': roles,
        'domain': (-4.0, 4.0), 'segments': 4, 'grid_points': 129,
        'saturation': 'clamp', 'qat_form': 'differentiable',
        'selection_policy': 'observed-range-max-then-mean-v1',
    }
    fit = fit_pwl_observations(
        candidate_id=f'pwl-{function_name}-test', policy=policy,
        observations={
            role: [torch.tensor([-3.0, -0.2, 1.3, 3.8])]
            for role in roles})
    reference = {'path': 'work_dirs/optimization/test/calibrate.json',
                 'sha256': 'a' * 64}
    manifest = build_pwl_installation_manifest(
        candidate_id=f'pwl-{function_name}-test', fit=fit,
        fit_reference=reference)
    report = validate_pwl_installation_manifest(
        manifest, expected_candidate_id=f'pwl-{function_name}-test',
        expected_fit_reference=reference)['report_object']
    return install_pwl_fit(model, fit=fit, expected_report=report)


def test_pwl_rejects_unsorted_or_discontinuous_segments():
    from mmpose.models.utils.hardware_friendly import PiecewiseLinearApproximation

    with pytest.raises(ValueError, match='strictly increasing'):
        PiecewiseLinearApproximation([0.0, -1.0, 1.0], [1.0, 1.0], [0.0, 0.0])
    with pytest.raises(ValueError, match='continuous'):
        PiecewiseLinearApproximation([-1.0, 0.0, 1.0], [1.0, 2.0], [0.0, 1.0])


def test_pwl_selects_segments_and_saturates_at_declared_domain():
    from mmpose.models.utils.hardware_friendly import PiecewiseLinearApproximation

    approximation = PiecewiseLinearApproximation(
        [-1.0, 0.0, 1.0], [1.0, 2.0], [0.0, 0.0])
    actual = approximation(torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0]))

    torch.testing.assert_close(
        actual, torch.tensor([-1.0, -0.5, 0.0, 1.0, 2.0]))
    assert approximation.domain == (-1.0, 1.0)
    assert approximation.segments == 2
    assert approximation.saturation == 'clamp'


@pytest.mark.parametrize('name, reference, domain', [
    ('silu', F.silu, (-6.0, 6.0)),
    ('gelu', F.gelu, (-5.0, 5.0)),
    ('softplus', F.softplus, (-8.0, 8.0)),
    ('exp', torch.exp, (-4.0, 4.0)),
])
def test_fit_pwl_is_deterministic_continuous_and_declares_error(
        name, reference, domain):
    from mmpose.models.utils.hardware_friendly import fit_pwl

    first = fit_pwl(reference, domain, segments=16, grid_points=4097,
                    function_name=name)
    second = fit_pwl(reference, domain, segments=16, grid_points=4097,
                     function_name=name)

    assert first.function_name == name
    assert first.state_dict().keys() == second.state_dict().keys()
    for key in first.state_dict():
        torch.testing.assert_close(first.state_dict()[key], second.state_dict()[key])
    assert first.max_error >= first.mean_error >= 0
    x = torch.linspace(*domain, 101, requires_grad=True)
    first(x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_disabled_pwl_is_bit_exact_and_has_no_state_keys():
    from mmpose.models.utils.hardware_friendly import PiecewiseLinearApproximation

    approximation = PiecewiseLinearApproximation.identity()
    x = torch.randn(17)

    assert approximation.state_dict() == {}
    torch.testing.assert_close(approximation(x), x, rtol=0, atol=0)


@pytest.mark.parametrize(('function_name', 'module_type'), [
    ('silu', nn.SiLU),
    ('gelu', nn.GELU),
])
def test_pwl_runtime_replaces_only_explicit_module_roles_and_is_called(
        function_name, module_type):
    from mmpose.models.utils.hardware_friendly import PiecewiseLinearApproximation

    model = nn.ModuleDict({'selected': module_type(), 'untouched': module_type()})
    before_keys = tuple(model.state_dict())
    report = _install_measured_pwl(
        model, function_name=function_name, source='module',
        roles=('selected',))
    assert report.function_name == function_name
    assert report.source == 'module'
    assert report.roles == ('selected',)
    assert isinstance(model['selected'], PiecewiseLinearApproximation)
    assert isinstance(model['untouched'], module_type)
    assert tuple(model.state_dict()) != before_keys
    calls = []
    model['selected'].register_forward_hook(lambda *_args: calls.append(True))
    value = torch.tensor([-3.7, -0.2, 1.3, 3.8])
    optimized = model['selected'](value)
    reference = (F.silu(value) if function_name == 'silu' else F.gelu(value))
    assert calls == [True]
    assert not torch.equal(optimized, reference)


class _FunctionalPWLHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.installed = {}

    def install_numeric_pwl(self, name, approximation):
        self.add_module(f'numeric_pwl_{name}', approximation)
        self.installed[name] = approximation

    def forward(self, name, value):
        return self.installed[name](value)


@pytest.mark.parametrize(('function_name', 'reference'), [
    ('softplus', F.softplus),
])
def test_pwl_runtime_installs_functional_ss2d_source_and_is_called(
        function_name, reference):
    model = nn.ModuleDict({'scan': _FunctionalPWLHost()})
    report = _install_measured_pwl(
        model, function_name=function_name, source='ss2d-transition',
        roles=('scan',))
    approximation = model['scan'].installed[function_name]
    calls = []
    approximation.register_forward_hook(lambda *_args: calls.append(True))
    value = torch.tensor([-3.7, -0.2, 1.3, 3.8])
    optimized = model['scan'](function_name, value)
    assert report.roles == ('scan',)
    assert calls == [True]
    assert not torch.equal(optimized, reference(value))


def test_pwl_runtime_default_off_is_operation_and_state_key_exact():
    from mambapose_opt.numeric_conversion import apply_numeric_runtime

    model = nn.Sequential(nn.SiLU())
    state_keys = tuple(model.state_dict())
    value = torch.randn(13)
    expected = model(value)

    assert apply_numeric_runtime(model, {'candidate_kind': 'baseline'}) is None
    assert tuple(model.state_dict()) == state_keys
    torch.testing.assert_close(model(value), expected, rtol=0, atol=0)


def test_pwl_runtime_rejects_unfitted_ambiguous_function_or_role_source():
    from mambapose_opt.numeric_conversion import (
        NumericBindingError, apply_numeric_runtime)

    model = nn.Sequential(nn.SiLU())
    base = {
        'candidate_kind': 'pwl',
        'pwl': {
            'enabled_function': ('silu', 'gelu'), 'source': 'module',
            'roles': ('0',), 'domain': (-4.0, 4.0), 'segments': 4,
            'grid_points': 129, 'saturation': 'clamp',
            'qat_form': 'differentiable',
        },
    }
    with pytest.raises(NumericBindingError, match='fit artifact'):
        apply_numeric_runtime(model, base)
    base['pwl']['enabled_function'] = 'silu'
    base['pwl']['source'] = 'ss2d-transition'
    with pytest.raises(NumericBindingError, match='fit artifact'):
        apply_numeric_runtime(model, base)


@pytest.mark.parametrize('name', ['silu', 'gelu', 'softplus', 'exp'])
def test_pwl_stage_config_installs_real_numeric_runtime_hook(name):
    config = Config.fromfile(f'configs/optimization/numeric/pwl_{name}.py')

    assert 'mambapose_opt.numeric_conversion' in config.custom_imports.imports
    assert any(
        hook.get('type') == 'NumericRuntimeHook'
        for hook in config.custom_hooks)


def test_numeric_runtime_hook_refuses_unfitted_pwl_policy():
    from mambapose_opt.numeric_conversion import (
        NumericBindingError, NumericRuntimeHook)

    model = nn.ModuleDict({'selected': nn.SiLU()})
    with pytest.raises(NumericBindingError, match='fit artifact'):
        NumericRuntimeHook.apply_to_model(model, {
            'candidate_kind': 'pwl',
            'pwl': {
                'enabled_function': 'silu', 'source': 'module',
                'roles': ('selected',), 'domain': (-4.0, 4.0),
                'segments': 4, 'grid_points': 129, 'saturation': 'clamp',
                'qat_form': 'differentiable',
            },
        })


@pytest.mark.parametrize('function_name', ['softplus', 'exp'])
def test_production_ss2d_executes_installed_transition_pwl(function_name):
    from mmpose.models.backbones.Vmamba.vmamba import SS2D
    from mmpose.models.utils.hardware_friendly import fit_pwl

    class Cross:
        @staticmethod
        def apply(value):
            flat = value.flatten(2)
            return torch.stack((flat, flat, flat, flat), dim=1)

    scan_contracts = []

    class Scan:
        @staticmethod
        def apply(u, delta, A, B, C, D, delta_bias, delta_softplus,
                  _a, _b, _c):
            scan_contracts.append((delta_bias, delta_softplus))
            return u

    class Merge:
        @staticmethod
        def apply(value):
            return value.sum(dim=1)

    reference = F.softplus if function_name == 'softplus' else torch.exp
    approximation = fit_pwl(
        reference, (-4.0, 4.0), segments=4, grid_points=129,
        function_name=function_name)
    calls = []
    approximation.register_forward_hook(lambda *_args: calls.append(True))
    module = SS2D(
        d_model=4, d_state=2, ssm_ratio=1.0, dt_rank=1,
        d_conv=1, channel_first=True, forward_type='v2').eval()
    module.install_numeric_pwl(function_name, approximation)
    module.forward_corev2(
        torch.randn(1, 4, 2, 2), SelectiveScan=Scan,
        CrossScan=Cross, CrossMerge=Merge)

    assert calls == [True]
    if function_name == 'softplus':
        assert scan_contracts == [(None, False)]
    else:
        assert scan_contracts[0][0] is not None
        assert scan_contracts[0][1] is True
