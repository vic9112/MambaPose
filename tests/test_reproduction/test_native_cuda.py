import importlib
from pathlib import Path
import subprocess

import pytest
import torch


NATIVE_MODULES = (
    'causal_conv1d_cuda',
    'selective_scan_cuda',
    'selective_scan_cuda_core',
    'selective_scan_cuda_ndstate',
    'selective_scan_cuda_oflex',
)


def test_native_extensions_contain_only_sm120_sass():
    triton_root = Path(importlib.import_module('triton').__file__).parent
    cuobjdump = triton_root / 'backends/nvidia/bin/cuobjdump'
    assert cuobjdump.is_file()
    for module_name in NATIVE_MODULES:
        module = importlib.import_module(module_name)
        result = subprocess.run(
            [str(cuobjdump), '--list-elf', module.__file__],
            check=True,
            capture_output=True,
            text=True)
        cubins = [
            line for line in result.stdout.splitlines() if '.cubin' in line]
        assert cubins, module_name
        assert all('.sm_120.cubin' in line for line in cubins), (
            module_name, cubins)


def _tolerance(dtype: torch.dtype) -> dict[str, float]:
    if dtype == torch.float32:
        return {'rtol': 2e-4, 'atol': 3e-4}
    if dtype == torch.float16:
        return {'rtol': 3e-2, 'atol': 3e-2}
    return {'rtol': 8e-2, 'atol': 8e-2}


@pytest.mark.parametrize(
    ('dtype', 'width', 'channel_last'),
    [
        (torch.float32, 2, False),
        (torch.float32, 3, True),
        (torch.float16, 4, False),
        (torch.bfloat16, 4, True),
    ])
def test_causal_conv_forward_backward_matches_reference(
        dtype, width, channel_last):
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref

    torch.manual_seed(11)
    batch, channels, length = 2, 16, 19
    if channel_last:
        x = torch.randn(
            batch, length, channels, device='cuda', dtype=dtype
        ).transpose(1, 2)
    else:
        x = torch.randn(
            batch, channels, length, device='cuda', dtype=dtype)
    x = x.detach().requires_grad_()
    weight = torch.randn(
        channels, width, device='cuda', dtype=dtype,
        requires_grad=True)
    bias = torch.randn(
        channels, device='cuda', dtype=dtype, requires_grad=True)

    x_ref = x.detach().clone(memory_format=torch.preserve_format).requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_()
    actual = causal_conv1d_fn(x, weight, bias, activation='silu')
    expected = causal_conv1d_ref(
        x_ref, weight_ref, bias_ref, activation='silu')
    gradient = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(
        actual, (x, weight, bias), gradient, retain_graph=False)
    expected_grads = torch.autograd.grad(
        expected, (x_ref, weight_ref, bias_ref), gradient,
        retain_graph=False)

    torch.testing.assert_close(actual, expected, **_tolerance(dtype))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(
            actual_grad, expected_grad, **_tolerance(dtype))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_causal_conv_update_matches_reference(dtype):
    from causal_conv1d import causal_conv1d_update
    from causal_conv1d.causal_conv1d_interface import causal_conv1d_update_ref

    torch.manual_seed(12)
    batch, channels, width = 2, 16, 4
    x = torch.randn(batch, channels, device='cuda', dtype=dtype)
    state = torch.randn(
        batch, channels, width, device='cuda', dtype=dtype)
    state_ref = state.clone()
    weight = torch.randn(channels, width, device='cuda', dtype=dtype)
    bias = torch.randn(channels, device='cuda', dtype=dtype)
    actual = causal_conv1d_update(
        x, state, weight, bias, activation='silu')
    expected = causal_conv1d_update_ref(
        x, state_ref, weight, bias, activation='silu')
    torch.testing.assert_close(actual, expected, **_tolerance(dtype))
    torch.testing.assert_close(state, state_ref, **_tolerance(dtype))


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_mamba_selective_scan_forward_backward_matches_reference(dtype):
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn,
        selective_scan_ref,
    )

    torch.manual_seed(13)
    batch, channels, length, state = 2, 8, 17, 4
    u = torch.randn(
        batch, channels, length, device='cuda', dtype=dtype,
        requires_grad=True)
    delta = torch.randn_like(u, requires_grad=True)
    A = (-torch.rand(channels, state, device='cuda')).requires_grad_()
    B = torch.randn(
        batch, 1, state, length, device='cuda', dtype=dtype,
        requires_grad=True)
    C = torch.randn_like(B, requires_grad=True)
    D = torch.randn(channels, device='cuda', requires_grad=True)
    z = torch.randn_like(u, requires_grad=True)
    delta_bias = torch.randn(channels, device='cuda', requires_grad=True)
    inputs = (u, delta, A, B, C, D, z, delta_bias)
    reference_inputs = tuple(
        value.detach().clone().requires_grad_() for value in inputs)

    actual = selective_scan_fn(
        *inputs, delta_softplus=True, return_last_state=False)
    expected = selective_scan_ref(
        *reference_inputs, delta_softplus=True, return_last_state=False)
    actual_grads = torch.autograd.grad(actual.sum(), inputs)
    expected_grads = torch.autograd.grad(expected.sum(), reference_inputs)
    torch.testing.assert_close(actual, expected, **_tolerance(dtype))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(
            actual_grad, expected_grad, **_tolerance(dtype))


def _vmamba_inputs(state: int = 4):
    torch.manual_seed(14 + state)
    batch, channels, length, groups = 2, 8, 17, 1
    u = torch.randn(batch, channels, length, device='cuda')
    delta = torch.randn_like(u)
    A = -torch.rand(channels, state, device='cuda')
    B = torch.randn(batch, groups, state, length, device='cuda')
    C = torch.randn_like(B)
    D = torch.randn(channels, device='cuda')
    delta_bias = torch.randn(channels, device='cuda')
    return u, delta, A, B, C, D, delta_bias


@pytest.mark.parametrize(
    ('module_name', 'class_name'),
    [
        ('selective_scan_cuda_core', 'SelectiveScanCore'),
        ('selective_scan_cuda_oflex', 'SelectiveScanOflex'),
    ])
def test_vmamba_core_and_oflex_match_mamba_reference(module_name, class_name):
    importlib.import_module(module_name)
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    from mmpose.models.backbones.Vmamba import csms6s

    inputs = tuple(value.requires_grad_() for value in _vmamba_inputs())
    scan_class = getattr(csms6s, class_name)
    actual = scan_class.apply(*inputs, True, 1, 1, True)
    expected = selective_scan_ref(
        *inputs[:6], None, inputs[6], delta_softplus=True)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=4e-4)
    gradients = torch.autograd.grad(actual.sum(), inputs)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_vmamba_ndstate_forward_backward_matches_state_one_reference():
    module = importlib.import_module('selective_scan_cuda_ndstate')
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    u, delta, A, B, C, D, delta_bias = _vmamba_inputs(state=1)
    A_flat = A[:, 0].contiguous()
    B_flat = B[:, :, 0, :].contiguous()
    C_flat = C[:, :, 0, :].contiguous()
    actual, workspace = module.fwd(
        u, delta, A_flat, B_flat, C_flat, D, delta_bias, True, 1)
    expected = selective_scan_ref(
        u, delta, A, B, C, D, None, delta_bias, True)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=4e-4)
    backward = module.bwd(
        u, delta, A_flat, B_flat, C_flat, D, delta_bias,
        torch.ones_like(actual), workspace, True, 1)
    assert all(torch.isfinite(tensor).all() for tensor in backward)
