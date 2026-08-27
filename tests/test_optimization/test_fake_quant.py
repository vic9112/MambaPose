from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn


def test_disabled_fake_quant_linear_is_key_and_output_bit_exact():
    from mmpose.models.utils.hardware_friendly import FakeQuantLinear, QuantSpec

    linear = nn.Linear(8, 4).eval()
    proxy = FakeQuantLinear.from_float(linear, QuantSpec(enabled=False))
    x = torch.randn(3, 8)

    assert proxy.state_dict().keys() == linear.state_dict().keys()
    torch.testing.assert_close(proxy(x), linear(x), rtol=0, atol=0)


def test_disabled_fake_quant_conv_is_key_and_output_bit_exact():
    from mmpose.models.utils.hardware_friendly import FakeQuantConv2d, QuantSpec

    conv = nn.Conv2d(3, 4, 3, padding=1).eval()
    proxy = FakeQuantConv2d.from_float(conv, QuantSpec(enabled=False))
    x = torch.randn(2, 3, 5, 5)

    assert proxy.state_dict().keys() == conv.state_dict().keys()
    torch.testing.assert_close(proxy(x), conv(x), rtol=0, atol=0)


def test_conv_activation_scales_apply_on_nchw_channel_axis():
    from mmpose.models.utils.hardware_friendly import FakeQuantConv2d, QuantSpec

    conv = nn.Conv2d(3, 4, 1)
    proxy = FakeQuantConv2d.from_float(
        conv, QuantSpec(activation_bits=8,
                        activation_scale=(0.1, 0.2, 0.3)))

    assert proxy(torch.randn(2, 3, 5, 7)).shape == (2, 4, 5, 7)


def test_fake_quant_keeps_fp32_master_and_uses_per_output_channel_scales():
    from mmpose.models.utils.hardware_friendly import FakeQuantLinear, QuantSpec

    linear = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, -0.5, 0.25],
                                          [8.0, -4.0, 2.0]]))
    proxy = FakeQuantLinear.from_float(linear, QuantSpec())

    assert proxy.weight is linear.weight
    assert proxy.weight.dtype == torch.float32
    assert proxy.weight_scales().tolist() == pytest.approx([1 / 127, 8 / 127])
    proxy(torch.ones(1, 3)).sum().backward()
    assert proxy.weight.grad is not None


def test_quant_contracts_are_immutable():
    from mmpose.models.utils.hardware_friendly import (
        ConversionReport, QuantPolicy, QuantSpec)

    values = [
        (QuantSpec(), 'enabled'),
        (QuantPolicy(allow=('stem',)), 'allow'),
        (ConversionReport(converted=(), skipped=(), original_weight_bytes=0,
                          simulated_weight_bytes=0, simulated_coverage=0.0),
         'converted'),
    ]
    for value, field in values:
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, None)


def test_conversion_is_explicit_and_reports_linear_and_conv_coverage():
    from mmpose.models.utils.hardware_friendly import (
        FakeQuantConv2d, FakeQuantLinear, QuantPolicy,
        convert_for_fake_quant)

    model = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3), nn.Conv2d(3, 2, 1))
    report = convert_for_fake_quant(
        model, QuantPolicy(allow=('0', '2'), deny=('1',)))

    assert isinstance(model[0], FakeQuantLinear)
    assert isinstance(model[2], FakeQuantConv2d)
    assert isinstance(model[1], nn.LayerNorm)
    assert report.converted == ('0', '2')
    assert report.skipped == ('1',)
    assert report.original_weight_bytes == (12 + 6) * 4
    assert report.simulated_weight_bytes == (12 + 6) + (3 + 2) * 4
    assert report.simulated_coverage == 1.0


@pytest.mark.parametrize('allow, match', [
    (('missing',), 'missing'),
    (('0',), 'unsupported'),
    (('',), 'fully-qualified'),
])
def test_conversion_rejects_missing_unsupported_or_ambiguous_targets(
        allow, match):
    from mmpose.models.utils.hardware_friendly import (
        QuantPolicy, convert_for_fake_quant)

    model = nn.Sequential(nn.LayerNorm(4))
    with pytest.raises(ValueError, match=match):
        convert_for_fake_quant(model, QuantPolicy(allow=allow))


def test_conversion_rejects_allow_deny_overlap():
    from mmpose.models.utils.hardware_friendly import QuantPolicy

    with pytest.raises(ValueError, match='both allowed and denied'):
        QuantPolicy(allow=('head',), deny=('head',))


def test_exported_int8_state_is_separate_from_fp32_master_storage():
    from mmpose.models.utils.hardware_friendly import (
        QuantPolicy, convert_for_fake_quant, export_int8_state)

    model = nn.Sequential(nn.Linear(4, 3))
    master = model[0].weight.detach().clone()
    report = convert_for_fake_quant(model, QuantPolicy(allow=('0',)))
    exported = export_int8_state(model, report)

    assert exported['0.weight']['values'].dtype == torch.int8
    assert exported['0.weight']['scales'].shape == (3,)
    assert model[0].weight.dtype == torch.float32
    torch.testing.assert_close(model[0].weight, master)
