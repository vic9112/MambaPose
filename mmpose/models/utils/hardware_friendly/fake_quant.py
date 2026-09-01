"""Explicit fake-quant wrappers and conversion policy."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class QuantSpec:
    """A symmetric int8 simulation contract with FP32 master weights."""

    enabled: bool = True
    weight_bits: int = 8
    activation_bits: int | None = None
    activation_scale: float | tuple[float, ...] | None = None
    per_output_channel: bool = True
    symmetric: bool = True

    def __post_init__(self) -> None:
        if self.weight_bits != 8:
            raise ValueError('Stage A supports symmetric int8 weights only')
        if self.activation_bits not in {None, 8}:
            raise ValueError('Stage A supports optional symmetric int8 activations')
        if not self.symmetric:
            raise ValueError('Stage A requires symmetric quantization')
        if not self.per_output_channel:
            raise ValueError('Stage A requires per-output-channel weight scales')
        if self.activation_bits is not None and self.activation_scale is None:
            raise ValueError('W8A8 requires a measured activation scale')
        scales = (
            (self.activation_scale,)
            if isinstance(self.activation_scale, (int, float))
            else self.activation_scale)
        if scales is not None and (
                not scales or any(
                    not isinstance(scale, (int, float))
                    or not math.isfinite(scale) or scale <= 0
                    for scale in scales)):
            raise ValueError('activation scales must be positive finite numbers')


@dataclass(frozen=True)
class QuantPolicy:
    """Exact fully-qualified module roles admitted or explicitly denied."""

    allow: tuple[str, ...]
    deny: tuple[str, ...] = ()
    spec: QuantSpec = QuantSpec()
    role_specs: tuple[tuple[str, QuantSpec], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'allow', tuple(self.allow))
        object.__setattr__(self, 'deny', tuple(self.deny))
        object.__setattr__(self, 'role_specs', tuple(self.role_specs))
        names = self.allow + self.deny
        if any(not isinstance(name, str) or not name or name.startswith('.')
               or name.endswith('.') or '..' in name for name in names):
            raise ValueError(
                'policy targets must be exact fully-qualified module roles')
        if len(set(self.allow)) != len(self.allow):
            raise ValueError('allowed module roles must be unique')
        if len(set(self.deny)) != len(self.deny):
            raise ValueError('denied module roles must be unique')
        overlap = set(self.allow) & set(self.deny)
        if overlap:
            raise ValueError(
                f'module roles cannot be both allowed and denied: {sorted(overlap)}')
        role_names = tuple(name for name, _spec in self.role_specs)
        if role_names and (len(set(role_names)) != len(role_names)
                           or set(role_names) != set(self.allow)):
            raise ValueError('per-role quant specs must exactly cover allowed roles')
        if any(not isinstance(spec, QuantSpec)
               for _name, spec in self.role_specs):
            raise ValueError('per-role quant specs must contain QuantSpec values')

    def spec_for(self, role: str) -> QuantSpec:
        return dict(self.role_specs).get(role, self.spec)


@dataclass(frozen=True)
class ConversionReport:
    converted: tuple[str, ...]
    skipped: tuple[str, ...]
    original_weight_bytes: int
    simulated_weight_bytes: int
    simulated_coverage: float
    simulation_only: bool = True
    integer_kernel_latency_claimed: bool = False


def _weight_scales(weight: Tensor) -> Tensor:
    flat = weight.detach().to(torch.float32).flatten(1)
    maximum = flat.abs().amax(dim=1)
    return torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127.0)


def _fake_quantize_weight(weight: Tensor) -> Tensor:
    scales = _weight_scales(weight)
    view = (weight.shape[0],) + (1,) * (weight.ndim - 1)
    quantized = torch.round(weight / scales.view(view)).clamp(-127, 127)
    dequantized = quantized * scales.view(view)
    return weight + (dequantized.to(weight.dtype) - weight).detach()


def _fake_quantize_activation(
        value: Tensor, spec: QuantSpec, *, channel_axis: int = -1) -> Tensor:
    if spec.activation_bits is None:
        return value
    scale = torch.as_tensor(
        spec.activation_scale, dtype=value.dtype, device=value.device)
    if scale.ndim == 1:
        normalized_axis = channel_axis % value.ndim
        if scale.numel() != value.shape[normalized_axis]:
            raise ValueError('activation scale count must match the channel axis')
        shape = [1] * value.ndim
        shape[normalized_axis] = -1
        scale = scale.view(shape)
    quantized = torch.round(value / scale).clamp(-127, 127)
    dequantized = quantized * scale
    return value + (dequantized - value).detach()


class FakeQuantLinear(nn.Linear):
    """Linear fake quant with unchanged parameter names and FP32 masters."""

    def __init__(self, *args, spec: QuantSpec | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = spec or QuantSpec()

    @classmethod
    def from_float(cls, module: nn.Linear, spec: QuantSpec) -> 'FakeQuantLinear':
        if not isinstance(module, nn.Linear):
            raise TypeError('FakeQuantLinear requires nn.Linear')
        proxy = cls(
            module.in_features, module.out_features,
            bias=module.bias is not None, device=module.weight.device,
            dtype=module.weight.dtype, spec=spec)
        proxy.weight = module.weight
        proxy.bias = module.bias
        proxy._channel_first_2d = type(module).__name__ == 'Linear2d'
        proxy.train(module.training)
        return proxy

    def weight_scales(self) -> Tensor:
        return _weight_scales(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        if not self.spec.enabled:
            if getattr(self, '_channel_first_2d', False):
                return F.conv2d(input, self.weight[:, :, None, None], self.bias)
            return F.linear(input, self.weight, self.bias)
        input = _fake_quantize_activation(
            input, self.spec,
            channel_axis=(1 if getattr(self, '_channel_first_2d', False)
                          else -1))
        if getattr(self, '_channel_first_2d', False):
            return F.conv2d(
                input, _fake_quantize_weight(self.weight)[:, :, None, None],
                self.bias)
        return F.linear(input, _fake_quantize_weight(self.weight), self.bias)


class FakeQuantConv2d(nn.Conv2d):
    """Conv2d fake quant with unchanged parameter names and FP32 masters."""

    def __init__(self, *args, spec: QuantSpec | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = spec or QuantSpec()

    @classmethod
    def from_float(cls, module: nn.Conv2d, spec: QuantSpec) -> 'FakeQuantConv2d':
        if not isinstance(module, nn.Conv2d):
            raise TypeError('FakeQuantConv2d requires nn.Conv2d')
        proxy = cls(
            module.in_channels, module.out_channels, module.kernel_size,
            stride=module.stride, padding=module.padding,
            dilation=module.dilation, groups=module.groups,
            bias=module.bias is not None, padding_mode=module.padding_mode,
            device=module.weight.device, dtype=module.weight.dtype, spec=spec)
        proxy.weight = module.weight
        proxy.bias = module.bias
        proxy.train(module.training)
        return proxy

    def weight_scales(self) -> Tensor:
        return _weight_scales(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        if not self.spec.enabled:
            return self._conv_forward(input, self.weight, self.bias)
        input = _fake_quantize_activation(input, self.spec, channel_axis=1)
        return self._conv_forward(
            input, _fake_quantize_weight(self.weight), self.bias)


def _parent_and_leaf(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, leaf = name.rpartition('.')
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, leaf


def convert_for_fake_quant(
        model: nn.Module, policy: QuantPolicy) -> ConversionReport:
    """Mutate only exact admitted roles and report simulated storage coverage."""
    modules = dict(model.named_modules())
    requested = policy.allow + policy.deny
    missing = tuple(name for name in requested if name not in modules)
    if missing:
        raise ValueError(f'policy targets are missing: {missing}')
    unsupported = tuple(
        name for name in policy.allow
        if not isinstance(modules[name], (nn.Linear, nn.Conv2d))
        or isinstance(modules[name], (FakeQuantLinear, FakeQuantConv2d)))
    if unsupported:
        raise ValueError(f'policy targets use unsupported module roles: {unsupported}')

    eligible = tuple(
        (name, module) for name, module in modules.items()
        if name and isinstance(module, (nn.Linear, nn.Conv2d))
        and not isinstance(module, (FakeQuantLinear, FakeQuantConv2d)))
    total_eligible = sum(module.weight.numel() for _, module in eligible)
    converted: list[str] = []
    original_bytes = 0
    simulated_bytes = 0
    converted_weights = 0
    for name in policy.allow:
        module = modules[name]
        wrapper = (
            FakeQuantLinear.from_float(module, policy.spec_for(name))
            if isinstance(module, nn.Linear)
            else FakeQuantConv2d.from_float(module, policy.spec_for(name)))
        parent, leaf = _parent_and_leaf(model, name)
        setattr(parent, leaf, wrapper)
        elements = module.weight.numel()
        original_bytes += elements * module.weight.element_size()
        simulated_bytes += elements + module.weight.shape[0] * 4
        converted_weights += elements
        converted.append(name)
    return ConversionReport(
        converted=tuple(converted),
        skipped=tuple(policy.deny),
        original_weight_bytes=original_bytes,
        simulated_weight_bytes=simulated_bytes,
        simulated_coverage=(converted_weights / total_eligible
                            if total_eligible else 0.0),
    )


def export_int8_state(
        model: nn.Module, report: ConversionReport) -> dict[str, dict[str, Tensor]]:
    """Pack admitted weights only; this is storage evidence, not a kernel."""
    modules = dict(model.named_modules())
    packed: dict[str, dict[str, Tensor]] = {}
    for name in report.converted:
        module = modules.get(name)
        if not isinstance(module, (FakeQuantLinear, FakeQuantConv2d)):
            raise ValueError(f'converted role is no longer fake-quantized: {name}')
        scales = module.weight_scales().detach().to(torch.float32).cpu()
        view = (module.weight.shape[0],) + (1,) * (module.weight.ndim - 1)
        values = torch.round(
            module.weight.detach().cpu() / scales.view(view)).clamp(
                -127, 127).to(torch.int8)
        record = {'values': values, 'scales': scales}
        if module.bias is not None:
            record['bias'] = module.bias.detach().to(torch.float32).cpu()
        packed[f'{name}.weight'] = record
    if tuple(name.removesuffix('.weight') for name in packed) != report.converted:
        raise ValueError('exported roles disagree with conversion report')
    return packed
