"""Hardware-facing numerical probes with fail-closed policies."""

from .fake_quant import (
    ConversionReport, FakeQuantConv2d, FakeQuantLinear, QuantPolicy,
    QuantSpec, convert_for_fake_quant, export_int8_state)
from .observers import ActivationRangeObserver
from .pwl import PiecewiseLinearApproximation, fit_pwl
from .binary_qk import (
    BinaryQKOperationReport, binary_qk_logits, binary_qk_operation_report,
    ste_sign)

__all__ = [
    'ActivationRangeObserver',
    'BinaryQKOperationReport',
    'ConversionReport',
    'FakeQuantConv2d',
    'FakeQuantLinear',
    'QuantPolicy',
    'QuantSpec',
    'PiecewiseLinearApproximation',
    'binary_qk_logits',
    'binary_qk_operation_report',
    'convert_for_fake_quant',
    'export_int8_state',
    'fit_pwl',
    'ste_sign',
]
