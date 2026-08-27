"""CPU-only parameter and module inventory for optimization candidates."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from torch import nn


@dataclass(frozen=True)
class ParameterInventory:
    total: int
    trainable: int
    bytes_by_dtype: Mapping[str, int]
    by_prefix: Mapping[str, int]


@dataclass(frozen=True)
class ModuleRecord:
    name: str
    kind: str
    parameters: int
    hazard: str | None


_HAZARDS = {
    'PoseInteraction': 'dynamic top-k',
    'Attention': 'attention score and softmax',
    'CrossScan': 'custom VMamba scan',
    'SelectiveScan': 'custom VMamba scan',
    'CrossMerge': 'custom VMamba scan',
}


def count_parameters(model: nn.Module) -> ParameterInventory:
    """Return deduplicated parameters, grouped without executing the model."""
    total = 0
    trainable = 0
    bytes_by_dtype: dict[str, int] = {}
    by_prefix: dict[str, int] = {}

    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
        dtype = str(parameter.dtype)
        bytes_by_dtype[dtype] = bytes_by_dtype.get(dtype, 0) + (
            count * parameter.element_size())
        prefix = name.partition('.')[0] or '<root>'
        by_prefix[prefix] = by_prefix.get(prefix, 0) + count

    return ParameterInventory(
        total=total,
        trainable=trainable,
        bytes_by_dtype=MappingProxyType(bytes_by_dtype),
        by_prefix=MappingProxyType(by_prefix),
    )


def collect_module_inventory(model: nn.Module) -> tuple[ModuleRecord, ...]:
    """Report modules and explicit hazards, without estimating custom-op FLOPs."""
    records = []
    for name, module in model.named_modules():
        kind = type(module).__name__
        records.append(ModuleRecord(
            name=name,
            kind=kind,
            parameters=sum(parameter.numel() for parameter in module.parameters(
                recurse=False)),
            hazard=_HAZARDS.get(kind),
        ))
    return tuple(records)
