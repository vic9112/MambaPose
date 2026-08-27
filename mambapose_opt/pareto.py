"""Final three-seed accuracy/hardware Pareto rules."""

from __future__ import annotations

import math
from typing import Mapping

from .binary_operation import (
    canonical_json_sha256, validate_binary_operation_manifest)


def _ap_map(value: Mapping[int, float], *, label: str) -> dict[int, float]:
    if not isinstance(value, Mapping) or set(value) != {0, 1, 2}:
        raise ValueError(f'{label} must contain exact seeds 0, 1, and 2')
    result = {}
    for seed, item in value.items():
        if (
                isinstance(item, bool) or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or not 0.0 <= float(item) <= 100.0):
            raise ValueError(f'{label} AP must be finite percentage points')
        result[seed] = float(item)
    return result


def build_final_pareto_record(
        *, candidate_id: str, candidate_kind: str,
        baseline_ap: Mapping[int, float], candidate_ap: Mapping[int, float],
        operation_manifests: Mapping[int, object],
        ) -> dict:
    """Build the final gate, explicitly admitting the Binary Q/K evidence kind."""
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError('Pareto candidate id must be non-empty')
    if candidate_kind != 'binary-qk':
        raise ValueError('this Pareto evidence builder requires binary-qk')
    baseline = _ap_map(baseline_ap, label='baseline')
    candidate = _ap_map(candidate_ap, label='candidate')
    if not isinstance(operation_manifests, Mapping) or set(
            operation_manifests) != {0, 1, 2}:
        raise ValueError('Binary Pareto operation manifests require seeds 0..2')
    operations = {
        seed: validate_binary_operation_manifest(value)
        for seed, value in operation_manifests.items()
    }
    operation_hashes = {
        canonical_json_sha256(value) for value in operations.values()}
    if len(operation_hashes) != 1:
        raise ValueError('Binary Pareto operation manifests disagree across seeds')
    operation = operations[0]
    drops = {seed: baseline[seed] - candidate[seed] for seed in (0, 1, 2)}
    mean_drop = sum(drops.values()) / 3.0
    max_drop = max(drops.values())
    accuracy_pass = mean_drop < 0.1 and max_drop < 0.3
    hardware_pass = (
        operation['theoretical_qk_multiplications_replaced'] == 6_489_600
        and operation['bitwise_kernel_present'] is False
        and operation['measured_integer_latency'] is False
        and operation['hardware_claim'] == 'none-software-proxy')
    record = {
        'schema_version': 1,
        'artifact_kind': 'mambapose-final-pareto-record',
        'candidate_id': candidate_id,
        'candidate_kind': candidate_kind,
        'seeds': [
            {
                'seed': seed,
                'baseline_ap_points': baseline[seed],
                'candidate_ap_points': candidate[seed],
                'ap_drop_points': drops[seed],
            }
            for seed in (0, 1, 2)
        ],
        'accuracy_gate': {
            'mean_ap_drop_points': mean_drop,
            'max_ap_drop_points': max_drop,
            'mean_limit_exclusive': 0.1,
            'max_limit_exclusive': 0.3,
            'passed': accuracy_pass,
        },
        'hardware_evidence': {
            'kind': 'binary-qk-theoretical-operation-replacement',
            'theoretical_qk_multiplications_replaced': 6_489_600,
            'operation_manifest_sha256': operation_hashes.pop(),
            'bitwise_kernel_present': False,
            'measured_integer_latency': False,
            'speedup_claim': 'none-software-proxy',
            'passed': hardware_pass,
        },
        'decision': (
            'pareto-eligible'
            if accuracy_pass and hardware_pass else 'rejected'),
    }
    validate_final_pareto_record(record)
    return record


def validate_final_pareto_record(value: object) -> Mapping:
    """Validate a published final Pareto record including Binary Q/K."""
    if not isinstance(value, Mapping) or set(value) != {
            'schema_version', 'artifact_kind', 'candidate_id', 'candidate_kind',
            'seeds', 'accuracy_gate', 'hardware_evidence', 'decision'}:
        raise ValueError('final Pareto record has invalid fields')
    if (
            value['schema_version'] != 1
            or value['artifact_kind'] != 'mambapose-final-pareto-record'
            or not isinstance(value['candidate_id'], str)
            or not value['candidate_id']
            or value['candidate_kind'] != 'binary-qk'
            or value['decision'] not in {'pareto-eligible', 'rejected'}):
        raise ValueError('final Pareto record identity is invalid')
    seeds = value['seeds']
    if (
            not isinstance(seeds, list) or len(seeds) != 3
            or [row.get('seed') for row in seeds] != [0, 1, 2]
            or any(
                not isinstance(row, Mapping) or set(row) != {
                    'seed', 'baseline_ap_points', 'candidate_ap_points',
                    'ap_drop_points'}
                for row in seeds)):
        raise ValueError('final Pareto seed evidence is invalid')
    drops = []
    for row in seeds:
        numbers = (
            row['baseline_ap_points'], row['candidate_ap_points'],
            row['ap_drop_points'])
        if any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                or not math.isfinite(float(item)) for item in numbers):
            raise ValueError('final Pareto seed AP evidence is invalid')
        observed = float(row['baseline_ap_points']) - float(
            row['candidate_ap_points'])
        if not math.isclose(
                observed, float(row['ap_drop_points']),
                rel_tol=0.0, abs_tol=1e-9):
            raise ValueError('final Pareto seed AP drop disagrees')
        drops.append(observed)
    expected_mean = sum(drops) / 3.0
    expected_max = max(drops)
    accuracy = value['accuracy_gate']
    hardware = value['hardware_evidence']
    if not isinstance(accuracy, Mapping) or set(accuracy) != {
            'mean_ap_drop_points', 'max_ap_drop_points',
            'mean_limit_exclusive', 'max_limit_exclusive', 'passed'}:
        raise ValueError('final Pareto accuracy gate fields are invalid')
    summary = (
        accuracy['mean_ap_drop_points'], accuracy['max_ap_drop_points'])
    if any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            or not math.isfinite(float(item)) for item in summary):
        raise ValueError('final Pareto accuracy summary is invalid')
    if (
            not math.isclose(
                float(accuracy['mean_ap_drop_points']), expected_mean,
                rel_tol=0.0, abs_tol=1e-9)
            or not math.isclose(
                float(accuracy['max_ap_drop_points']), expected_max,
                rel_tol=0.0, abs_tol=1e-9)):
        raise ValueError(
            'final Pareto accuracy summary disagrees with seed evidence')
    expected_accuracy_pass = expected_mean < 0.1 and expected_max < 0.3
    accuracy_pass = (
        accuracy.get('mean_limit_exclusive') == 0.1
        and accuracy.get('max_limit_exclusive') == 0.3
        and accuracy.get('passed') is expected_accuracy_pass)
    hardware_pass = (
        isinstance(hardware, Mapping)
        and set(hardware) == {
            'kind', 'theoretical_qk_multiplications_replaced',
            'operation_manifest_sha256', 'bitwise_kernel_present',
            'measured_integer_latency', 'speedup_claim', 'passed'}
        and hardware.get('kind') == (
            'binary-qk-theoretical-operation-replacement')
        and hardware.get('theoretical_qk_multiplications_replaced') == 6_489_600
        and isinstance(hardware.get('operation_manifest_sha256'), str)
        and len(hardware['operation_manifest_sha256']) == 64
        and all(
            character in '0123456789abcdef'
            for character in hardware['operation_manifest_sha256'])
        and hardware.get('bitwise_kernel_present') is False
        and hardware.get('measured_integer_latency') is False
        and hardware.get('speedup_claim') == 'none-software-proxy'
        and hardware.get('passed') is True)
    if not hardware_pass:
        raise ValueError('final Pareto hardware evidence is invalid')
    if not accuracy_pass:
        if value['decision'] != 'rejected':
            raise ValueError('final Pareto decision overstates failed evidence')
    elif value['decision'] != 'pareto-eligible':
        raise ValueError('final Pareto decision disagrees with passed evidence')
    return value
