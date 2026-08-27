import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import threading

import pytest


def test_external_gpu_owner_blocks_admission(tmp_path, monkeypatch):
    from mambapose_opt import gpu_guard
    from mambapose_opt.gpu_guard import (
        ExternalGpuContention,
        GpuProcess,
        exclusive_cuda_stage,
    )

    monkeypatch.setattr(gpu_guard, 'query_compute_processes', lambda _: (
        GpuProcess(pid=9001, used_memory_mib=4096, command='foreign.py'),
    ))

    with pytest.raises(ExternalGpuContention, match='9001'):
        with exclusive_cuda_stage(
                tmp_path / 'gpu.lock', 0, {os.getpid()},
                stage_id='train:fixture'):
            pass


def test_same_lock_serializes_stage_kinds(tmp_path, monkeypatch):
    from mambapose_opt import gpu_guard
    from mambapose_opt.gpu_guard import (
        ConcurrentCudaStage,
        exclusive_cuda_stage,
    )

    monkeypatch.setattr(gpu_guard, 'query_compute_processes', lambda _: ())

    with exclusive_cuda_stage(
            tmp_path / 'gpu.lock', 0, {os.getpid()}, stage_id='train'):
        with pytest.raises(ConcurrentCudaStage):
            with exclusive_cuda_stage(
                    tmp_path / 'gpu.lock', 0, {os.getpid()},
                    stage_id='latency'):
                pass


def test_controller_descendant_is_derived_from_proc_and_allowed(
        tmp_path, monkeypatch):
    from mambapose_opt import gpu_guard
    from mambapose_opt.gpu_guard import GpuProcess, exclusive_cuda_stage

    proc_root = tmp_path / 'proc'
    (proc_root / '100').mkdir(parents=True)
    (proc_root / '101').mkdir()
    (proc_root / '202').mkdir()
    (proc_root / 'sys/kernel/random').mkdir(parents=True)
    (proc_root / 'sys/kernel/random/boot_id').write_text('fixture-boot-id\n')
    (proc_root / '100' / 'stat').write_text('100 (controller) S 1 0 0 0\n')
    (proc_root / '101' / 'stat').write_text('101 (worker thread) S 100 0 0 0\n')
    (proc_root / '202' / 'stat').write_text('202 (unrelated) S 1 0 0 0\n')
    monkeypatch.setattr(gpu_guard, 'PROC_ROOT', proc_root)
    monkeypatch.setattr(gpu_guard, 'query_compute_processes', lambda _: (
        GpuProcess(pid=101, used_memory_mib=1024, command='worker'),
    ))

    with exclusive_cuda_stage(
            tmp_path / 'gpu.lock', 0, {100}, stage_id='evaluate') as lease:
        assert lease.allowed_pids == (100, 101)
        assert lease.stage_id == 'evaluate'
        assert lease.pid == os.getpid()


def test_query_compute_processes_uses_nvidia_smi_without_shell(
        monkeypatch, tmp_path):
    from mambapose_opt import gpu_guard

    proc_root = tmp_path / 'proc'
    (proc_root / '77').mkdir(parents=True)
    (proc_root / '77' / 'cmdline').write_bytes(b'python\0train.py\0')
    monkeypatch.setattr(gpu_guard, 'PROC_ROOT', proc_root)
    seen = {}

    class Result:
        stdout = '77, 2048 MiB\n'

    def fake_run(argv, **kwargs):
        seen['argv'] = argv
        seen['kwargs'] = kwargs
        return Result()

    monkeypatch.setattr(gpu_guard.subprocess, 'run', fake_run)

    processes = gpu_guard.query_compute_processes(2)

    assert processes == (
        gpu_guard.GpuProcess(77, 2048, 'python train.py'),
    )
    assert seen['argv'] == [
        'nvidia-smi', '--id=2',
        '--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits',
    ]
    assert seen['kwargs']['shell'] is False
    assert seen['kwargs']['check'] is True


def test_lease_record_contains_durable_owner_identity(tmp_path, monkeypatch):
    import json

    from mambapose_opt import gpu_guard
    from mambapose_opt.gpu_guard import exclusive_cuda_stage

    monkeypatch.setattr(gpu_guard, 'query_compute_processes', lambda _: ())
    lock_path = tmp_path / 'gpu.lock'

    with exclusive_cuda_stage(
            lock_path, 0, {os.getpid()}, stage_id='profile:full-s-v1') as lease:
        record = json.loads(lock_path.read_text())
        assert record['stage_id'] == 'profile:full-s-v1'
        assert record['pid'] == os.getpid()
        assert record['boot_id'] == lease.boot_id
        assert record['timestamp'] == lease.timestamp
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == (
            record['boot_id'])


def test_lease_heartbeat_refreshes_same_locked_inode_after_301_seconds(
        tmp_path, monkeypatch):
    from mambapose_opt import gpu_guard
    from mambapose_opt.gpu_guard import exclusive_cuda_stage

    monkeypatch.setattr(gpu_guard, 'query_compute_processes', lambda _: ())
    start = datetime(2026, 8, 27, tzinfo=timezone.utc)
    clock = iter((start, start + timedelta(seconds=301)))
    refresh_observed = threading.Event()
    waits = 0
    initial_inode = []

    def heartbeat_wait(stop, interval):
        nonlocal waits
        waits += 1
        if waits == 1:
            initial_inode.append(lock_path.stat().st_ino)
            return False
        refresh_observed.set()
        return stop.wait(1.0)

    lock_path = tmp_path / 'gpu.lock'
    with exclusive_cuda_stage(
            lock_path, 0, {os.getpid()}, stage_id='fixture:latency',
            heartbeat_interval=30.0, clock=lambda: next(clock),
            heartbeat_wait=heartbeat_wait):
        assert refresh_observed.wait(1.0)
        record = json.loads(lock_path.read_text())
        assert datetime.fromisoformat(record['timestamp']) == (
            start + timedelta(seconds=301))
        assert lock_path.stat().st_ino == initial_inode[0]
