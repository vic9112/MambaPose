"""Fail-closed, process-aware exclusivity for every CUDA stage."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import secrets
import subprocess
import threading
from typing import Callable, Collection, Iterator


PROC_ROOT = Path('/proc')


class ExternalGpuContention(RuntimeError):
    """Raised when GPU ownership cannot be proved exclusive."""


class ConcurrentCudaStage(RuntimeError):
    """Raised when another stage already holds the common CUDA lock."""


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    used_memory_mib: int
    command: str


@dataclass(frozen=True)
class GpuLease:
    stage_id: str
    pid: int
    boot_id: str
    timestamp: str
    device_index: int
    allowed_pids: tuple[int, ...]
    lease_id: str


def _write_lease(stream, lease: GpuLease) -> None:
    payload = (json.dumps(asdict(lease), sort_keys=True) + '\n').encode('utf-8')
    os.pwrite(stream.fileno(), payload, 0)
    os.ftruncate(stream.fileno(), len(payload))
    os.fsync(stream.fileno())


def _boot_id() -> str:
    try:
        return (PROC_ROOT / 'sys/kernel/random/boot_id').read_text().strip()
    except OSError:
        # A missing boot identity prevents a durable owner record.
        raise ExternalGpuContention('cannot read the system boot ID')


def _command(pid: int) -> str:
    try:
        raw = (PROC_ROOT / str(pid) / 'cmdline').read_bytes()
    except OSError:
        return '<unavailable>'
    return ' '.join(
        item.decode(errors='replace') for item in raw.split(b'\0') if item)


def query_compute_processes(device_index: int) -> tuple[GpuProcess, ...]:
    """Return compute owners reported by ``nvidia-smi`` for one device."""
    try:
        result = subprocess.run(
            [
                'nvidia-smi', f'--id={device_index}',
                '--query-compute-apps=pid,used_memory',
                '--format=csv,noheader,nounits',
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ExternalGpuContention(
            f'cannot establish GPU {device_index} ownership: {error}') from error

    processes: list[GpuProcess] = []
    for line in result.stdout.splitlines():
        if not line.strip() or line.strip().lower().startswith('no running'):
            continue
        parts = [part.strip() for part in line.split(',')]
        try:
            pid = int(parts[0])
            memory = int(parts[1].removesuffix('MiB').strip())
        except (IndexError, ValueError) as error:
            raise ExternalGpuContention(
                f'cannot parse GPU {device_index} ownership row: {line!r}') from error
        processes.append(GpuProcess(pid, memory, _command(pid)))
    return tuple(processes)


def _parent_pid(stat_path: Path) -> int | None:
    try:
        value = stat_path.read_text()
        # comm may contain spaces and closing parentheses; state and PPID begin
        # only after the final closing parenthesis.
        fields = value[value.rfind(')') + 1:].split()
        return int(fields[1])
    except (OSError, ValueError, IndexError):
        return None


def controller_process_tree(roots: Collection[int]) -> tuple[int, ...]:
    """Resolve the live transitive process tree rooted at controller PIDs."""
    allowed = {int(pid) for pid in roots if int(pid) > 0}
    parents: dict[int, int] = {}
    try:
        entries = tuple(PROC_ROOT.iterdir())
    except OSError as error:
        raise ExternalGpuContention(
            f'cannot inspect controller descendants: {error}') from error
    for entry in entries:
        if not entry.name.isdigit():
            continue
        parent = _parent_pid(entry / 'stat')
        if parent is not None:
            parents[int(entry.name)] = parent
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in allowed and pid not in allowed:
                allowed.add(pid)
                changed = True
    return tuple(sorted(allowed))


@contextmanager
def exclusive_cuda_stage(
        lock_path: Path, device_index: int, allowed_pids: Collection[int],
        *, stage_id: str = 'cuda-stage', heartbeat_interval: float = 30.0,
        clock: Callable[[], datetime] | None = None,
        heartbeat_wait: Callable[[threading.Event, float], bool] | None = None,
        ) -> Iterator[GpuLease]:
    """Hold the shared lock and reject every non-controller GPU owner."""
    if (
            isinstance(heartbeat_interval, bool)
            or not isinstance(heartbeat_interval, (int, float))
            or heartbeat_interval <= 0):
        raise ValueError('heartbeat_interval must be positive')
    now = clock if clock is not None else lambda: datetime.now(timezone.utc)
    wait = (
        heartbeat_wait if heartbeat_wait is not None
        else lambda stop, interval: stop.wait(interval))
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = lock_path.open('r+', encoding='utf-8')
    except FileNotFoundError:
        try:
            stream = lock_path.open('x+', encoding='utf-8')
        except FileExistsError:
            stream = lock_path.open('r+', encoding='utf-8')
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ConcurrentCudaStage(
                f'another CUDA stage owns {lock_path}') from error

        process_tree = controller_process_tree(allowed_pids)
        permitted = set(process_tree)
        owners = query_compute_processes(device_index)
        external = tuple(owner for owner in owners if owner.pid not in permitted)
        if external:
            rendered = ', '.join(
                f'pid={owner.pid} memory={owner.used_memory_mib}MiB '
                f'command={owner.command!r}' for owner in external)
            raise ExternalGpuContention(
                f'external compute owner on GPU {device_index}: {rendered}')

        lease = GpuLease(
            stage_id=stage_id,
            pid=os.getpid(),
            boot_id=_boot_id(),
            timestamp=now().isoformat(),
            device_index=device_index,
            allowed_pids=process_tree,
            lease_id=secrets.token_hex(32),
        )
        _write_lease(stream, lease)
        stop = threading.Event()
        heartbeat_errors: list[BaseException] = []

        def refresh() -> None:
            try:
                while not wait(stop, float(heartbeat_interval)):
                    refreshed = replace(lease, timestamp=now().isoformat())
                    _write_lease(stream, refreshed)
            except BaseException as error:
                heartbeat_errors.append(error)
                stop.set()

        heartbeat = threading.Thread(
            target=refresh, name=f'gpu-lease-{stage_id}', daemon=True)
        heartbeat.start()
        body_failed = False
        try:
            yield lease
        except BaseException:
            body_failed = True
            raise
        finally:
            stop.set()
            heartbeat.join(timeout=max(1.0, float(heartbeat_interval) + 1.0))
            if heartbeat.is_alive() and not body_failed:
                raise RuntimeError('GPU lease heartbeat did not stop')
            if heartbeat_errors and not body_failed:
                raise RuntimeError('GPU lease heartbeat failed') from heartbeat_errors[0]
    finally:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()
