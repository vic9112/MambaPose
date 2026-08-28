#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SYSTEMD_SOURCE="${REPO_ROOT}/systemd"
PWL_RUNTIME_ROOT="/home/vicchen/workspace/MambaPose/.worktrees/algo-ssm-quant-pwl-frozen"
ACCOUNT_NAME="$(id -un)"
ACCOUNT_HOME="$(getent passwd "${ACCOUNT_NAME}" | cut -d: -f6)"
USER_UNIT_DIR="${ACCOUNT_HOME}/.config/systemd/user"
CAMPAIGN_ROOT="${PWL_RUNTIME_ROOT}/work_dirs/optimization"
EVIDENCE_DIR="/home/vicchen/workspace/MambaPose/work_dirs/optimization/evidence"
FIXTURE_MODE="${1:-}"
if [[ "${FIXTURE_MODE}" == "--fixture-smoke" && -n "${2:-}" ]]; then
    RENDER_DIR="${2}"
    REMOVE_RENDER_DIR=0
else
    RENDER_DIR="$(mktemp -d)"
    REMOVE_RENDER_DIR=1
fi

cleanup_rendered_units() {
    if [[ "${REMOVE_RENDER_DIR}" == "1" ]]; then
        rm -f \
            "${RENDER_DIR}/mambapose-pwl.service" \
            "${RENDER_DIR}/mambapose-pwl-observer.service" \
            "${RENDER_DIR}/mambapose-pwl-observer.timer"
        rmdir "${RENDER_DIR}"
    fi
}
trap cleanup_rendered_units EXIT

mkdir -p "${RENDER_DIR}"
for name in \
        mambapose-pwl.service \
        mambapose-pwl-observer.service \
        mambapose-pwl-observer.timer; do
    install -m 0644 "${SYSTEMD_SOURCE}/${name}" "${RENDER_DIR}/${name}"
done

systemd-analyze --user verify \
    "${RENDER_DIR}/mambapose-pwl.service" \
    "${RENDER_DIR}/mambapose-pwl-observer.service" \
    "${RENDER_DIR}/mambapose-pwl-observer.timer"

if [[ "${FIXTURE_MODE}" == "--fixture-smoke" ]]; then
    echo "rendered_repo_root=${PWL_RUNTIME_ROOT}"
    echo "fixture systemd verification passed"
    exit 0
fi

if [[ ! -d "${PWL_RUNTIME_ROOT}" ]]; then
    echo "frozen PWL runtime is missing: ${PWL_RUNTIME_ROOT}" >&2
    exit 78
fi
SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
RUNTIME_COMMIT="$(git -C "${PWL_RUNTIME_ROOT}" rev-parse HEAD)"
if [[ "${SOURCE_COMMIT}" != "${RUNTIME_COMMIT}" ]]; then
    echo "frozen PWL runtime commit differs from installer source" >&2
    exit 78
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]] || \
        [[ -n "$(git -C "${PWL_RUNTIME_ROOT}" status --porcelain)" ]] || \
        [[ -n "$(git -C "${PWL_RUNTIME_ROOT}" symbolic-ref -q HEAD)" ]]; then
    echo "installer source and frozen PWL runtime must be clean; runtime detached" >&2
    exit 78
fi
if [[ -L "${CAMPAIGN_ROOT}" ]] || \
        { [[ -e "${CAMPAIGN_ROOT}" ]] && [[ ! -d "${CAMPAIGN_ROOT}" ]]; }; then
    echo "frozen PWL campaign root must be a local directory" >&2
    exit 78
fi
if [[ -e "${CAMPAIGN_ROOT}" ]] && \
        [[ -n "$(find "${CAMPAIGN_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "frozen PWL campaign root must be fresh" >&2
    exit 78
fi

mkdir -p "${USER_UNIT_DIR}" "${EVIDENCE_DIR}"
for name in \
        mambapose-pwl.service \
        mambapose-pwl-observer.service \
        mambapose-pwl-observer.timer; do
    install -m 0644 "${RENDER_DIR}/${name}" "${USER_UNIT_DIR}/${name}"
done

systemctl --user daemon-reload
systemctl --user enable mambapose-pwl.service
systemctl --user enable mambapose-pwl-observer.timer
SERVICE_ENABLED="$(systemctl --user is-enabled mambapose-pwl.service)"
TIMER_ENABLED="$(systemctl --user is-enabled mambapose-pwl-observer.timer)"
LINGER_VALUE="$(loginctl show-user "${ACCOUNT_NAME}" -p Linger --value)"
if [[ "${LINGER_VALUE}" == "yes" ]]; then
    DURABILITY_SCOPE="terminal_disconnect,last_logout,reboot"
else
    DURABILITY_SCOPE="terminal_disconnect_only; last_logout_and_reboot_blocked"
fi

"${REPO_ROOT}/.venv/bin/python" -B - \
    "${EVIDENCE_DIR}/pwl-service-install.json" \
    "${ACCOUNT_NAME}" "${LINGER_VALUE}" "${DURABILITY_SCOPE}" \
    "${SERVICE_ENABLED}" "${TIMER_ENABLED}" \
    "${RENDER_DIR}/mambapose-pwl.service" \
    "${RENDER_DIR}/mambapose-pwl-observer.service" \
    "${RENDER_DIR}/mambapose-pwl-observer.timer" <<'PY'
import hashlib
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

path, account, linger, durability_scope, service_enabled, timer_enabled, *units = sys.argv[1:]
destination = Path(path)
temporary = destination.with_suffix('.json.tmp')
temporary.write_text(json.dumps({
    'installed_at': datetime.now(timezone.utc).isoformat(),
    'account': account,
    'linger': linger,
    'durability_scope': durability_scope,
    'service_enabled': service_enabled,
    'timer_enabled': timer_enabled,
    'network': 'denied',
    'runtime_root': '/home/vicchen/workspace/MambaPose/.worktrees/algo-ssm-quant-pwl-frozen',
    'unit_sha256': {
        Path(unit).name: hashlib.sha256(Path(unit).read_bytes()).hexdigest()
        for unit in units
    },
}, indent=2, sort_keys=True) + '\n')
os.replace(temporary, destination)
PY

echo "linger=${LINGER_VALUE} durability_scope=${DURABILITY_SCOPE}"
if [[ "${LINGER_VALUE}" != "yes" ]]; then
    echo "linger must be enabled before formal campaign launch" >&2
    exit 78
fi
