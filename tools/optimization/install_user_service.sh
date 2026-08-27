#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SYSTEMD_SOURCE="${REPO_ROOT}/systemd"
ACCOUNT_NAME="$(id -un)"
ACCOUNT_HOME="$(getent passwd "${ACCOUNT_NAME}" | cut -d: -f6)"
USER_UNIT_DIR="${ACCOUNT_HOME}/.config/systemd/user"
EVIDENCE_DIR="${REPO_ROOT}/work_dirs/optimization/evidence"
SOURCE_CHECKOUT="/home/vicchen/workspace/MambaPose"
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
            "${RENDER_DIR}/mambapose-optimization.service" \
            "${RENDER_DIR}/mambapose-optimization-observer.service" \
            "${RENDER_DIR}/mambapose-optimization-observer.timer"
        rmdir "${RENDER_DIR}"
    fi
}
trap cleanup_rendered_units EXIT

mkdir -p "${RENDER_DIR}"
"${REPO_ROOT}/.venv/bin/python" - \
    "${SYSTEMD_SOURCE}" "${RENDER_DIR}" \
    "${SOURCE_CHECKOUT}" "${REPO_ROOT}" <<'PY'
from pathlib import Path
import sys

source, destination, old_root, new_root = sys.argv[1:]
for name in (
        'mambapose-optimization.service',
        'mambapose-optimization-observer.service',
        'mambapose-optimization-observer.timer'):
    value = (Path(source) / name).read_text(encoding='utf-8')
    if name.endswith('.service') and old_root not in value:
        raise SystemExit(f'unit lacks checkout provenance marker: {name}')
    (Path(destination) / name).write_text(
        value.replace(old_root, new_root), encoding='utf-8')
PY

systemd-analyze --user verify \
    "${RENDER_DIR}/mambapose-optimization.service" \
    "${RENDER_DIR}/mambapose-optimization-observer.service" \
    "${RENDER_DIR}/mambapose-optimization-observer.timer"

if [[ "${FIXTURE_MODE}" == "--fixture-smoke" ]]; then
    echo "rendered_repo_root=${REPO_ROOT}"
    echo "fixture systemd verification passed"
    exit 0
fi

mkdir -p "${USER_UNIT_DIR}" "${EVIDENCE_DIR}"
install -m 0644 "${RENDER_DIR}/mambapose-optimization.service" \
    "${USER_UNIT_DIR}/mambapose-optimization.service"
install -m 0644 "${RENDER_DIR}/mambapose-optimization-observer.service" \
    "${USER_UNIT_DIR}/mambapose-optimization-observer.service"
install -m 0644 "${RENDER_DIR}/mambapose-optimization-observer.timer" \
    "${USER_UNIT_DIR}/mambapose-optimization-observer.timer"

systemctl --user daemon-reload
systemctl --user enable mambapose-optimization.service
systemctl --user enable mambapose-optimization-observer.timer
SERVICE_ENABLED="$(systemctl --user is-enabled mambapose-optimization.service)"
TIMER_ENABLED="$(systemctl --user is-enabled mambapose-optimization-observer.timer)"
LINGER_VALUE="$(loginctl show-user "${ACCOUNT_NAME}" -p Linger --value)"
if [[ "${LINGER_VALUE}" == "yes" ]]; then
    DURABILITY_SCOPE="terminal_disconnect,last_logout,reboot"
else
    DURABILITY_SCOPE="terminal_disconnect_only; last_logout_and_reboot_blocked"
fi

"${REPO_ROOT}/.venv/bin/python" - \
    "${EVIDENCE_DIR}/service-install.json" \
    "${ACCOUNT_NAME}" "${LINGER_VALUE}" "${DURABILITY_SCOPE}" \
    "${SERVICE_ENABLED}" "${TIMER_ENABLED}" \
    "${RENDER_DIR}/mambapose-optimization.service" \
    "${RENDER_DIR}/mambapose-optimization-observer.service" \
    "${RENDER_DIR}/mambapose-optimization-observer.timer" <<'PY'
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
