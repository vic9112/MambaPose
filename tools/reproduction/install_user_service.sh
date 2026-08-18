#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SYSTEMD_SOURCE="${REPO_ROOT}/systemd"
ACCOUNT_NAME="$(id -un)"
ACCOUNT_HOME="$(getent passwd "${ACCOUNT_NAME}" | cut -d: -f6)"
USER_UNIT_DIR="${ACCOUNT_HOME}/.config/systemd/user"
EVIDENCE_DIR="${REPO_ROOT}/work_dirs/reproduction/evidence"

systemd-analyze --user verify \
    "${SYSTEMD_SOURCE}/mambapose-reproduction.service" \
    "${SYSTEMD_SOURCE}/mambapose-observer.service" \
    "${SYSTEMD_SOURCE}/mambapose-observer.timer"

if [[ "${1:-}" == "--fixture-smoke" ]]; then
    echo "fixture systemd verification passed"
    exit 0
fi

mkdir -p "${USER_UNIT_DIR}" "${EVIDENCE_DIR}"
install -m 0644 "${SYSTEMD_SOURCE}/mambapose-reproduction.service" \
    "${USER_UNIT_DIR}/mambapose-reproduction.service"
install -m 0644 "${SYSTEMD_SOURCE}/mambapose-observer.service" \
    "${USER_UNIT_DIR}/mambapose-observer.service"
install -m 0644 "${SYSTEMD_SOURCE}/mambapose-observer.timer" \
    "${USER_UNIT_DIR}/mambapose-observer.timer"

systemctl --user daemon-reload
systemctl --user enable mambapose-reproduction.service
systemctl --user enable mambapose-observer.timer
SERVICE_ENABLED="$(systemctl --user is-enabled mambapose-reproduction.service)"
TIMER_ENABLED="$(systemctl --user is-enabled mambapose-observer.timer)"

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
    "${SYSTEMD_SOURCE}/mambapose-reproduction.service" \
    "${SYSTEMD_SOURCE}/mambapose-observer.service" \
    "${SYSTEMD_SOURCE}/mambapose-observer.timer" <<'PY'
import hashlib
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

path, account, linger, durability_scope, service_enabled, timer_enabled, *unit_paths = sys.argv[1:]
unit_sha256 = {
    Path(unit).name: hashlib.sha256(Path(unit).read_bytes()).hexdigest()
    for unit in unit_paths
}
destination = Path(path)
temporary = destination.with_suffix('.json.tmp')
temporary.write_text(json.dumps({
    'installed_at': datetime.now(timezone.utc).isoformat(),
    'account': account,
    'linger': linger,
    'durability_scope': durability_scope,
    'service_enabled': service_enabled,
    'timer_enabled': timer_enabled,
    'unit_sha256': unit_sha256,
}, indent=2, sort_keys=True) + '\n')
os.replace(temporary, destination)
PY

echo "linger=${LINGER_VALUE} durability_scope=${DURABILITY_SCOPE}"
if [[ "${LINGER_VALUE}" != "yes" ]]; then
    echo "linger must be enabled before formal campaign launch" >&2
    exit 78
fi
