#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REBUILD_ROOT="$(mktemp -d "${REPO_ROOT}/.tools/rebuild.XXXXXX")"
REBUILD_PREFIX="${REBUILD_ROOT}/venv"
REBUILD_EVIDENCE="${REBUILD_ROOT}/evidence"
PERSISTENT_EVIDENCE="${REPO_ROOT}/work_dirs/reproduction/evidence/rebuild.json"

cleanup() {
    case "${REBUILD_ROOT}" in
        "${REPO_ROOT}"/.tools/rebuild.*)
            rm -rf -- "${REBUILD_ROOT}"
            ;;
        *)
            echo "refusing to remove unexpected rebuild path: ${REBUILD_ROOT}" >&2
            exit 78
            ;;
    esac
}
trap cleanup EXIT

mkdir -p "${REBUILD_EVIDENCE}"
export PYTHONNOUSERSITE=1
export MAMBAPOSE_ENV_PREFIX="${REBUILD_PREFIX}"
export MAMBAPOSE_EVIDENCE_OUTPUT="${REBUILD_EVIDENCE}/environment.json"
export MAMBAPOSE_NATIVE_EVIDENCE_OUTPUT="${REBUILD_EVIDENCE}/native.json"

bash "${SCRIPT_DIR}/bootstrap_env.sh"

"${REBUILD_PREFIX}/bin/python" -m pip install \
    --disable-pip-version-check --force-reinstall --no-deps \
    "${REPO_ROOT}"/work_dirs/reproduction/wheelhouse/*.whl
"${REBUILD_PREFIX}/bin/python" -m pip install \
    --disable-pip-version-check --no-deps --editable "${REPO_ROOT}"

export CUDA_HOME="${REBUILD_PREFIX}"
export PATH="${REBUILD_PREFIX}/bin:${PATH}"
"${REBUILD_PREFIX}/bin/python" "${SCRIPT_DIR}/verify_environment.py"
"${REBUILD_PREFIX}/bin/python" "${SCRIPT_DIR}/verify_native.py"
"${REBUILD_PREFIX}/bin/python" -m pip check

"${REBUILD_PREFIX}/bin/python" - \
    "${REPO_ROOT}/.venv/bin/python" \
    "${REPO_ROOT}/work_dirs/reproduction/evidence/environment.json" \
    "${REBUILD_EVIDENCE}/environment.json" \
    "${REPO_ROOT}/work_dirs/reproduction/evidence/native.json" \
    "${REBUILD_EVIDENCE}/native.json" \
    "${PERSISTENT_EVIDENCE}" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

primary_python, primary_env_path, rebuilt_env_path, primary_native_path, rebuilt_native_path, output_path = map(Path, sys.argv[1:])
primary_env = json.loads(primary_env_path.read_text())
rebuilt_env = json.loads(rebuilt_env_path.read_text())
environment_keys = [
    'python_version', 'torch_version', 'torchvision_version', 'cuda_runtime',
    'nvcc_version', 'device_capability', 'cxx11_abi'
]
environment_match = all(
    primary_env[key] == rebuilt_env[key] for key in environment_keys)

primary_native = json.loads(primary_native_path.read_text())
rebuilt_native = json.loads(rebuilt_native_path.read_text())
primary_hashes = {
    item['module']: item['sha256'] for item in primary_native['artifacts']
}
rebuilt_hashes = {
    item['module']: item['sha256'] for item in rebuilt_native['artifacts']
}
native_match = primary_hashes == rebuilt_hashes

def package_versions(python):
    output = subprocess.run(
        [str(python), '-m', 'pip', 'list', '--format=json'],
        check=True, capture_output=True, text=True).stdout
    return {
        package['name'].lower(): package['version']
        for package in json.loads(output)
    }

primary_packages = package_versions(primary_python)
rebuilt_packages = package_versions(sys.executable)
package_versions_match = primary_packages == rebuilt_packages
evidence = {
    'verified_at': datetime.now(timezone.utc).isoformat(),
    'environment_match': environment_match,
    'native_artifact_match': native_match,
    'package_versions_match': package_versions_match,
    'environment_keys': environment_keys,
    'native_sha256': rebuilt_hashes,
    'package_versions': rebuilt_packages,
}
if not environment_match or not native_match or not package_versions_match:
    raise SystemExit(f'rebuild mismatch: {evidence}')
output_path.parent.mkdir(parents=True, exist_ok=True)
temporary = output_path.with_suffix('.json.tmp')
temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
temporary.replace(output_path)
print(json.dumps(evidence, indent=2, sort_keys=True))
PY
