#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TOOLS_DIR="${REPO_ROOT}/.tools"
ENV_PREFIX="${REPO_ROOT}/.venv"
MAMBA_ROOT_PREFIX="${TOOLS_DIR}/micromamba-root"
MICROMAMBA_VERSION="2.3.2"
MICROMAMBA_URL="https://micro.mamba.pm/api/micromamba/linux-64/${MICROMAMBA_VERSION}"
MICROMAMBA_ARCHIVE="${TOOLS_DIR}/micromamba-${MICROMAMBA_VERSION}.tar.bz2"
MICROMAMBA_BIN="${TOOLS_DIR}/micromamba"
MICROMAMBA_SHA256="5512233cdd8564a671626081026dc861537a963baa06706baab08fac6f3bb9d2"

export PYTHONNOUSERSITE=1
export MAMBA_ROOT_PREFIX

mkdir -p "${TOOLS_DIR}"

if [[ ! -f "${MICROMAMBA_ARCHIVE}" ]]; then
    curl --fail --location --retry 8 --retry-all-errors \
        --continue-at - --output "${MICROMAMBA_ARCHIVE}.part" \
        "${MICROMAMBA_URL}"
    mv "${MICROMAMBA_ARCHIVE}.part" "${MICROMAMBA_ARCHIVE}"
fi

actual_sha256="$(sha256sum "${MICROMAMBA_ARCHIVE}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${MICROMAMBA_SHA256}" ]]; then
    quarantine="${MICROMAMBA_ARCHIVE}.bad.$(date -u +%Y%m%dT%H%M%SZ)"
    mv "${MICROMAMBA_ARCHIVE}" "${quarantine}"
    echo "micromamba SHA-256 mismatch; quarantined at ${quarantine}" >&2
    exit 78
fi

if [[ ! -x "${MICROMAMBA_BIN}" ]]; then
    tar -xOf "${MICROMAMBA_ARCHIVE}" bin/micromamba > "${MICROMAMBA_BIN}.tmp"
    chmod 0755 "${MICROMAMBA_BIN}.tmp"
    mv "${MICROMAMBA_BIN}.tmp" "${MICROMAMBA_BIN}"
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    "${MICROMAMBA_BIN}" create --yes --prefix "${ENV_PREFIX}" \
        --channel nvidia --channel conda-forge --strict-channel-priority \
        python=3.11 cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8 pip
fi

export CUDA_HOME="${ENV_PREFIX}"
export PATH="${ENV_PREFIX}/bin:${PATH}"

"${ENV_PREFIX}/bin/python" -m pip install --disable-pip-version-check \
    --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.7.1' 'torchvision==0.22.1'
"${ENV_PREFIX}/bin/python" -m pip install --disable-pip-version-check \
    --requirement "${REPO_ROOT}/requirements/reproduction.in"

"${ENV_PREFIX}/bin/python" "${SCRIPT_DIR}/verify_environment.py"
