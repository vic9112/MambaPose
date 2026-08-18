# MambaPose Blackwell Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible repo-local `.venv` whose Python, PyTorch, CUDA compiler, native Mamba kernels, and MambaPose imports are validated on RTX 5090 `sm_120`.

**Architecture:** A repo-local Micromamba binary creates `.venv` with Python 3.11 and CUDA 12.8 build tools; pip installs a pinned PyTorch cu128/runtime stack. Native extensions are source-built into a hashed local wheelhouse and admitted only after architecture and numerical gates pass. Optional imports prevent unused MMCV-op models from blocking the selected MambaPose path.

**Tech Stack:** Python 3.11, Micromamba, PyTorch 2.7.1+cu128, CUDA 12.8, GCC 13, mmengine 0.10.7, mmcv-lite 2.1.0, pytest

**Spec:** `docs/superpowers/specs/2026-08-18-mambapose-paper-reproduction-design.md`

## Global Constraints

- The environment prefix is exactly `.venv`, and `PYTHONNOUSERSITE=1` is mandatory for every build and runtime entrypoint.
- Python is 3.11; PyTorch is `2.7.1+cu128`; torchvision is `0.22.1+cu128`; NumPy is `1.26.4`.
- CUDA compiler is 12.8 or newer and native artifacts contain `sm_120` SASS.
- Do not install `mmpose/models/backbones/Vim/vim/vim_requirements.txt` wholesale.
- Do not start dataset training until E0-E7 gates pass.

---

### Task 1: Deterministic Environment Bootstrap

**Files:**
- Create: `requirements/reproduction.in`
- Create: `tools/reproduction/bootstrap_env.sh`
- Create: `tools/reproduction/verify_environment.py`
- Test: `tests/test_reproduction/test_environment_contract.py`

**Interfaces:**
- Consumes: host driver and RTX 5090 exposed by `nvidia-smi`.
- Produces: `.venv/bin/python`, `.venv/bin/nvcc`, `work_dirs/reproduction/evidence/environment.json`, and `verify_environment.main() -> int`.

- [ ] **Step 1: Write the failing environment-contract tests**

```python
def test_requirement_pins():
    text = Path('requirements/reproduction.in').read_text()
    for pin in ('torch==2.7.1', 'torchvision==0.22.1',
                'numpy==1.26.4', 'mmcv-lite==2.1.0',
                'mmengine==0.10.7', 'timm==0.9.16'):
        assert pin in text

def test_bootstrap_disables_user_site():
    text = Path('tools/reproduction/bootstrap_env.sh').read_text()
    assert 'PYTHONNOUSERSITE=1' in text
    assert 'python=3.11' in text
    assert 'cuda-nvcc=12.8' in text
```

- [ ] **Step 2: Run tests and verify the files are absent**

Run: `python3 -m pytest tests/test_reproduction/test_environment_contract.py -q`

Expected: FAIL because the reproduction requirements/bootstrap files do not exist.

- [ ] **Step 3: Implement bootstrap and verifier**

`bootstrap_env.sh` must download a pinned Micromamba artifact to `.tools/`, verify its declared SHA-256, create `.venv` with Python 3.11 plus CUDA 12.8 compiler/development packages, install the exact cu128 Torch wheels, and install the pinned runtime list with `python -m pip`. `verify_environment.py` must emit JSON containing Python/Torch/torchvision/CUDA/driver/nvcc/GCC/ABI/device capability and fail unless the required versions and `(12, 0)` match.

The native build helper pin is `ninja==1.11.1.4`; the older 1.11.1.1 wheel advertises only legacy manylinux tags and fails `pip check` under this Python/pip stack. `chumpy==0.70` remains installed only because MMPose 1.3.1 declares it in package metadata; the MambaPose path does not import it.

- [ ] **Step 4: Run the contract and real bootstrap gates**

Run: `python3 -m pytest tests/test_reproduction/test_environment_contract.py -q`

Run: `bash tools/reproduction/bootstrap_env.sh`

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/verify_environment.py`

Expected: tests PASS; verifier exits 0 and writes an environment record with Python 3.11, Torch 2.7.1+cu128, nvcc 12.8, and capability `[12, 0]`.

- [ ] **Step 5: Freeze and commit**

Run: `.venv/bin/python -m pip freeze --all > work_dirs/reproduction/evidence/pip-freeze.txt`

Run: `git add requirements/reproduction.in tools/reproduction/bootstrap_env.sh tools/reproduction/verify_environment.py tests/test_reproduction/test_environment_contract.py && git commit -m "build: bootstrap RTX 5090 reproduction environment"`

### Task 2: Optional Import Boundary for MMCV Lite

**Files:**
- Modify: `mmpose/models/heads/__init__.py`
- Modify: `mmpose/models/backbones/__init__.py`
- Modify: `mmpose/models/backbones/Vim/vim/models_mamba.py`
- Modify: `setup.py`
- Test: `tests/test_reproduction/test_mambapose_imports.py`

**Interfaces:**
- Consumes: mmcv-lite without `mmcv.ops` and the target MambaPose config paths.
- Produces: `register_all_modules()` and target config construction without loading unused EDPose/MMCV native ops.

- [ ] **Step 1: Write failing subprocess import tests**

```python
def test_target_registration_does_not_require_mmcv_ops():
    code = "from mmpose.utils import register_all_modules; register_all_modules()"
    result = subprocess.run([sys.executable, '-c', code], text=True,
                            capture_output=True, env={**os.environ,
                            'PYTHONNOUSERSITE': '1'})
    assert result.returncode == 0, result.stderr

def test_vim_uses_package_relative_rope():
    text = Path('mmpose/models/backbones/Vim/vim/models_mamba.py').read_text()
    assert 'from rope import *' not in text

def test_setup_metadata_does_not_require_missing_readme():
    subprocess.run([sys.executable, 'setup.py', '--name'], check=True)
```

- [ ] **Step 2: Confirm the eager import failures**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_mambapose_imports.py -q`

Expected: FAIL at the unconditional `mmcv.ops` or native-backbone import and at the absolute `rope` import.

- [ ] **Step 3: Add explicit optional registration boundaries**

Guard only optional EDPose and unused native-backbone exports with `try/except ImportError`; re-raise when the missing module belongs to the selected MambaPose dependency path, and expose a diagnostic map of unavailable optional components. Remove the duplicate absolute `from rope import *`. Make setup metadata use a defined fallback when root `README.md` is absent so editable installation cannot fail at metadata generation.

- [ ] **Step 4: Verify target and failure semantics**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_mambapose_imports.py -q`

Expected: target registration PASS; explicitly requesting an unavailable optional model still raises a named diagnostic instead of disappearing silently.

- [ ] **Step 5: Commit**

Run: `git add mmpose/models tests/test_reproduction/test_mambapose_imports.py && git commit -m "fix: isolate optional native model imports"`

### Task 3: Blackwell Native Build Policy

**Files:**
- Create: `tools/reproduction/native_build.py`
- Modify: `mmpose/models/backbones/Vim/mamba-1p1p1/setup.py`
- Modify: `mmpose/models/backbones/Vmamba/kernels/selective_scan/setup.py`
- Test: `tests/test_reproduction/test_native_build_policy.py`

**Interfaces:**
- Consumes: `CUDA_HOME`, nvcc version, PyTorch ABI, and extension source directories.
- Produces: `blackwell_gencode(cuda_version: tuple[int, int]) -> list[str]`, force-build environment, wheels under `work_dirs/reproduction/wheelhouse/`, and a build manifest.

- [ ] **Step 1: Write policy tests**

```python
def test_blackwell_gencode_is_sm120_only():
    assert blackwell_gencode((12, 8)) == [
        '-gencode', 'arch=compute_120,code=sm_120']

def test_cuda_older_than_128_is_rejected():
    with pytest.raises(RuntimeError, match='CUDA 12.8'):
        blackwell_gencode((12, 7))
```

- [ ] **Step 2: Confirm current setup scripts fail the policy**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_native_build_policy.py -q`

Expected: FAIL because the helper is absent and hard-coded sm70/80/90 flags remain.

- [ ] **Step 3: Implement a shared strict build policy**

Make both setup scripts import the helper by absolute repository path, use only `compute_120/sm_120`, and reject an older compiler. `native_build.py` must set `MAMBA_FORCE_BUILD=TRUE`, avoid guessed wheel downloads, run builds with a fixed `MAX_JOBS`, save stdout/stderr and SHA-256, and capture `torch._C._GLIBCXX_USE_CXX11_ABI`.

- [ ] **Step 4: Verify source policy**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_native_build_policy.py -q`

Expected: PASS with no `compute_70`, `compute_80`, or `compute_90` in the selected setup files.

- [ ] **Step 5: Commit**

Run: `git add tools/reproduction/native_build.py mmpose/models/backbones/Vim/mamba-1p1p1/setup.py mmpose/models/backbones/Vmamba/kernels/selective_scan/setup.py tests/test_reproduction/test_native_build_policy.py && git commit -m "build: target Mamba kernels at sm120"`

### Task 4: Compatible Causal Convolution and Native Numerical Gates

**Files:**
- Create: `tools/reproduction/fetch_causal_conv.py`
- Create: `tools/reproduction/verify_native.py`
- Test: `tests/test_reproduction/test_causal_source.py`
- Test: `tests/test_reproduction/test_native_cuda.py`
- Test: `tests/test_reproduction/test_model_cuda.py`

**Interfaces:**
- Consumes: pinned upstream causal-conv1d v1.1.0 source archive and `native_build.py`.
- Produces: a source-hash-verified causal-conv wheel, Mamba/VMamba wheels, `verify_native.main() -> int`, and `work_dirs/reproduction/evidence/native.json`.

- [ ] **Step 1: Write source/API and CUDA reference tests**

```python
def test_causal_source_has_mamba_compatible_bindings(source_dir):
    binding = (source_dir / 'csrc/causal_conv1d.cpp').read_text()
    assert 'seq_idx' in binding
    assert 'causal_conv1d_fwd' in binding

@pytest.mark.cuda
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_causal_forward_backward_matches_reference(dtype):
    actual, expected = run_causal_case(dtype=dtype, width=4)
    torch.testing.assert_close(actual, expected, **tolerance(dtype))
```

- [ ] **Step 2: Confirm old bundled ABI is rejected**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_causal_source.py -q`

Expected: FAIL because the bundled 1.0.0 four/six-argument binding lacks the 1.1.0 API.

- [ ] **Step 3: Fetch, verify, patch architecture, and build**

`fetch_causal_conv.py` must accept only the pinned v1.1.0 archive SHA-256, extract safely under `work_dirs/reproduction/sources/`, apply the same sm120-only policy, and record the upstream URL/commit/license. Build causal first, then mamba_ssm, then VMamba core/ndstate/oflex; install from the resulting local wheels.

- [ ] **Step 4: Run native admission matrix**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/native_build.py --all`

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_native_cuda.py -q`

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_model_cuda.py -q`

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/verify_native.py`

Expected: all native modules import, contain sm120 SASS by artifact inspection, have finite gradients, and match references within dtype-specific tolerances. VisionMamba small/base and VMamba VSSM each complete a CUDA train step in FP32 and BF16 without NaN/Inf.

- [ ] **Step 5: Commit**

Run: `git add tools/reproduction/fetch_causal_conv.py tools/reproduction/verify_native.py tests/test_reproduction/test_causal_source.py tests/test_reproduction/test_native_cuda.py tests/test_reproduction/test_model_cuda.py && git commit -m "test: gate Blackwell Mamba native kernels"`

### Task 5: Rebuild and End-to-End Environment Evidence

**Files:**
- Create: `tools/reproduction/rebuild_check.sh`
- Create: `docs/reproduction/environment.md`
- Test: `tests/test_reproduction/test_environment_docs.py`

**Interfaces:**
- Consumes: bootstrap script, wheelhouse, requirements pins, environment/native verifiers.
- Produces: documented rebuild command and complete E0-E6 evidence set.

- [ ] **Step 1: Write a docs/evidence contract test**

```python
def test_environment_docs_name_all_evidence_gates():
    text = Path('docs/reproduction/environment.md').read_text()
    for gate in ('E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6'):
        assert gate in text
```

- [ ] **Step 2: Confirm documentation is absent**

Run: `python3 -m pytest tests/test_reproduction/test_environment_docs.py -q`

Expected: FAIL because the rebuild entrypoint and environment evidence guide do not exist.

- [ ] **Step 3: Implement idempotent rebuild verification**

`rebuild_check.sh` must create a fresh temporary prefix without touching `.venv`, install only from the pinned inputs and local wheels, run environment/native/import/model-one-step gates, compare version/hash manifests, then remove only its validated temporary directory.

- [ ] **Step 4: Run the full environment gate**

Run: `bash tools/reproduction/rebuild_check.sh`

Run: `python3 -m pytest tests/test_reproduction/test_environment_docs.py -q`

Expected: PASS and evidence hashes match the primary `.venv` artifacts.

- [ ] **Step 5: Commit**

Run: `git add tools/reproduction/rebuild_check.sh docs/reproduction/environment.md tests/test_reproduction/test_environment_docs.py && git commit -m "docs: record reproducible Blackwell environment"`
