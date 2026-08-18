# RTX 5090 reproduction environment

The repository-local `.venv` is a Micromamba prefix with Python 3.11,
PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, a CUDA 12.8 compiler, and
native kernels containing only `sm_120` SASS. Every command is run with
`PYTHONNOUSERSITE=1`; the host Python packages are outside the experiment.

The environment is created with:

```bash
bash tools/reproduction/bootstrap_env.sh
PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/native_build.py --all
PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/verify_native.py
```

The top-level packages are in `requirements/reproduction.in`, the exact
transitive resolution is in `requirements/reproduction-constraints.txt`, and
the source-built wheels plus SHA-256 manifest are retained below
`work_dirs/reproduction/wheelhouse` and `work_dirs/reproduction/evidence`.

## Admission gates

- **E0 — environment:** Python, Torch, torchvision, CUDA runtime/compiler,
  driver, compiler ABI, RTX 5090 capability `(12, 0)`, and a real CUDA matrix
  operation are recorded by `verify_environment.py`.
- **E1 — artifacts:** causal-conv1d, Mamba selective scan, and VMamba
  core/ndstate/oflex import successfully; `cuobjdump` finds only
  `.sm_120.cubin` artifacts.
- **E2 — causal convolution:** widths 2/3/4, both channel layouts, and
  FP32/FP16/BF16 forward, backward, and update match the upstream reference.
- **E3 — Mamba scan:** FP32/BF16 forward and backward match the PyTorch
  selective-scan reference with finite gradients.
- **E4 — VMamba scan:** core, ndstate, and oflex forward/backward agree with
  reference behavior; the paper configuration uses oflex.
- **E5 — imports/configs:** MMPose registration succeeds with mmcv-lite and
  all eleven paper configs load without optional MMCV-op models.
- **E6 — model CUDA:** VisionMamba small/base width profiles and the complete
  MambaPose S-V1 backbone/head finish FP32 and BF16 CUDA training steps.
- **E7 — real data:** each dataset/model path must finish a real train and
  evaluation batch after data preflight and before formal training.
- **E8 — clean rebuild:** `rebuild_check.sh` creates a second temporary prefix,
  installs the pinned stack and local wheels, reruns E0–E6, compares native
  hashes and every installed distribution version, and removes only that
  validated temporary prefix.

The native sources still emit upstream PyTorch deprecation warnings for the
legacy `torch.cuda.amp.custom_fwd/custom_bwd` API. They are not admission
failures because the numerical and model-level gates exercise the actual
forward/backward paths on CUDA 12.8 and Blackwell.
