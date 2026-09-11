# MambaPose reproduction and FPGA-oriented optimization

This repository reproduces the ICME 2025 MambaPose results and contains three
algorithm-level optimization experiments for a future YOLO → MambaPose → motion
trajectory pipeline. The intended pose output is the COCO 17-keypoint skeleton;
temporal association and trajectory history are downstream tasks, not outputs of
the pose network itself.

As of 2026-09-01, all optimization GPU workloads are paused. The code,
configuration, validation contracts, and current results are preserved. No
MambaPose FPGA synthesis, routed implementation, or U50 board measurement has
been performed yet, so this repository does not claim FPGA resource or latency
improvements.

## Reproduction status

The main paper results were independently reproduced from VMamba pretrained
backbones because the authors did not publish complete MambaPose pose
checkpoints. Nine best inference checkpoints and their nine full-state
`epoch_300` resume checkpoints are available from the
[`mambapose-icme2025-reproduction-v1`](https://github.com/vic9112/MambaPose/releases/tag/mambapose-icme2025-reproduction-v1)
release; exact URLs and SHA-256 values are recorded in
[`reproduction/checkpoints.json`](reproduction/checkpoints.json).
The three available optimization training states are published separately in
[`mambapose-optimization-resume-v1`](https://github.com/vic9112/MambaPose/releases/tag/mambapose-optimization-resume-v1)
and catalogued in
[`optimization/resume_checkpoints.json`](optimization/resume_checkpoints.json).

| Dataset | Variant | Paper AP | Reproduced AP | Delta AP |
| --- | --- | ---: | ---: | ---: |
| COCO val2017 | S-V1 | 72.800 | 72.832 | +0.032 |
| COCO val2017 | S-V2 | 74.200 | 74.212 | +0.012 |
| COCO val2017 | B | 75.000 | 74.895 | -0.105 |
| CrowdPose test | S-V1 | 65.600 | 65.422 | -0.178 |
| CrowdPose test | S-V2 | 67.000 | 67.060 | +0.060 |

The full scientific report, including limitations and ablations, is in
[`docs/reproduction/results.md`](docs/reproduction/results.md). In particular,
the paper's claimed PIF benefit was not reproduced: the local no-PIF model was
0.244 AP higher than the matched full S-V1 model. This makes no-PIF promising,
but it is not evidence that every paper ablation claim is reliable.

## Intended application pipeline

1. YOLO detects people and supplies image-space person bounding boxes.
2. The top-down preprocessing used by MMPose crops and resizes each person to
   the MambaPose input resolution. YOLO output does not need to be a COCO JSON
   file at runtime, but its box coordinates and image metadata must be converted
   to the same convention used by the COCO-trained top-down pipeline.
3. MambaPose predicts 17 COCO keypoints and confidence values per person.
4. A tracker preserves person identity and stores observed keypoint history to
   produce skeleton trajectories. This is observed-history tracking, not future
   motion prediction.

## Three optimization directions

The acceptance target requested for algorithm optimization is less than
0.1 COCO AP loss before any hardware-specific approximation.

| Direction | What is changed | Current accuracy evidence | Compute/resource implication | Decision |
| --- | --- | --- | --- | --- |
| 1. Structural pruning / lower compute: no-PIF | Remove the PIF branch, including 38 tensors, 3 head Mamba blocks, 2 LayerNorms, the scalar gate, and dynamic top-k/gather/scan work | Frozen matched evaluation: 73.076 AP vs 72.832 baseline, or +0.244 AP. A separate config-matched local seed-0 run reached 72.968 AP with flip test; it is not an admitted arm of the merged formal contract | Parameters 22.029M → 20.713M (-5.97%); local RTX 5090 batch-1 median latency improved 29.361 → 26.367 ms with flip (-10.20%) and 24.989 → 23.323 ms without flip (-6.67%) | **Primary recommendation** |
| 2. Hardware-friendly nonlinear approximation: PWL Softplus | Replace the five SS2D transition Softplus roles with a 16-segment PWL function on [-8, 8] and continuous asymptotic tails | Flip AP 72.848 vs 72.832 (+0.016); no-flip AP 72.288 vs 72.313 (-0.025). Both satisfy the 0.1 AP target on the frozen checkpoint | Parameters and checkpoint bytes are unchanged. It removes these five exact Softplus evaluations, but current PyTorch median latency does not prove a speedup | **Complementary candidate** after direct no-PIF+PWL validation |
| 3. Low-bit attention plus QAT/self-distillation: Binary Q/K | Deterministic scaled sign Q/K with STE, 60-epoch QAT, frozen float teacher, and final-heatmap distillation | Local post-QAT evaluation: flip AP 71.078, a 1.754 AP drop; no-flip AP 69.487, a 2.826 AP drop | Only 6,489,600 of 115,418,160 attention multiplications (5.62%) are replaced in the modeled operation; softmax, V path, projections, and the Mamba backbone remain floating point. The current implementation is a float proxy, not a bit-packed FPGA kernel | **Reject for the current accuracy target** |

### Direction 1: no-PIF

No-PIF is the best current algorithm-level choice because it removes a complete
dynamic branch and reduced both model size and measured end-to-end CUDA latency
without an observed AP penalty. The published inference checkpoint is
`mambapose-coco-s-v1-no-pif-best.pth`, SHA-256
`28cd02405e58d619896a0430f91936684084ef7e3423d507c7a10b743759a5fb`.
The parameter counts and RTX latency values above come from preserved local
profile/latency artifacts; those small artifacts have not been promoted into a
tracked public evidence bundle.

The newer formal paired campaign is defined by
[`optimization/formal_stage_c.json`](optimization/formal_stage_c.json). A
separate config-matched local no-PIF seed-0 run completed 300 epochs and selected
epoch 280 (flip AP 72.968, no-flip AP 72.326; checkpoint SHA-256
`0c71efb40a0daeabc215ad3b772b4de440311b95310f4c5739986cc539824eb4`). That run
predates the merged formal artifact contract and does not contain the required
`run-init.json` and `train-result.json`, so the current formal loader does not
admit it as a completed campaign arm. The matched full seed-0 run was paused
during epoch 66, after an epoch-65 health evaluation. Consequently, no formal
paired delta or multi-seed confidence interval exists. The large run trees stay
outside Git, but the last full-state checkpoints for the paused full run and the
completed no-PIF run are downloadable from the optimization resume release.
Publishing them does not retroactively satisfy the formal campaign's missing
`run-init.json` / `train-result.json` admission contract.

### Direction 2: PWL Softplus

Calibration selected Softplus rather than SiLU, GELU, or exp. For the selected
candidate, 25.85% of observed inputs were outside [-8, 8], so clamping would be
incorrect; continuous tails are part of the candidate. The observed-range max
and mean approximation errors were 0.02887 and 0.000428. The tracked,
hash-bound evidence is under
[`optimization/evidence_snapshots/no_pif_softplus/`](optimization/evidence_snapshots/no_pif_softplus/).

PWL should be considered an FPGA mapping aid, not a demonstrated model-level
speedup. The combined no-PIF + PWL implementation and evidence validator are
present, but direct combined COCO evaluation was not run before the campaign was
paused. It must therefore remain a candidate rather than the default model.

### Direction 3: Binary Q/K

The recovery design and exact operation declaration are recorded in
[`optimization/binary_qk_recovery.json`](optimization/binary_qk_recovery.json).
QAT/self-distillation recovered part of the direct binarization loss but remained
far outside the 0.1 AP budget. It also attacks a small fraction of this model's
workload, so it is a poor first FPGA optimization for MambaPose. The post-QAT
metric artifacts are local/non-public; the tracked repository publishes the
design, admission contracts, tests, and this explicitly qualified result summary.
Its epoch-60 full-state checkpoint is available in the optimization resume
release for audit or new research, but its publication does not change the
rejected deployment decision.

## Continue training from a published checkpoint

Release assets whose names contain `resume` were restricted-loaded and verified
to contain model, optimizer, scheduler, message-hub, epoch, and iteration state.
For example, download the COCO S-V1 state and start a new 30-epoch continuation:

```bash
mkdir -p work_dirs/continuations/coco-s-v1
gh release download mambapose-icme2025-reproduction-v1 \
  --pattern mambapose-coco-s-v1-resume-epoch300.pth \
  --dir work_dirs/continuations/coco-s-v1
echo '61d543df733ec08c74bd295b6f5d0b8b6da88db9587411cdc2ab31f41de472e1  work_dirs/continuations/coco-s-v1/mambapose-coco-s-v1-resume-epoch300.pth' \
  | sha256sum --check
PYTHONNOUSERSITE=1 .venv/bin/python tools/train.py \
  configs/reproduction/coco_s_v1.py \
  --work-dir work_dirs/continuations/coco-s-v1 \
  --resume work_dirs/continuations/coco-s-v1/mambapose-coco-s-v1-resume-epoch300.pth \
  --cfg-options train_cfg.max_epochs=330
```

The original paper schedule ends at epoch 300, so this example is a new
fine-tuning experiment and must not be reported as the original reproduction.
Select the matching config, asset, and SHA-256 for other variants from
`reproduction/checkpoints.json`. For the unfinished matched full seed-0
optimization run, use the epoch-65 asset and its config listed in
`optimization/resume_checkpoints.json`; this can continue to the existing
300-epoch limit. The completed no-PIF and Binary Q/K assets require a deliberately
revised schedule before they can take additional optimizer steps. PWL Softplus
has no separate training state because it is a post-training operator
replacement.

## Recommended algorithm before FPGA work

Use the no-PIF S-V1 model as the primary deployment baseline. Then evaluate one
combined no-PIF + tail-aware 16-segment Softplus-PWL candidate against the same
checkpoint, COCO detector input, flip policy, and batch-1 protocol. Keep PWL only
if the direct combined result remains within 0.1 AP. Do not carry Binary Q/K into
the first hardware implementation.

This ordering gives the FPGA design a smaller graph before hardware-specific
work begins. The remaining dominant targets still include five SS2D/selective
scan blocks, six floating-point attention blocks, convolution/linear layers,
normalization, activation, preprocessing, and memory movement. Their trade-offs
must be measured with HLS synthesis, routed timing, and U50 board execution;
PyTorch parameter counts or RTX latency cannot substitute for those results.

## Environment and verification

The repository-local environment is `.venv`. Dataset preparation and the exact
COCO/CrowdPose directory contracts are documented in
[`docs/reproduction/results.md`](docs/reproduction/results.md) and the
reproduction tools under [`tools/reproduction/`](tools/reproduction/).

CPU-only publication verification can be run without starting a training job:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest \
  tests/test_optimization -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest \
  tests/test_reproduction -q
```

Training and evaluation commands are deliberately not started by importing the
package or running the tests. All long-running GPU services and timers were
stopped before this publication snapshot.
