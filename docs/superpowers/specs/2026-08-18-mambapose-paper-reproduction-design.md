# MambaPose Paper Reproduction Design

## Objective

Reproduce the locally evaluable results from *MambaPose: Efficient 2D Human Pose Estimation with Pose-Prior Guided State Space Model* on the workstation's single NVIDIA GeForce RTX 5090. The reproduction must remain active if the Codex session, SSH connection, or download connection ends, and every reported result must be traceable to the paper, repository revision, exact resolved config, dependency lock, dataset inventory, checkpoint, and evaluation output.

COCO test-dev is included through generation and validation of submission JSON files. The official AP values require an authenticated CodaLab upload and are therefore outside the unattended local run.

## Evidence Order

Every model or training choice follows this order:

1. `reference/pdf/icme-2025-mambapose.pdf` is authoritative.
2. `reference/fragments/icme-2025-mambapose/whole-document.md` is used for searchable paper text and is checked against the PDF tables when extraction is ambiguous.
3. Repository configs and implementation supply details omitted by the paper.
4. A new assumption is allowed only when neither source answers the question. It must be recorded in the experiment's resolved config and provenance record.

Repository behavior does not silently override a conflicting paper value. Conflicts are recorded, and the paper value is used for the primary reproduction unless it cannot execute.

## Source Integration

The paper publication branch `origin/agent/publish-mambapose-paper-fragments` is a direct descendant of the original `main` commit and is fast-forwarded into local `main`. The merged commit adds only the source PDF, its Markdown fragment, extraction inventory, manifest, and image assets.

Environment, reproduction tooling, config corrections, and model changes are committed separately from this imported paper commit so that provenance remains clear. Runtime artifacts under `.venv`, `data`, `work_dirs`, and download caches remain untracked.

## Paper Experiment Matrix

The primary matrix contains five training runs and two test-dev exports:

| Run | Dataset | Paper architecture | Paper target | Local evaluation |
| --- | --- | --- | --- | --- |
| `coco-s-v1` | COCO 2017 | SS2D `[1,1,2,1]`, 768 output channels | AP 72.8, AP50 89.7, AP75 80.5, APM 69.4, APL 79.2, AR 78.2, 2.8 GFLOPs | COCO val |
| `coco-s-v2` | COCO 2017 | SS2D `[1,2,3,2]`, 768 output channels | AP 74.2, AP50 90.5, AP75 82.0, APM 70.9, APL 80.6, AR 79.6, 4.0 GFLOPs | COCO val |
| `coco-b` | COCO 2017 | SS2D `[2,2,5,2]`, 768 output channels | AP 75.0, AP50 90.5, AP75 82.7, APM 71.3, APL 81.5, AR 80.1, 5.2 GFLOPs | COCO val |
| `crowdpose-s-v1` | CrowdPose | SS2D `[1,1,2,1]`, 768 output channels | AP 65.6, AR 75.2, 2.8 GFLOPs | CrowdPose test |
| `crowdpose-s-v2` | CrowdPose | SS2D `[1,2,3,2]`, 768 output channels | AP 67.0, AR 77.0, 4.0 GFLOPs | CrowdPose test |
| `coco-testdev-s-v1` | COCO 2017 test-dev | checkpoint from `coco-s-v1` | AP 72.4, APM 69.5, APL 77.7 | validated submission JSON only |
| `coco-testdev-s-v2` | COCO 2017 test-dev | checkpoint from `coco-s-v2` | AP 73.5, APM 70.5, APL 78.8 | validated submission JSON only |

All training runs use the paper's 256x192 input, six Transformer layers, ImageNet-pretrained VMamba-T, 300 epochs, Adam with learning rate `1e-3`, and step reductions to `1e-4` and `1e-5` at epochs 200 and 260. Validation uses flip testing and the paper-comparable top-down detection boxes.

The ablation matrix adds four training runs. The corresponding full-PIF S-V1 runs above are reused as baselines:

| Run | Dataset | Variant | Paper AP target |
| --- | --- | --- | --- |
| `coco-s-v1-no-pif` | COCO 2017 | Transformer keypoint tokens map directly to heatmaps | 72.6 |
| `crowdpose-s-v1-no-pif` | CrowdPose | Transformer keypoint tokens map directly to heatmaps | 65.3 |
| `crowdpose-s-v1-no-prior` | CrowdPose | PIF retained; local Mamba scans unrearranged keypoint order | 65.35 |
| `crowdpose-s-v1-no-cycling` | CrowdPose | Pose-prior subsequences retained without cyclic returns | 65.49 |

The PDF reports CrowdPose full-PIF S-V1 as 2.8 GFLOPs in Table IV but 5.2 GFLOPs in Table V. This is treated as a paper inconsistency. FLOPs are measured from the resolved implementation; neither number is rewritten to force agreement.

## Repository Conflicts and Required Model Interface

The existing COCO configs map to the three paper architectures. The CrowdPose S-V2 config currently uses `[1,2,3,1]`, conflicting with paper Table II `[1,2,3,2]`; the primary reproduction uses `[1,2,3,2]` and preserves the original config value in the audit record.

`TokenPose_TB_base.forward` hard-codes full PIF, while alternative no-prior and no-cycling bodies exist only as commented code. This is converted into one tested implementation with an explicit `pif_mode` selected from:

- `full`: semantic top-k fusion followed by pose-prior bidirectional cyclic scanning.
- `disabled`: Transformer keypoint tokens go directly to the heatmap projection.
- `no_prior`: semantic fusion is retained; local bidirectional Mamba consumes the unrearranged keypoint order.
- `no_cycling`: semantic fusion and pose-prior ordering are retained; repeated return-to-center/limb tokens are removed.

The default is `full`, so existing configs retain their behavior. Scan index construction and reconstruction are isolated into small functions for COCO's 17 and CrowdPose's 14 keypoints. Unit tests assert index bounds, output ordering, mode behavior, and output shapes before training.

## Isolated RTX 5090 Environment

The repository contains a real `.venv` prefix created with a repo-local Micromamba bootstrap and Python 3.11. The target stack is PyTorch 2.7.1 and torchvision 0.22.1 from the official CUDA 12.8 wheel index, plus a CUDA 12.8 compiler toolchain in the same environment. NumPy stays on the 1.x ABI for the older OpenMMLab and COCO packages.

MMPose 1.3.1 requires the MMCV 2.x family. The reproduction uses `mmcv-lite==2.1.0` because this model path does not require MMCV CUDA ops; the required custom GPU ops come from the repository's bundled Mamba sources. `mmengine`, the runtime requirements, `timm`, `fvcore`, `easydict`, and the bundled compatible `causal_conv1d` and `mamba_ssm` packages are pinned in a generated lock/inventory.

Three bundled extension builds are required:

1. `mmpose/models/backbones/Vim/causal-conv1d`
2. `mmpose/models/backbones/Vim/mamba-1p1p1`
3. `mmpose/models/backbones/Vmamba/kernels/selective_scan`

Their setup scripts currently hard-code `sm_70`, `sm_80`, and `sm_90`. A shared build policy targets only `compute_120/sm_120` on this machine, checks that `nvcc` is at least 12.8, and records compiler and ABI details. CPU reference comparisons validate forward and backward numerical behavior for each selective-scan extension before any dataset run.

The repository itself is made importable without depending on an absent root `README.md`; setup and verification must not accidentally import the user-global Python packages.

## Data and Weight Preparation

Downloads are resumable, staged to a partial name, verified, and atomically renamed. The inventory records source URL, retrieval time, byte count, and SHA-256. A preflight refuses to start training if any required file, image count, JSON schema, image reference, or checkpoint key inventory is invalid.

COCO layout:

```text
data/coco/
├── train2017/
├── val2017/
├── test2017/
├── annotations/
│   ├── person_keypoints_train2017.json
│   ├── person_keypoints_val2017.json
│   └── image_info_test-dev2017.json
└── person_detection_results/
    ├── COCO_val2017_detections_AP_H_56_person.json
    └── COCO_test-dev2017_detections_AP_H_609_person.json
```

CrowdPose layout:

```text
data/crowdpose/
├── images/
└── annotations/
    ├── mmpose_crowdpose_trainval.json
    ├── mmpose_crowdpose_test.json
    └── det_for_crowd_test_0.1_0.5.json
```

The VMamba-T checkpoint is stored under `pretrained/` with an inventory. A load audit records exact matched, missing, unexpected, and shape-mismatched keys. Training is blocked if the expected VMamba backbone is not substantially loaded; a filename alone is not accepted as evidence.

## Single-GPU Batch Fidelity

The paper does not state GPU count or per-GPU batch size, so repo configs supply the intended effective batch size. A calibration job probes the largest stable FP32 micro-batch on the RTX 5090 with the real model, resolution, optimizer, and one forward/backward step.

If the config batch does not fit, gradient accumulation preserves its effective batch size and the paper learning rate. The resolved micro-batch, accumulation count, peak allocated/reserved memory, and any deviation are recorded. Automatic mixed precision is not enabled for the primary run unless FP32 cannot execute at a practical micro-batch; if AMP becomes necessary, it is recorded as a deviation rather than silently treated as paper-equivalent.

## Durable Execution and Monitoring

A manifest-driven orchestrator runs one GPU job at a time in this order: environment verification, data verification, batch calibration, five primary trainings, four ablation trainings, local evaluations, and two test-dev exports.

Each run has a stable ID and work directory. The orchestrator uses an exclusive `flock`, writes state atomically, and considers a stage complete only when its declared artifacts validate. Re-entry skips validated stages and resumes an interrupted training run with MMPose `--resume auto` from `last_checkpoint`.

A checked-in user-systemd service links to the absolute repository path and runs the orchestrator. It uses `Restart=on-failure`, a delay between retries, and systemd start-rate limiting. Download and transient network failures receive bounded exponential retries; deterministic config, schema, CUDA, OOM-after-calibration, and data-integrity failures stop for diagnosis instead of looping forever.

A systemd timer invokes a read-only monitor periodically. The monitor writes `work_dirs/reproduction/status.json` and appends health history containing:

- active stage and experiment ID;
- PID, start time, last progress time, restart count, and last exit code;
- current epoch/iteration and latest validation metrics parsed from structured logs;
- checkpoint path and age;
- GPU utilization, memory use, temperature, and free disk space;
- a health state of `starting`, `running`, `stalled`, `failed`, or `complete` with a reason.

The service survives Codex, terminal, SSH, and network disconnection because it is owned by the user systemd manager. This host currently has `Linger=no`; a complete user logout or reboot requires a later login to start the enabled user unit unless the user separately runs `sudo loginctl enable-linger vicchen`.

## Testing and Admission Gates

No full training is admitted until all earlier gates pass:

1. Dependency resolver check and isolated-import check.
2. PyTorch CUDA tensor operation on device capability `(12, 0)`.
3. Native `sm_120` extension import and forward/backward comparison.
4. MambaPose config load and registry construction for every resolved config.
5. VMamba-T checkpoint load audit.
6. Dataset and detection-box preflight.
7. Synthetic end-to-end model loss and optimizer step.
8. Real-data short smoke run, checkpoint creation, interruption, and automatic resume.
9. Background service disconnect test and monitor freshness check.

Tests are written before the corresponding model or orchestration change. All gate outputs are retained beneath `work_dirs/reproduction/evidence/`.

## Result Acceptance and Reporting

For primary local metrics, an AP within 0.5 of the paper target is considered a direct reproduction. Ablations must also preserve the paper's direction of effect. A miss triggers evidence-based diagnosis of data identity, detection boxes, checkpoint loading, resolved config, numerical kernels, effective batch, and seed. At most one controlled rerun is automatically admitted after a concrete correction; repeated blind seed searches are not used to manufacture agreement.

The final report presents every paper target beside the measured value and delta, including AP submetrics, AR, measured GFLOPs, checkpoint hash, and resolved config hash. It separates direct reproductions, deviations, unavailable external CodaLab scores, and paper/repo inconsistencies. No paper value is presented as locally reproduced without a corresponding validated artifact.
