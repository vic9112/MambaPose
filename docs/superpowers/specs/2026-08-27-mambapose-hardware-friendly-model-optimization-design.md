# MambaPose Hardware-Friendly Model Optimization Design

## Objective

Optimize the reproduced COCO S-V1 MambaPose model for a later FPGA deployment
without spending the accuracy margin that will be needed during hardware
mapping. The optimized model keeps the existing top-down contract: a frame and
one or more person bounding boxes are transformed to `N x 3 x 256 x 192`, and
the model returns COCO-17 keypoint coordinates, scores, and optionally
`17 x 64 x 48` heatmaps.

The immediate deliverable is an algorithm-level Pareto comparison, not an FPGA
bitstream. A candidate is admissible only when its matched-seed mean COCO val AP
drop is strictly less than 0.1 AP point and its result is reproducible. FPGA
resource allocation, clocking, HBM placement, and fixed-point implementation
remain a later project phase with a separate accuracy budget.

The downstream product boundary is:

```text
video frame
  -> YOLO person detection and tracking
  -> tracked xyxy person boxes
  -> MambaPose top-down affine crop
  -> COCO-17 keypoints and scores
  -> per-track historical keypoint buffer and smoothing
  -> stick figure and observed joint/body trajectories
```

This project optimizes the MambaPose stage. It does not add future-motion
prediction, replace the YOLO tracker, or design the final temporal smoother.

## Source and Decision Order

Model and training decisions follow this order:

1. `reference/pdf/icme-2025-mambapose.pdf` and its checked fragment define the
   intended MambaPose architecture and paper evaluation protocol.
2. The repository implementation and resolved reproduction configs supply
   details omitted by the paper.
3. The completed local reproduction supplies baseline checkpoints and measured
   behavior.
4. External primary research motivates new optimization candidates. It cannot
   silently change the baseline or acceptance rule.

The most relevant external directions are BinaryAttention for one-bit Q/K in
vision attention, PTQ4VM and Quamba for Mamba-specific activation and state
outliers, FastMamba for power-of-two SSM quantization and nonlinear
approximations, ViM-Q for Vision Mamba FPGA-oriented quantization, and DWPose
for pose distillation. Every implemented candidate records which mechanism it
uses rather than claiming a generic "quantized Mamba" result.

## Frozen Baseline

Optimization branches start from release commit
`61b7ff8f476dc284f2684fbb2e5b968291431881`. The reproduced checkpoints were
trained at source commit `1f4364d62279bf9fe3ac55e0e2d57339036aeb81`;
the later release commit adds result documentation and the checkpoint catalog
without changing the trained model implementation. Both commits are recorded
so a branch root is never confused with checkpoint training provenance. The
seed-0 reference checkpoint is
`work_dirs/reproduction/runs/coco-s-v1/best_coco_AP_epoch_300.pth`, whose
published SHA-256 is
`a6f76dae86db4d92c445f26a428b61911e8b42c4de2119348997e9537cc7cdd2` and
whose measured COCO val AP is 72.83223444297235.

The existing COCO S-V1 no-PIF result, 73.07620686556272 AP at seed 0, is useful
preliminary evidence but is not sufficient to accept deletion of PIF. The
paper did not report multi-seed uncertainty, and the local PIF direction did
not reproduce. Formal conclusions therefore use newly paired baseline and
candidate runs rather than comparing unrelated single checkpoints.

Paper-comparable AP uses the existing person detections and `flip_test=True`.
The same candidates are also measured with `flip_test=False` to expose the
future deployment cost, but no-flip AP is a secondary metric and is never
substituted for the primary paper-comparable gate.

## Current Optimization Surface

The S-V1 model has 22.029 million parameters and a local traced lower bound of
2.738 GFLOPs. The traced value omits custom scan operators and is not treated
as an exact total. The current one-off local parameter-count estimate is:

| Component | Parameters |
| --- | ---: |
| VMamba backbone | 15,763,872 |
| Complete pose head | 6,265,345 |
| Six-layer Transformer in the head | 3,945,984 |
| Pose Information Fusion | 1,315,841 |
| Heatmap projection MLP | 790,016 |
| Patch embedding | 196,864 |

Stage A must reproduce this breakdown with a tracked script and structured
artifact before any component-level reduction is claimed. Until then, only the
tracked 22.029 million total is publication evidence and the finer rows are
planning estimates.

The backbone produces an `8 x 6 x 768` feature map. The head converts this to
48 visual tokens, prepends 17 keypoint tokens, applies six Transformer layers
to 65 tokens, then applies PIF and a heatmap projection. PIF contains a dynamic
`17 x 17` similarity matrix, per-sample top-5 selection, advanced indexing,
three Mamba blocks, and fixed pose-prior scan tables. The VMamba backbone
contains CrossScan, SelectiveScan, and CrossMerge custom operators.

Consequently, Binary Q/K applies only to the small 65-token Transformer head;
it cannot optimize the dominant VMamba backbone. It remains a measured Route 3
candidate rather than the main strategy.

## Branch and Worktree Governance

The published Git `main` remains the stable reproduced baseline. The commit
containing this design document is the branch point on top of the frozen model
commit. The three research branches start from that documentation-only commit
so every worktree contains the same contract while the model code baseline
remains unchanged. Model research uses these isolated branches:

| Branch | Ownership | Purpose |
| --- | --- | --- |
| `algo/accuracy-first` | primary agent | Shared benchmark contract, distillation, staged integration, and final Pareto candidate |
| `algo/structural-pif` | structural subagent | Remove or regularize dynamic PIF behavior |
| `algo/ssm-quant-pwl` | quantization subagent | Mamba-aware fake quantization, PWL experiments, and head-local Binary Q/K |

The primary agent owns shared schemas, evaluation semantics, integration, and
comparison reports. Each subagent commits only to its assigned branch. Code,
configs, tests, logs, checkpoints, and metrics are isolated by worktree and run
ID. The `.venv`, COCO data, pretrained weights, and published reproduction
artifacts are shared read-only.

Every CUDA-consuming step, including calibration, PTQ profiling, training,
evaluation, and latency measurement, goes through one campaign scheduler and
holds the same exclusive GPU lock. Admission also checks that no external
compute process owns memory on the target CUDA device. If external contention
appears during a run, the observer marks the affected interval; latency data is
discarded, and formal training/evaluation is stopped and resumed from the last
validated pre-contention checkpoint. CPU code, unit tests, static analysis, and
result parsing may run concurrently. No branch is merged to Git `main` before
the initial comparison is reviewed and a final direction is selected.

## Common Experiment Contract

Every experiment has an immutable run record containing:

- branch commit, dirty-state refusal, resolved config, and config hash;
- parent and teacher checkpoint paths and hashes;
- environment lock and CUDA/PyTorch inventory;
- dataset, annotation, detection-box, and split hashes;
- seed, initialization rule, training schedule, and effective batch size;
- candidate features and quantization/PWL parameters;
- structured training/evaluation logs, best and resumable checkpoint hashes;
- AP metrics, numerical checks, model statistics, and latency protocol.

Artifacts live under `work_dirs/optimization/<route>/<candidate>/<seed>/` and
must never alter `work_dirs/reproduction/`. The campaign reuses the established
background-service principles: exclusive controller lock, atomic state, bounded
retry, append-only event history, validated resume checkpoints, and an observer
that cannot start or complete work. Training requires no network access after
preflight.

Checkpoints are retained at a bounded cadence with the best checkpoint and the
two latest valid resume points. This bounds interruption loss while preventing
the multi-seed campaign from consuming disk without limit.

## Route 1: Accuracy-First Integration

Route 1 is the integration trunk and the recommended path. It does not apply all
approximations at once. It admits one attributable change at a time and keeps a
measured rollback point.

1. Establish a common seed-0 evaluator and profile the unchanged full and
   no-PIF checkpoints under identical flip/no-flip and latency protocols.
2. Import the best structural and numeric candidates only after their isolated
   tests and preliminary full-val evaluation pass.
3. Recover accuracy with supervised heatmap loss plus frozen-teacher
   distillation. The reproduced COCO-B checkpoint is the default high-capacity
   teacher because it uses the same 17-keypoint task and reaches 74.895 AP.
   Teacher heatmaps are aligned to the student's existing decoder contract;
   feature distillation is added only if heatmap distillation is insufficient.
4. Combine no more than one structural candidate and one numeric candidate per
   experiment so any loss remains attributable.
5. Run formal matched-seed training only for the best integrated candidate.

Distillation is a training-time mechanism. It adds no operation to the exported
student graph.

## Route 2: Structural PIF Simplification

Route 2 targets PIF because it contains little arithmetic compared with the
backbone but introduces dynamic top-k, data-dependent gather, short scan
sequences, and three extra Mamba blocks that complicate hardware mapping.

Candidates are evaluated in this order:

1. `no_pif`: use the existing explicit `pif_mode='disabled'` path and validate
   the promising seed-0 result under the common evaluator. The compatible
   training checkpoint still contains unreachable PIF weights; the deployment
   export must remove them, and parameter/byte claims use that pruned export.
2. `static_neighbor_pif`: retain global pose interaction but replace per-image
   top-k with a fixed `17 x 5` table. The table is derived only from teacher
   activations on the COCO training split, versioned, and never calibrated on
   COCO val. This removes runtime top-k and data-dependent indices.
3. `static_graph_lite`: if PIF is still useful, retain the residual/norm
   interface and replace the three short-sequence Mamba mixers. First test a
   fixed row-normalized `17 x 17` graph mix followed by one shared channel
   projection. Only if that fails for a diagnosed locality reason, test a
   depthwise kernel-3 one-dimensional convolution over the versioned anatomical
   order followed by the same channel projection. No wider architecture search
   is admitted in this branch.

The already fixed anatomical scan indices alone are not counted as an
optimization. A structural candidate must eliminate dynamic selection or
remove Mamba blocks from the runtime graph. Shape and checkpoint compatibility
are explicit tests.

## Route 3: SSM-Aware Quantization, PWL, and Binary Q/K

Route 3 begins with observer-only profiling. It records token-wise and
channel-wise ranges at VMamba projections, selective-scan inputs and outputs,
state-transition terms, Transformer Q/K/V, PIF, and heatmap projection. The
profile determines where per-tensor quantization is unsafe.

Candidates progress from least to most invasive:

1. Weight-only 8-bit fake quantization on linear and convolution weights, with
   per-output-channel scales where supported.
2. W8A8 fake quantization on MAC-heavy paths, using per-token or smoothed
   activation scales for layers whose measured outliers reject per-tensor
   scaling. Selective-scan accumulation remains higher precision at this stage.
3. Power-of-two scale candidates for convolution and SSM parameters, admitted
   only after numerical error and full-val checks.
4. PWL replacements for measured nonlinear bottlenecks. Each approximation has
   a declared domain, saturation behavior, segment count, maximum/mean error,
   and differentiable QAT form. `exp`, `softplus`, SiLU, and GELU are not
   replaced as a group; each is evaluated separately.
5. A mechanism-inspired deterministic sign-only Q/K candidate in the exact six
   Transformer layers. This preliminary path uses zero-to-positive sign,
   identity STE, the original floating scale, and floating softmax, V, AV, and
   output projection. It intentionally omits BinaryAttention's learnable
   attention bias and is therefore not a reproduction of the full
   BinaryAttention method. QAT plus self-distillation is a single bounded
   recovery option only after the candidate passes the Stage-B admission gate.
   Because this affects only the small head, it is retained only if its exact
   operation inventory removes meaningful QK multiplications without consuming
   the AP budget.

The Binary Q/K PyTorch path is a software proxy. BinaryAttention's reported
speedup depends on a dedicated bitwise A100 kernel; this candidate has no such
kernel and makes no latency, bitwise-kernel, FPGA-resource, or speedup claim.

PTQ is a screening mechanism. A candidate outside the recoverable preliminary
band may not proceed to a long QAT run without an identified error source.
Quantized storage savings and fake-quant coverage are reported separately from
true integer-kernel latency; fake quantization does not prove an FPGA speedup.

## Evaluation Ladder

### Stage A: Admission and Numerical Tests

Before dataset evaluation, every candidate must pass:

- config construction and checkpoint load audit;
- expected `N x 3 x 256 x 192` input and `N x 17 x 64 x 48` heatmap shape;
- finite forward, backward, loss, and optimizer step;
- identity-mode equivalence for wrappers and observers;
- deterministic fixed-index tables and no validation-set calibration;
- serialization/resume and clean-export tests;
- operation inventory proving the claimed dynamic operation or precision change.

For Binary Q/K, Stage A is an explicit full-model one-batch smoke before any
profile or evaluation. All six canonical `to_qkv.weight` targets must have
finite, non-zero Q/K-slice gradients, changed Q/K parameters, and finite Adam
state after the optimizer step. The exported restricted state must rebuild
with all implicit pretrained initializers neutralized and reproduce state and
output exactly. Profile, evaluation, and latency artifacts are hash-bound to
this smoke artifact and its exact six-layer operation manifest.

Binary admission additionally inherits one public-valid passed PWL Stage-B
result. Its source commit, current candidate-row identity, complete inherited
config closure, checkpoint, schema-v3 fit and sample order, selection policy,
installation report, and installed operation hash must all agree across flip
and no-flip evidence. Missing, stale, alternate, or symlink-aliased authority
fails closed.

### Stage B: Preliminary Screen

Preliminary comparison uses the complete COCO val2017 set, not a favorable
subset:

- unchanged seed-0 full S-V1 and existing seed-0 no-PIF checkpoints;
- structural warm-start or short fine-tune candidates;
- calibrated PTQ candidates;
- QAT/distillation only for candidates with a diagnosed recoverable error.

A candidate with more than 0.3 AP seed-0 drop is normally rejected. A candidate
may remain for accuracy recovery only when the branch documents a specific
mechanism and a bounded next experiment. Stage B never establishes the final
`<0.1 AP` claim.

After Stage B, work pauses before formal long runs and publishes one comparison
table covering accuracy, parameter/byte footprint, operation changes, low-bit
coverage, numerical error, batch-1 latency, and remaining hardware hazards.
The final long-run direction is selected from this evidence.

### Stage C: Formal Matched-Seed Gate

The formal comparison uses seeds 0, 1, and 2. Existing seed-0 reproduction
metrics remain context but do not substitute for the formal baseline. For each
seed `s`, baseline and candidate are both retrained under the optimization
campaign's deterministic comparison protocol: `deterministic=True`, a fixed
worker count, persistent workers disabled, independently seeded data-loader
workers, a seeded sampler, and a recorded per-epoch sample-order hash. A short
preflight repeats the loader trace and refuses admission unless its order hashes
match. Baseline and candidate also use the same augmentation seed policy,
initialization source, 300-epoch schedule, effective batch, detections,
evaluator, and test-time augmentation. Custom selective-scan kernels are tested
for repeatability; any remaining nondeterminism is recorded as a protocol
limitation rather than hidden behind the paired statistic. Define the paired
drop in AP points as:

```text
d_s = AP_full_S-V1,s - AP_candidate,s
mean_drop = mean(d_0, d_1, d_2)
```

The accuracy gate passes only when both conditions hold:

```text
mean_drop < 0.1 AP
max(d_0, d_1, d_2) < 0.3 AP
```

Mean, standard deviation, individual paired differences, and a paired 95%
confidence interval are always reported. If the three-seed confidence interval
intersects 0.1 AP, seeds 3 and 4 are added and the same mean and maximum rules
are recomputed over all five seeds. Seeds are fixed before training and are
never searched or discarded for favorable results.

The point estimate is always paired. The seed-0 value 72.832 is not converted
into one absolute threshold for unrelated seeds.

### Stage D: Secondary Quality and Hardware Proxies

Passing AP is necessary but not sufficient. The comparison also reports:

- AP50, AP75, APM, APL, and AR for abnormal subgroup regressions;
- flip and no-flip AP;
- parameter count, FP32 and exported low-bit weight bytes;
- measured low-bit operation coverage and accumulator precision;
- removal of dynamic top-k, gather, scan, or nonlinear operations;
- traced FLOPs with unsupported-operator inventory;
- batch-1 warm latency distributions for head-only and full model;
- peak GPU memory only as a software measurement, not an FPGA resource claim;
- numerical error against the floating reference at module and full-model
  outputs.

When representative sports video becomes available, temporal jitter and
track-conditioned missing-keypoint rate are added as downstream qualification
metrics. They do not replace COCO AP.

## Pareto Selection Rule

Only candidates that pass the formal accuracy gate enter the final Pareto set.
A retained candidate must also provide a concrete hardware-facing improvement:

- eliminate runtime data-dependent PIF selection or one or more PIF Mamba
  blocks; or
- store at least 80% of MAC-bearing weights at 8 bits or below with a measured
  model-byte reduction; or
- replace a declared nonlinear operator with a bounded PWL implementation that
  survives the AP gate; or
- provide the canonical six-layer Binary Q/K manifest proving 6,489,600
  theoretical floating QK multiplications are replaced, while explicitly
  recording that no bitwise kernel or measured speedup exists.

Among passing candidates, selection prefers the simplest exported graph first,
then lower model bytes and greater low-bit MAC coverage. GPU latency breaks ties
but is not treated as a prediction of FPGA latency. If no candidate passes, the
baseline remains the result; the AP threshold is not relaxed.

## YOLO and Trajectory Interface Safeguards

Integration examples use `mmpose.apis.inference_topdown()` or a small typed
adapter around it, because the current `Pose2DInferencer.preprocess_single()`
clears externally supplied boxes on the top-down path. The adapter accepts
`xyxy` boxes, detector scores, and `track_id`; it preserves detector scores
outside `inference_topdown()`, which currently assigns every passed box a score
of one.

Tests verify that each YOLO box produces exactly one pose result, coordinates
map back to the original frame, and `track_id` ordering is preserved. These are
interface tests only; YOLO and tracker accuracy are outside the model gate.

## Failure Handling and Stop Conditions

An experiment fails closed on dirty source, changed dataset/config hashes,
unloadable checkpoint, non-finite values, missing evaluation records, or a
claimed optimization absent from the exported operation inventory. Transient
I/O or worker failures may resume from the latest validated checkpoint within a
bounded retry budget. Deterministic CUDA, shape, quantization, or data failures
do not retry indefinitely.

The campaign stops a branch when:

- its candidate exceeds the preliminary AP band without an attributable
  recovery hypothesis;
- the claimed hardware simplification is absent or moves cost elsewhere;
- numerical error is unbounded over observed calibration and validation ranges;
- the formal AP or stability gate fails; or
- the experiment would require changing YOLO, the tracker, or FPGA architecture
  before the model comparison is complete.

## Deliverables

The algorithm phase produces:

1. Three isolated branches with tests and immutable experiment manifests.
2. A seed-0 preliminary comparison and a recorded route-selection decision.
3. Formal matched-seed checkpoints and metrics for the selected candidates.
4. An exported floating and, where applicable, fake-quantized model with an
   operation/precision manifest.
5. A Pareto report stating which candidates passed, failed, or remained
   statistically unresolved and why.
6. A hardware handoff contract listing tensor shapes, ranges, precision by
   operator, static tables, remaining nonlinearities, and unsupported dynamic
   operations.

No checkpoint or branch is merged to Git `main`, and no FPGA performance claim
is made, until the preliminary comparison has been reviewed and the selected
formal candidate has passed the accuracy gate.

## Primary Research References

- MambaPose paper: `reference/pdf/icme-2025-mambapose.pdf`
- VMamba: <https://proceedings.neurips.cc/paper_files/paper/2024/hash/baa2da9ae4bfed26520bb61d259a3653-Abstract.html>
- BinaryAttention: <https://openaccess.thecvf.com/content/CVPR2026/html/Xiao_BinaryAttention_One-Bit_QK-Attention_for_Vision_and_Diffusion_Transformers_CVPR_2026_paper.html>
- PTQ4VM: <https://arxiv.org/abs/2412.20386>
- Quamba: <https://arxiv.org/abs/2410.13229>
- FastMamba: <https://arxiv.org/abs/2505.18975>
- ViM-Q: <https://arxiv.org/abs/2605.01935>
- DWPose: <https://arxiv.org/abs/2307.15880>
