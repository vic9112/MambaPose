# MambaPose Hardware-Friendly Optimization Stage A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a reproducible three-route Stage A/B screen that identifies hardware-friendly COCO S-V1 candidates without making a formal `<0.1 AP` claim before matched-seed training.

**Architecture:** A shared `mambapose_opt` package owns immutable candidate records, GPU exclusivity, profiling, metrics, and comparison. Three isolated worktrees implement accuracy-first integration, structural PIF, and SSM-aware numeric optimization. All CUDA work is serialized, and execution stops after full COCO val seed-0 comparison so a later plan can target only the selected formal candidate.

**Tech Stack:** Python 3.11, PyTorch 2.7.1/CUDA 12.8, MMEngine 0.10.7, MMPose 1.3.1, pytest, JSON/Markdown evidence, user systemd, NVIDIA RTX 5090.

**Spec:** `docs/superpowers/specs/2026-08-27-mambapose-hardware-friendly-model-optimization-design.md`

## Global Constraints

- Model branch root is release commit `61b7ff8f476dc284f2684fbb2e5b968291431881`; checkpoint training provenance is `1f4364d62279bf9fe3ac55e0e2d57339036aeb81`.
- Primary model is COCO S-V1 at `256 x 192` with 17 keypoints and paper-comparable `flip_test=True`.
- Stage B uses complete COCO val2017 and the existing paper-comparable detection JSON.
- Stage B may retain candidates within 0.3 AP for bounded recovery but cannot establish the formal `<0.1 AP` claim.
- Formal matched-seed training begins only after the Stage B route-selection decision.
- Every CUDA-consuming step uses one lock and fails closed on external GPU ownership.
- `.venv`, `data`, `pretrained`, and `work_dirs/reproduction` are shared read-only; new artifacts stay under `work_dirs/optimization`.
- Git `main` remains the publication baseline. Work occurs on `algo/accuracy-first`, `algo/structural-pif`, and `algo/ssm-quant-pwl`.
- Fake quantization, GPU latency, and operation counts are proxy evidence, not FPGA resource or speed claims.
- Tests are written first and each task ends in a focused commit.

---

### Task 1: Bootstrap Worktrees and the Shared Candidate Contract

**Ownership:** Primary agent on `algo/accuracy-first`. Cherry-pick the reviewed contract commit into both route branches.

**Files:**
- Create: `mambapose_opt/__init__.py`
- Create: `mambapose_opt/schema.py`
- Create: `mambapose_opt/inventory.py`
- Create: `optimization/candidates.json`
- Create: `tools/optimization/profile_model.py`
- Create: `tests/test_optimization/test_schema.py`
- Create: `tests/test_optimization/test_inventory.py`

**Interfaces:**
- Produces: `JSONScalar = str | int | float | bool | None`
- Produces: exception `CandidateManifestError(ValueError)`
- Produces: `CandidateSpec.from_dict(value: Mapping[str, Any]) -> CandidateSpec`
- Produces: `load_candidate_manifest(path: Path | str) -> tuple[CandidateSpec, ...]`
- Produces: `count_parameters(model: nn.Module) -> ParameterInventory`
- Produces: `collect_module_inventory(model: nn.Module) -> tuple[ModuleRecord, ...]`

- [ ] **Step 1: Create all three worktrees through the worktree skill**

Use the design commit as branch point, then verify:

```bash
git worktree list --porcelain
git -C .worktrees/algo-accuracy-first status --short --branch
git -C .worktrees/algo-structural-pif status --short --branch
git -C .worktrees/algo-ssm-quant-pwl status --short --branch
```

Expected: each worktree names its assigned `algo/*` branch and is clean.

- [ ] **Step 2: Write failing strict-schema tests**

```python
def test_candidate_manifest_rejects_unknown_fields(tmp_path):
    from mambapose_opt.schema import CandidateManifestError, load_candidate_manifest
    path = tmp_path / 'candidates.json'
    path.write_text(json.dumps({
        'schema_version': 1,
        'candidates': [{
            'id': 'full-s-v1', 'route': 'baseline', 'kind': 'float',
            'config': 'configs/reproduction/coco_s_v1.py',
            'checkpoint': 'work_dirs/reproduction/runs/coco-s-v1/best_coco_AP_epoch_300.pth',
            'checkpoint_sha256': 'a' * 64, 'seed': 0,
            'features': {}, 'unexpected': True,
        }],
    }))
    with pytest.raises(CandidateManifestError, match='unknown fields'):
        load_candidate_manifest(path)
```

Also assert rejection of duplicate IDs, absolute/traversing paths, invalid hashes, invalid routes, and non-integer seeds.

- [ ] **Step 3: Run the schema test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_optimization/test_schema.py -q`

Expected: FAIL because `mambapose_opt.schema` does not exist.

- [ ] **Step 4: Implement the strict candidate schema**

```python
@dataclass(frozen=True)
class CandidateSpec:
    id: str
    route: Literal['baseline', 'accuracy-first', 'structural-pif', 'ssm-quant-pwl']
    kind: Literal['float', 'structural', 'fake-quant', 'pwl', 'binary-qk', 'integrated']
    config: Path
    checkpoint: Path
    checkpoint_sha256: str
    seed: int
    features: Mapping[str, JSONScalar]
```

Reject unknown keys and unsafe paths. The manifest must not contain shell commands.
Define `JSONScalar = str | int | float | bool | None`; reject nested feature
values so the recorded feature set remains stable and comparable.

- [ ] **Step 5: Write failing inventory tests**

```python
def test_parameter_inventory_separates_trainable_and_total():
    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2, bias=False))
    model[1].weight.requires_grad_(False)
    inv = count_parameters(model)
    assert inv.total == 21
    assert inv.trainable == 15
    assert inv.bytes_by_dtype['torch.float32'] == 84
```

Also assert explicit hazard records for `PoseInteraction` dynamic top-k and VMamba custom scans.

- [ ] **Step 6: Run inventory tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_optimization/test_inventory.py -q`

Expected: FAIL because inventory interfaces do not exist.

- [ ] **Step 7: Implement inventory and profiling CLI**

Implement `ParameterInventory(total, trainable, bytes_by_dtype, by_prefix)` and `ModuleRecord(name, kind, parameters, hazard)`. Explicitly recognize `PoseInteraction`, `Attention`, `CrossScan`, `SelectiveScan`, and `CrossMerge`. Do not estimate unsupported custom-op FLOPs.

`profile_model.py` loads config/checkpoint, validates its hash, and atomically writes commit, config, input/output shapes, parameter breakdown, and module records.

- [ ] **Step 8: Add frozen candidates and run tests**

Add `full-s-v1`, `no-pif-s-v1`, and `coco-b-teacher` with exact hashes from `reproduction/checkpoints.json`.

```bash
.venv/bin/python -m pytest tests/test_optimization/test_schema.py tests/test_optimization/test_inventory.py -q
.venv/bin/python -m pytest tests/test_reproduction/test_results.py tests/test_reproduction/test_pif.py -q
```

Expected: all pass.

- [ ] **Step 9: Commit and propagate the contract**

```bash
git add mambapose_opt optimization tools/optimization/profile_model.py tests/test_optimization
git commit -m "feat: add optimization candidate and inventory contract"
```

Cherry-pick that exact commit into `algo/structural-pif` and `algo/ssm-quant-pwl` and rerun focused tests there.

---

### Task 2: Enforce GPU Exclusivity and Durable Execution

**Ownership:** Primary agent on `algo/accuracy-first`. Propagate the reviewed commit to route branches.

**Files:**
- Create: `mambapose_opt/gpu_guard.py`
- Create: `mambapose_opt/controller.py`
- Create: `mambapose_opt/observe.py`
- Create: `tools/optimization/run_campaign.py`
- Create: `tools/optimization/observe.py`
- Create: `tools/optimization/install_user_service.sh`
- Create: `systemd/mambapose-optimization.service`
- Create: `systemd/mambapose-optimization-observer.service`
- Create: `systemd/mambapose-optimization-observer.timer`
- Create: `tests/test_optimization/test_gpu_guard.py`
- Create: `tests/test_optimization/test_controller.py`
- Create: `tests/test_optimization/test_observer.py`
- Create: `tests/test_optimization/test_systemd.py`

**Interfaces:**
- Produces: exceptions `ExternalGpuContention` and `ConcurrentCudaStage`
- Produces: dataclasses `GpuProcess`, `GpuLease`, and `StageOutcome`
- Produces: `query_compute_processes(device_index: int) -> tuple[GpuProcess, ...]`
- Produces: `exclusive_cuda_stage(lock_path: Path, device_index: int, allowed_pids: Collection[int]) -> ContextManager[GpuLease]`
- Produces: `OptimizationController.run_next() -> StageOutcome`
- Produces: `observe(root: Path, heartbeat_max_age: float) -> dict[str, Any]`

- [ ] **Step 1: Write failing GPU ownership tests**

```python
def test_external_gpu_owner_blocks_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_guard, 'query_compute_processes', lambda _: (
        GpuProcess(pid=9001, used_memory_mib=4096, command='foreign.py'),
    ))
    with pytest.raises(ExternalGpuContention, match='9001'):
        with exclusive_cuda_stage(tmp_path / 'gpu.lock', 0, {os.getpid()}):
            pass

def test_same_lock_serializes_stage_kinds(tmp_path):
    with exclusive_cuda_stage(tmp_path / 'gpu.lock', 0, {os.getpid()}):
        with pytest.raises(ConcurrentCudaStage):
            with exclusive_cuda_stage(tmp_path / 'gpu.lock', 0, {os.getpid()}):
                pass
```

- [ ] **Step 2: Run GPU tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_optimization/test_gpu_guard.py -q`

Expected: FAIL because `mambapose_opt.gpu_guard` does not exist.

- [ ] **Step 3: Implement one CUDA lease**

Use `fcntl.flock` at `work_dirs/optimization/gpu.lock`. Invoke `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits` without a shell. Resolve controller descendants through `/proc`; any other compute PID blocks admission. Record stage ID, PID, boot ID, and timestamp in the lock.

- [ ] **Step 4: Write controller and observer tests**

Assert that only validated artifacts complete a stage, contention returns 75, schema/hash failure returns 78, retry lineage is append-only, and the observer never starts/stops processes or writes `state.json`.

- [ ] **Step 5: Run controller/observer tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_optimization/test_controller.py tests/test_optimization/test_observer.py tests/test_optimization/test_systemd.py -q`

Expected: FAIL because the controller, observer, and service units do not exist.

- [ ] **Step 6: Implement controller, observer, and service units**

Reuse `mambapose_repro.state.StateStore` and orchestrator failure fingerprints. Support `profile`, `calibrate`, `train`, `evaluate`, `latency`, and `compare`; all except `compare` acquire the CUDA lease. Keep best plus two valid resume checkpoints.

The campaign unit uses `RestartPreventExitStatus=78` and `KillMode=control-group`. The timer runs the read-only observer.

- [ ] **Step 7: Run focused and regression tests**

```bash
.venv/bin/python -m pytest tests/test_optimization/test_gpu_guard.py tests/test_optimization/test_controller.py tests/test_optimization/test_observer.py tests/test_optimization/test_systemd.py -q
.venv/bin/python -m pytest tests/test_reproduction/test_orchestrator.py tests/test_reproduction/test_observer.py tests/test_reproduction/test_systemd_units.py -q
```

Expected: all pass.

- [ ] **Step 8: Commit durable execution**

```bash
git add mambapose_opt tools/optimization systemd tests/test_optimization
git commit -m "feat: serialize and monitor optimization CUDA stages"
```

---

### Task 3: Add Deterministic Evaluation and Baseline Profiling

**Ownership:** Primary agent on `algo/accuracy-first`. Propagate evaluator/profiler code to route branches.

**Files:**
- Create: `mambapose_opt/determinism.py`
- Create: `mambapose_opt/evaluation.py`
- Create: `mambapose_opt/latency.py`
- Create: `configs/optimization/coco_s_v1_deterministic.py`
- Create: `configs/optimization/coco_s_v1_no_pif_seed0.py`
- Create: `tools/optimization/trace_dataloader.py`
- Create: `tools/optimization/evaluate_candidate.py`
- Create: `tools/optimization/measure_latency.py`
- Create: `mambapose_opt/yolo_adapter.py`
- Create: `tests/test_optimization/test_determinism.py`
- Create: `tests/test_optimization/test_evaluation.py`
- Create: `tests/test_optimization/test_latency.py`
- Create: `tests/test_optimization/test_yolo_adapter.py`

**Interfaces:**
- Produces: `seed_worker(worker_id: int) -> None`
- Produces: `OrderHashRecorder.update(epoch: int, sample_ids: Sequence[int]) -> None`
- Produces: `load_coco_metrics(path: Path) -> CocoMetrics`
- Produces: `CandidateResult.from_artifacts(root: Path) -> CandidateResult`
- Produces: `measure_latency(callable, warmup: int, repeats: int) -> LatencySummary`
- Produces: immutable dataclasses `TrackedBox` and `TrackedPose`
- Produces: `infer_tracked_poses(model, frame: np.ndarray, boxes: Sequence[TrackedBox]) -> list[TrackedPose]`

- [ ] **Step 1: Write failing determinism tests**

```python
def test_order_hash_is_repeatable_and_epoch_sensitive():
    a = OrderHashRecorder()
    b = OrderHashRecorder()
    a.update(0, [4, 1, 9])
    b.update(0, [4, 1, 9])
    assert a.hexdigest(0) == b.hexdigest(0)
    b.update(1, [4, 1, 9])
    assert b.hexdigest(1) != b.hexdigest(0)

def test_optimization_config_enables_determinism():
    cfg = Config.fromfile('configs/optimization/coco_s_v1_deterministic.py')
    assert cfg.randomness == dict(seed=0, deterministic=True)
    assert cfg.train_dataloader.persistent_workers is False
```

- [ ] **Step 2: Run determinism tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_optimization/test_determinism.py -q`

Expected: FAIL because the modules/config do not exist.

- [ ] **Step 3: Implement deterministic controls**

Derive from `configs/reproduction/coco_s_v1.py`, set deterministic mode, fixed worker count, and disabled persistent workers. Seed Python, NumPy, and Torch from the worker initial seed. `trace_dataloader.py` writes per-epoch image-ID order hashes and repeats preflight traces.

- [ ] **Step 4: Write failing metric and latency tests**

```python
def test_metric_loader_requires_primary_fields(tmp_path):
    path = tmp_path / 'metrics.json'
    path.write_text('{"coco/AP": 72.8}')
    with pytest.raises(MetricError, match='coco/AP50'):
        load_coco_metrics(path)

def test_latency_summary_reports_tail():
    summary = LatencySummary.from_samples_ms([1.0, 1.1, 1.2, 9.0])
    assert summary.median_ms == pytest.approx(1.15)
    assert summary.p95_ms > summary.median_ms
```

- [ ] **Step 5: Run metric/latency tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_optimization/test_evaluation.py tests/test_optimization/test_latency.py -q`

Expected: FAIL because the metric and latency interfaces do not exist.

- [ ] **Step 6: Implement normalized metrics and latency**

Require AP, AP50, AP75, APM, APL, AR, and checkpoint/config/data hashes. Measure full-model batch-1 with CUDA events, 50 warmups, 200 synchronized iterations, and median/p90/p95. Record flip and no-flip separately under a quiescent lease.

Define `CandidateResult` as the immutable normalized row containing candidate
ID, route, COCO metrics, flip mode, profile/latency summaries, provenance
hashes, calibration split, GPU lease record, and artifact paths.

- [ ] **Step 7: Write the failing YOLO adapter contract test**

```python
def test_yolo_adapter_preserves_track_order_and_scores(fake_model):
    boxes = [
        TrackedBox(track_id=9, xyxy=(1., 2., 20., 40.), score=0.91),
        TrackedBox(track_id=4, xyxy=(3., 5., 22., 44.), score=0.73),
    ]
    poses = infer_tracked_poses(
        fake_model, np.zeros((64, 64, 3), dtype=np.uint8), boxes,
        inference_fn=fake_inference_topdown)
    assert [pose.track_id for pose in poses] == [9, 4]
    assert [pose.detector_score for pose in poses] == [0.91, 0.73]
```

- [ ] **Step 8: Run the YOLO adapter test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_optimization/test_yolo_adapter.py -q`

Expected: FAIL because `mambapose_opt.yolo_adapter` does not exist.

- [ ] **Step 9: Implement the YOLO adapter**

The adapter calls `mmpose.apis.inference_topdown()` with one ordered `xyxy`
array, requires one returned pose per box, and carries `track_id` and detector
score beside the pose result because `inference_topdown()` overwrites bbox
scores. It does not instantiate `Pose2DInferencer`.

- [ ] **Step 10: Evaluate/profile frozen seed-0 references**

```bash
.venv/bin/python tools/optimization/profile_model.py configs/reproduction/coco_s_v1.py work_dirs/reproduction/runs/coco-s-v1/best_coco_AP_epoch_300.pth --output work_dirs/optimization/baseline/full-s-v1/profile.json
.venv/bin/python tools/optimization/evaluate_candidate.py --candidate full-s-v1 --flip --output work_dirs/optimization/baseline/full-s-v1/flip/metrics.json
.venv/bin/python tools/optimization/evaluate_candidate.py --candidate full-s-v1 --no-flip --output work_dirs/optimization/baseline/full-s-v1/no-flip/metrics.json
.venv/bin/python tools/optimization/evaluate_candidate.py --candidate no-pif-s-v1 --flip --output work_dirs/optimization/baseline/no-pif-s-v1/flip/metrics.json
```

Expected: flip AP agrees with 72.83223444297235 and 73.07620686556272 within evaluator serialization precision.

- [ ] **Step 11: Run tests and commit**

```bash
.venv/bin/python -m pytest tests/test_optimization/test_determinism.py tests/test_optimization/test_evaluation.py tests/test_optimization/test_latency.py tests/test_optimization/test_yolo_adapter.py -q
git add mambapose_opt configs/optimization tools/optimization tests/test_optimization
git commit -m "feat: add deterministic optimization evaluation"
```

---

### Task 4: Implement Route 2 Structural PIF Candidates

**Ownership:** Structural subagent on `algo/structural-pif` only.

**Files:**
- Modify: `mmpose/models/heads/heatmap_heads/pif.py`
- Modify: `mmpose/models/heads/heatmap_heads/tokenbase.py`
- Modify: `mmpose/models/heads/heatmap_heads/mamba_token_head.py`
- Create: `mambapose_opt/static_neighbors.py`
- Create: `mambapose_opt/export.py`
- Create: `configs/optimization/structural/static_neighbor_pif.py`
- Create: `configs/optimization/structural/static_graph_lite.py`
- Create: `tools/optimization/calibrate_static_neighbors.py`
- Create: `tools/optimization/export_no_pif.py`
- Create: `tests/test_optimization/test_static_neighbors.py`
- Create: `tests/test_optimization/test_structural_pif.py`
- Create: `tests/test_optimization/test_no_pif_export.py`

**Interfaces:**
- Produces: `StaticNeighborAccumulator.update(tokens: Tensor) -> None`
- Produces: `StaticNeighborAccumulator.indices(k: int = 5) -> Tensor` with shape `[17, 5]`
- Produces: PIF modes `static_neighbor` and `static_graph_lite`
- Produces: `export_pruned_no_pif(model: nn.Module) -> nn.Module`

- [ ] **Step 1: Write failing calibration tests**

```python
def test_static_neighbors_are_train_only_and_deterministic():
    acc = StaticNeighborAccumulator(num_keypoints=17, split='train2017')
    acc.update(torch.eye(17).unsqueeze(0))
    assert torch.equal(acc.indices(5), acc.indices(5))
    assert acc.indices(5).shape == (17, 5)
    with pytest.raises(ValueError, match='train2017'):
        StaticNeighborAccumulator(num_keypoints=17, split='val2017')
```

- [ ] **Step 2: Run calibration tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_optimization/test_static_neighbors.py -q`

Expected: FAIL because `StaticNeighborAccumulator` does not exist.

- [ ] **Step 3: Implement train-only accumulation**

Accumulate FP64 sums of normalized `17 x 17` keypoint-token similarity from frozen COCO-B teacher activations. Resolve ties by joint index. Save indices, sample count, split, teacher/config hashes, and accumulator hash.

- [ ] **Step 4: Write failing structural/export tests**

Assert static indices are buffers, runtime `torch.topk` is absent in `static_neighbor`, full mode remains bit-exact, lite output is finite `[B,17,256]`, and exported no-PIF state has no `pose_interaction.*` while matching disabled-mode heatmaps exactly.

- [ ] **Step 5: Run structural/export tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_optimization/test_structural_pif.py tests/test_optimization/test_no_pif_export.py -q`

Expected: FAIL because the new modes and export do not exist.

- [ ] **Step 6: Implement bounded structural modes**

`static_neighbor` replaces only per-image top-k. `static_graph_lite` uses a fixed row-normalized `17 x 17` graph mix, one shared `nn.Linear(dim, dim)`, existing residual scale/dropout, and LayerNorm. Keep the existing full/ablation modes unchanged.

- [ ] **Step 7: Implement pruned no-PIF export**

Build an export head without `PoseInteraction`, strictly load reachable weights, and require disabled/export heatmaps to match at `rtol=0, atol=0`.

- [ ] **Step 8: Run Stage A and commit**

```bash
.venv/bin/python -m pytest tests/test_optimization/test_static_neighbors.py tests/test_optimization/test_structural_pif.py tests/test_optimization/test_no_pif_export.py tests/test_reproduction/test_pif.py -q
git add mmpose/models/heads/heatmap_heads mambapose_opt configs/optimization/structural tools/optimization tests/test_optimization
git commit -m "feat: add hardware-friendly PIF candidates"
```

- [ ] **Step 9: Run Route 2 Stage B**

Evaluate pruned no-PIF first. Calibrate static neighbors on train2017, warm-start from full S-V1, run the manifest-bounded short fine-tune, and evaluate complete COCO val. Run graph-lite only when results show PIF information remains useful. Store normalized evidence under `work_dirs/optimization/structural-pif`.

---

### Task 5: Implement Route 3 Numeric Candidates

**Ownership:** Quantization subagent on `algo/ssm-quant-pwl` only.

**Files:**
- Create: `mmpose/models/utils/hardware_friendly/__init__.py`
- Create: `mmpose/models/utils/hardware_friendly/observers.py`
- Create: `mmpose/models/utils/hardware_friendly/fake_quant.py`
- Create: `mmpose/models/utils/hardware_friendly/pwl.py`
- Create: `mmpose/models/utils/hardware_friendly/binary_qk.py`
- Modify: `mmpose/models/heads/heatmap_heads/tokenbase.py`
- Modify: `mmpose/models/heads/heatmap_heads/mamba_token_head.py`
- Create: `mambapose_opt/numeric_conversion.py`
- Create: `configs/optimization/numeric/w8_weight_only.py`
- Create: `configs/optimization/numeric/w8a8.py`
- Create: `configs/optimization/numeric/binary_qk.py`
- Create: `tools/optimization/calibrate_numeric.py`
- Create: `tests/test_optimization/test_observers.py`
- Create: `tests/test_optimization/test_fake_quant.py`
- Create: `tests/test_optimization/test_pwl.py`
- Create: `tests/test_optimization/test_binary_qk.py`

**Interfaces:**
- Produces: `ActivationRangeObserver(granularity: Literal['tensor','channel','token'])`
- Produces: immutable dataclasses `QuantSpec`, `QuantPolicy`, and `ConversionReport`
- Produces: `FakeQuantLinear.from_float(module: nn.Linear, spec: QuantSpec)`
- Produces: `convert_for_fake_quant(model: nn.Module, policy: QuantPolicy) -> ConversionReport`
- Produces: `PiecewiseLinearApproximation(breakpoints, slopes, intercepts)`
- Produces: attention `qk_mode` in `{'float', 'binary'}`

- [ ] **Step 1: Write failing observer/equivalence tests**

```python
def test_token_observer_keeps_one_range_per_token():
    obs = ActivationRangeObserver('token')
    obs(torch.tensor([[[1., -2.], [4., -3.]]]))
    assert obs.max_abs.tolist() == [[2.0, 4.0]]

def test_disabled_fake_quant_is_bit_exact():
    linear = nn.Linear(8, 4).eval()
    proxy = FakeQuantLinear.from_float(linear, QuantSpec(enabled=False))
    x = torch.randn(3, 8)
    torch.testing.assert_close(proxy(x), linear(x), rtol=0, atol=0)
```

- [ ] **Step 2: Run observer/fake-quant tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_optimization/test_observers.py tests/test_optimization/test_fake_quant.py -q`

Expected: FAIL because the hardware-friendly numeric modules do not exist.

- [ ] **Step 3: Implement observers and W8 fake quant**

Support symmetric int8, per-output-channel weight scales, and tensor/channel/token activation observers. Keep FP32 master weights. Return converted/skipped counts; never wrap normalization, softmax, or selective-scan accumulation silently.

- [ ] **Step 4: Write bounded PWL tests**

Test sorted breakpoints, saturation, continuity, segment selection, and max/mean error on a supplied grid.

- [ ] **Step 5: Run PWL tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_optimization/test_pwl.py -q`

Expected: FAIL because PWL interfaces do not exist.

- [ ] **Step 6: Implement bounded PWL fitting**

Implement deterministic `fit_pwl(reference_fn, domain, segments, grid_points)`. Fit SiLU, GELU, softplus, and exp separately; activate only one per candidate.

- [ ] **Step 7: Write Binary Q/K tests**

```python
def test_binary_qk_signed_dot_reference():
    q = torch.tensor([[[[1., -2., 3., -4.]]]])
    k = torch.tensor([[[[-1., -2., 3., 4.]]]])
    assert binary_qk_logits(q, k).item() == 0.0

def test_float_qk_mode_keeps_checkpoint_keys(float_attention, configurable_attention):
    configurable_attention.load_state_dict(float_attention.state_dict(), strict=True)
    x = torch.randn(2, 65, 256)
    torch.testing.assert_close(
        configurable_attention(x)[0], float_attention(x)[0], rtol=0, atol=0)
```

- [ ] **Step 8: Run Binary Q/K tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_optimization/test_binary_qk.py -q`

Expected: FAIL because binary Q/K interfaces do not exist.

- [ ] **Step 9: Implement head-local Binary Q/K**

Thread `qk_mode='float'` through head, TokenPose, Transformer, and Attention. Binary mode applies deterministic STE sign to Q/K and signed-dot logits; scale, softmax, V, and output projection stay multi-bit. Float mode preserves keys and outputs exactly.

- [ ] **Step 10: Calibrate ranges on train2017**

Hook VMamba projections, scan boundaries, Transformer Q/K/V, PIF, and heatmap projection. Save ranges, percentiles, outlier ratios, sample count, split, and hashes. Reject `val2017`.

- [ ] **Step 11: Run Stage A and commit**

```bash
.venv/bin/python -m pytest tests/test_optimization/test_observers.py tests/test_optimization/test_fake_quant.py tests/test_optimization/test_pwl.py tests/test_optimization/test_binary_qk.py -q
git add mmpose/models/utils/hardware_friendly mmpose/models/heads/heatmap_heads mambapose_opt configs/optimization/numeric tools/optimization tests/test_optimization
git commit -m "feat: add Mamba-aware numeric optimization probes"
```

- [ ] **Step 12: Run Route 3 Stage B**

Evaluate W8 weight-only first, W8A8 only where ranges support it, one PWL function at a time by ascending reference error, and Binary Q/K last. A candidate beyond 0.3 AP needs a specific error report before one bounded QAT/distillation recovery. Store normalized evidence under `work_dirs/optimization/ssm-quant-pwl`.

---

### Task 6: Implement Accuracy-First Heatmap Distillation

**Ownership:** Primary agent on `algo/accuracy-first` after route APIs stabilize.

**Files:**
- Create: `mmpose/models/distillers/mambapose_heatmap_distiller.py`
- Modify: `mmpose/models/distillers/__init__.py`
- Create: `configs/optimization/accuracy_first/distill_s_v1_from_b.py`
- Create: `configs/optimization/accuracy_first/integrated_candidate.py`
- Create: `tools/optimization/export_student.py`
- Create: `tests/test_optimization/test_mambapose_distiller.py`
- Create: `tests/test_optimization/test_integration_policy.py`

**Interfaces:**
- Produces: `MambaPoseHeatmapDistiller(teacher, student, heatmap_loss_weight)`
- Produces: immutable `IntegrationSpec(features, parent_results)`
- Produces: `validate_integration_features(spec: IntegrationSpec) -> None`
- Produces: `export_student_checkpoint(distiller, path: Path) -> Path`

- [ ] **Step 1: Write failing distiller tests**

```python
def test_teacher_is_frozen_and_eval_only(distiller):
    assert distiller.teacher.training is False
    assert all(not p.requires_grad for p in distiller.teacher.parameters())

def test_export_contains_no_teacher(distiller, tmp_path):
    path = export_student_checkpoint(distiller, tmp_path / 'student.pth')
    state = torch.load(path, weights_only=False)['state_dict']
    assert state
    assert not any(key.startswith('teacher.') for key in state)
```

- [ ] **Step 2: Run distiller tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_optimization/test_mambapose_distiller.py -q`

Expected: FAIL because the MambaPose heatmap distiller does not exist.

- [ ] **Step 3: Implement heatmap distillation**

Build teacher/student from configs, hash-validate the COCO-B teacher, freeze it, and compute teacher heatmaps under `torch.no_grad()`. Student loss is existing supervised heatmap MSE plus separately logged teacher/student heatmap MSE. Do not reuse `DWPoseDistiller` because it expects SimCC outputs.

- [ ] **Step 4: Write integration policy tests**

Reject more than one structural or numeric feature. Allow one structural plus one numeric feature only when both parent Stage B artifacts exist and their hashes are recorded.

- [ ] **Step 5: Run integration policy tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_optimization/test_integration_policy.py -q`

Expected: FAIL because `validate_integration_features` does not exist.

- [ ] **Step 6: Implement the integration policy**

Validate feature counts, require parent result paths and SHA-256 values, and reject parent artifacts whose candidate IDs do not match the enabled features.

- [ ] **Step 7: Run real-batch smoke and commit**

Verify finite losses, gradients only on student, checkpoint round-trip, student export, and matching exported heatmaps.

```bash
.venv/bin/python -m pytest tests/test_optimization/test_mambapose_distiller.py tests/test_optimization/test_integration_policy.py -q
git add mmpose/models/distillers configs/optimization/accuracy_first tools/optimization tests/test_optimization
git commit -m "feat: add accuracy-first MambaPose distillation"
```

- [ ] **Step 8: Run bounded recovery/integration**

Distill the best recoverable candidate from each route. Do not combine structural and numeric changes until isolated full-val results exist. Admit at most one integrated Stage B candidate before comparison.

---

### Task 7: Compare Stage B Results and Stop at Selection

**Ownership:** Primary agent on `algo/accuracy-first`; an independent reviewer audits evidence.

**Files:**
- Create: `mambapose_opt/compare.py`
- Create: `tools/optimization/compare_candidates.py`
- Create: `tests/test_optimization/test_compare.py`
- Generate: `work_dirs/optimization/comparison/stage-b-results.json`
- Create: `docs/optimization/stage-b-results.md`

**Interfaces:**
- Produces: `compare_candidates(baseline: CandidateResult, candidates: Sequence[CandidateResult]) -> ComparisonReport`
- Produces: statuses `screen_pass`, `recoverable`, `screen_fail`, `invalid`
- Produces: JSON and Markdown from identical rows

- [ ] **Step 1: Write failing comparison tests**

```python
def test_stage_b_never_labels_formal_pass():
    report = compare_candidates(BASELINE, [candidate(ap=72.80)])
    assert report.rows[0].status == 'screen_pass'
    assert 'formal_pass' not in report.to_json()

def test_point_three_drop_fails_screen():
    report = compare_candidates(baseline(ap=72.83), [candidate(ap=72.52)])
    assert report.rows[0].drop_ap == pytest.approx(0.31)
    assert report.rows[0].status == 'screen_fail'

def test_missing_hash_is_invalid():
    result = candidate(ap=73.0, checkpoint_sha256=None)
    assert compare_candidates(BASELINE, [result]).rows[0].status == 'invalid'
```

- [ ] **Step 2: Run comparison tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_optimization/test_compare.py -q`

Expected: FAIL because comparison interfaces do not exist.

- [ ] **Step 3: Implement fail-closed comparison**

Require complete COCO metrics, profile, hashes, quiescent-GPU latency evidence, operation inventory, and train-only calibration provenance. Compute contextual seed-0 AP drop but never emit a formal `<0.1 AP` pass.

- [ ] **Step 4: Generate the Pareto table**

Include route, candidate, AP submetrics, flip/no-flip, drop, parameters, exported bytes, low-bit coverage, median/p95 latency, removed/remaining hazards, and reason. Sort by screen status, structural simplicity, model bytes, then low-bit coverage; latency only breaks ties.

- [ ] **Step 5: Run full verification**

```bash
.venv/bin/python -m pytest tests/test_optimization -q
.venv/bin/python -m pytest tests/test_reproduction -q
git diff --check
```

Expected: all pass and comparison hashes validate.

- [ ] **Step 6: Request objective evidence review**

The reviewer verifies full COCO val usage, train-only calibration, matching evaluator settings, uncontaminated latency, honest fake-quant claims, and absence of a formal gate claim.

- [ ] **Step 7: Commit tracked tooling/report**

```bash
git add mambapose_opt tools/optimization tests/test_optimization docs/optimization/stage-b-results.md
git commit -m "docs: compare hardware-friendly MambaPose candidates"
```

Large checkpoints and runtime artifacts remain outside Git.

- [ ] **Step 8: Stop for formal-route selection**

Report success, failure, and recoverable reasons. The next plan retrains deterministic baseline/candidate seeds 0-2, adds seeds 3-4 only when the confidence interval intersects 0.1 AP, and enforces the spec's formal mean/max gate.
