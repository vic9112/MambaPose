# MambaPose Reproduction Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the paper matrix into tested resolved configs, verified datasets/weights, recoverable experiment state, validated metrics, and a non-controlling health observer.

**Architecture:** A small `mambapose_repro` package owns immutable manifests, atomic mutable state, downloads/preflight, checkpoint validation, and monitoring. Paper-derived configs remain explicit files, while PIF variation is a tested model argument. A user-systemd unit only drives the orchestrator; a timer only records health.

**Tech Stack:** Python 3.11, MMEngine/MMPose configs, pytest, JSON/JSONL, systemd user units, `flock`, `nvidia-smi`

**Spec:** `docs/superpowers/specs/2026-08-18-mambapose-paper-reproduction-design.md`

## Global Constraints

- Model/training decisions follow paper, then repository, then recorded assumption.
- CrowdPose S-V2 uses paper depths `[1,2,3,2]`; data root is `data/crowdpose/`.
- Primary seed is exactly 0; all checkpoints save every epoch and retain at least the two newest valid candidates.
- COCO test-dev uses `CocoMetric(format_only=True, outfile_prefix=...)` and is never locally reported as AP.
- A stage is complete only after artifact validation; transient exit is 75 and permanent exit is 78.
- The observer never starts, stops, retries, or completes work.

---

### Task 1: Pose-Interaction Modes

**Files:**
- Create: `mmpose/models/heads/heatmap_heads/pif.py`
- Modify: `mmpose/models/heads/heatmap_heads/tokenbase.py`
- Test: `tests/test_reproduction/test_pif.py`

**Interfaces:**
- Consumes: token tensor `[batch, keypoints, channels]`, dataset keypoint count, and `pif_mode`.
- Produces: `build_scan_indices(num_keypoints: int, mode: str) -> tuple[Tensor, Tensor]` and `PoseInteraction(mode='full')`.

- [ ] **Step 1: Write failing index/mode tests**

```python
@pytest.mark.parametrize('keypoints', [14, 17])
@pytest.mark.parametrize('mode', ['full', 'disabled', 'no_prior', 'no_cycling'])
def test_pif_modes_preserve_keypoint_output_shape(keypoints, mode):
    layer = PoseInteraction(dim=256, num_keypoints=keypoints, mode=mode)
    x = torch.randn(2, keypoints, 256)
    assert layer(x).shape == x.shape

def test_scan_indices_are_bounded_and_reconstruct_all_keypoints():
    scan, restore = build_scan_indices(17, 'full')
    assert scan.min() >= 0 and scan.max() < 17
    assert set(restore.tolist()) == set(range(17))
```

- [ ] **Step 2: Verify tests fail against the hard-coded forward body**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_pif.py -q`

Expected: FAIL because explicit PIF modes and index helpers do not exist.

- [ ] **Step 3: Extract the author path without changing `full` numerics**

Move the active semantic top-k, ordering, Mamba scan, and reconstruction into `PoseInteraction`; transcribe the repository's commented ablations into the three named modes. Add `pif_mode='full'` to `TokenPose_TB_base` and keep the full-mode tensor order byte-for-byte equivalent on fixed inputs.

- [ ] **Step 4: Run unit and regression tests**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_pif.py -q`

Expected: all modes PASS for 14/17 keypoints, full-mode output matches a captured author-path reference, and gradients are finite.

- [ ] **Step 5: Commit**

Run: `git add mmpose/models/heads/heatmap_heads/pif.py mmpose/models/heads/heatmap_heads/tokenbase.py tests/test_reproduction/test_pif.py && git commit -m "feat: expose paper PIF ablation modes"`

### Task 2: Resolved Paper Config Matrix

**Files:**
- Create: `configs/reproduction/mambapose_common.py`
- Create: `configs/reproduction/coco_s_v1.py`
- Create: `configs/reproduction/coco_s_v2.py`
- Create: `configs/reproduction/coco_b.py`
- Create: `configs/reproduction/crowdpose_s_v1.py`
- Create: `configs/reproduction/crowdpose_s_v2.py`
- Create: `configs/reproduction/coco_testdev_s_v1.py`
- Create: `configs/reproduction/coco_testdev_s_v2.py`
- Create: `configs/reproduction/ablations/*.py`
- Test: `tests/test_reproduction/test_configs.py`

**Interfaces:**
- Consumes: existing author configs, paper Table II/III/IV/V/VI, `pif_mode`.
- Produces: eleven loadable configs with stable `experiment_id`, `paper_target`, seed, full paths, and epoch-one checkpointing.

- [ ] **Step 1: Write exact matrix tests**

```python
@pytest.mark.parametrize(('name', 'depths', 'batch'), [
    ('coco_s_v1.py', [1, 1, 2, 1], 128),
    ('coco_s_v2.py', [1, 2, 3, 2], 128),
    ('coco_b.py', [2, 2, 5, 2], 64),
    ('crowdpose_s_v1.py', [1, 1, 2, 1], 64),
    ('crowdpose_s_v2.py', [1, 2, 3, 2], 64),
])
def test_primary_matrix(name, depths, batch):
    cfg = Config.fromfile(CONFIG_DIR / name)
    assert cfg.model.backbone.depths == depths
    assert cfg.train_dataloader.batch_size == batch
    assert cfg.randomness.seed == 0
    assert cfg.default_hooks.checkpoint.interval == 1
```

- [ ] **Step 2: Confirm author-config conflicts**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_configs.py -q`

Expected: FAIL because the resolved configs do not exist; audit also shows CrowdPose S-V2 `[1,2,3,1]` and wrong data root.

- [ ] **Step 3: Create explicit resolved configs**

Inherit from author configs, override only paper conflicts and operational fields, use `pretrained/vssm_tiny_0230_ckpt_epoch_262.pth`, set seed 0, epoch-one checkpoints, dataset roots, and paper targets. Test-dev configs replace the evaluator with `dict(type='CocoMetric', format_only=True, outfile_prefix='...')` and use test-dev image info/detection paths.

- [ ] **Step 4: Load and inspect every config**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_configs.py -q`

Run: `for f in configs/reproduction/*.py configs/reproduction/ablations/*.py; do PYTHONNOUSERSITE=1 .venv/bin/python tools/misc/print_config.py "$f" >/dev/null || exit 1; done`

Expected: PASS; no unresolved `crowdpose/` roots, no val evaluator in test-dev, and every mode/target is explicit.

- [ ] **Step 5: Commit**

Run: `git add configs/reproduction tests/test_reproduction/test_configs.py && git commit -m "config: resolve MambaPose paper matrix"`

### Task 3: Immutable Manifest and Atomic State

**Files:**
- Create: `mambapose_repro/__init__.py`
- Create: `mambapose_repro/manifest.py`
- Create: `mambapose_repro/state.py`
- Create: `reproduction/manifest.json`
- Test: `tests/test_reproduction/test_manifest_state.py`

**Interfaces:**
- Consumes: experiment config paths, provenance hashes, declared artifact validators.
- Produces: `load_manifest(path) -> Manifest`, `StateStore.transition(...)`, atomic `state.json`, append-only `events.jsonl` and `results.jsonl`.

- [ ] **Step 1: Write schema and crash-safety tests**

```python
def test_manifest_has_unique_stable_experiment_ids():
    manifest = load_manifest(Path('reproduction/manifest.json'))
    ids = [run.id for run in manifest.runs]
    assert len(ids) == len(set(ids)) == 11

def test_atomic_state_never_exposes_partial_json(tmp_path):
    store = StateStore(tmp_path)
    store.transition('coco-s-v1', 'running', attempt=1)
    assert json.loads((tmp_path / 'state.json').read_text())['stage'] == 'running'
```

- [ ] **Step 2: Verify missing package/schema failures**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_manifest_state.py -q`

Expected: FAIL because manifest/state APIs do not exist.

- [ ] **Step 3: Implement strict dataclasses and atomic persistence**

Reject unknown fields, duplicate IDs, undeclared artifacts, non-relative work directories, and secrets. Atomic writes use a same-directory temporary file, flush, `fsync`, `os.replace`, and directory `fsync`; events/results are JSONL with UTC timestamps and boot ID. Only the orchestrator owns transitions.

- [ ] **Step 4: Run manifest/state tests including injected write failure**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_manifest_state.py -q`

Expected: PASS; after an injected pre-replace exception, prior `state.json` remains valid.

- [ ] **Step 5: Commit**

Run: `git add mambapose_repro reproduction/manifest.json tests/test_reproduction/test_manifest_state.py && git commit -m "feat: add durable reproduction state"`

### Task 4: Resumable Data, Dataset, and Weight Preflight

**Files:**
- Create: `reproduction/sources.json`
- Create: `mambapose_repro/download.py`
- Create: `mambapose_repro/preflight.py`
- Create: `tools/reproduction/prepare_data.py`
- Test: `tests/test_reproduction/test_download.py`
- Test: `tests/test_reproduction/test_preflight.py`

**Interfaces:**
- Consumes: source records with URL/class/hash/size/license/auth metadata and target paths.
- Produces: resumable verified files, `data/inventory.json`, `pretrained/inventory.json`, and `PreflightReport`.

- [ ] **Step 1: Write HTTP-resume and schema failure tests**

```python
def test_resume_appends_only_when_server_honors_range(http_server, tmp_path):
    result = download_verified(http_server.url, tmp_path / 'x.zip',
                               expected_sha256=http_server.sha256)
    assert result.resumed and result.sha256 == http_server.sha256

def test_annotation_preflight_rejects_missing_images(coco_fixture):
    (coco_fixture / 'val2017/000000000001.jpg').unlink()
    report = preflight_coco(coco_fixture)
    assert report.status == 'permanent_failure'
```

- [ ] **Step 2: Confirm downloader/preflight APIs are absent**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_download.py tests/test_reproduction/test_preflight.py -q`

Expected: FAIL because the APIs and source manifest do not exist.

- [ ] **Step 3: Implement safe acquisition and validation**

Use `.part`, HTTP Range verification, bounded jittered retries, archive path traversal protection, archive test, and atomic rename. The source manifest explicitly names COCO `http://images.cocodataset.org/zips/{train2017,val2017,test2017}.zip`, `http://images.cocodataset.org/annotations/{annotations_trainval2017,image_info_test2017}.zip`; OpenMMLab `https://download.openmmlab.com/mmpose/datasets/crowdpose_annotations.tar`; VMamba `https://github.com/MzeroMiko/VMamba/releases/download/%2320240316/vssm_tiny_0230_ckpt_epoch_262.pth`; the MMPose-documented HRNet detection-result folder; and the official CrowdPose repository/distribution page. Classify 401/403/404, interactive quota/auth, repeated hash mismatch, and schema mismatch as permanent/manual states. Preflight COCO/CrowdPose image counts, referenced image IDs/files, keypoint lengths, detector box schema, VMamba tensor match inventory, and exact required paths.

- [ ] **Step 4: Run fixtures and dry-run source resolution**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_download.py tests/test_reproduction/test_preflight.py -q`

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/prepare_data.py --dry-run`

Expected: fixture tests PASS; dry run lists official/derived/manual-review source classes without prompting or deleting archives.

- [ ] **Step 5: Commit**

Run: `git add reproduction/sources.json mambapose_repro/download.py mambapose_repro/preflight.py tools/reproduction/prepare_data.py tests/test_reproduction/test_download.py tests/test_reproduction/test_preflight.py && git commit -m "feat: verify MambaPose reproduction data"`

### Task 5: Checkpoint Validation and Retry-Safe Orchestrator

**Files:**
- Create: `mambapose_repro/checkpoint.py`
- Create: `mambapose_repro/orchestrator.py`
- Create: `tools/reproduction/run_campaign.py`
- Test: `tests/test_reproduction/test_checkpoint.py`
- Test: `tests/test_reproduction/test_orchestrator.py`

**Interfaces:**
- Consumes: Manifest, StateStore, config/data/env/repo hashes, subprocess exit/progress.
- Produces: one-GPU-at-a-time stage execution, validated resume selection, exit 75/78 semantics, bounded attempts/fingerprints.

- [ ] **Step 1: Write corruption, provenance, and retry tests**

```python
def test_corrupt_latest_falls_back_to_valid_predecessor(checkpoint_dir):
    (checkpoint_dir / 'epoch_2.pth').write_bytes(b'corrupt')
    assert select_resume(checkpoint_dir).name == 'epoch_1.pth'

def test_changed_config_forbids_resume(valid_checkpoint, provenance):
    provenance['config_sha256'] = 'different'
    with pytest.raises(PermanentFailure, match='provenance'):
        validate_resume(valid_checkpoint, provenance)

def test_identical_transient_failure_exhausts_budget(fake_runner):
    fake_runner.always_exit(75, fingerprint='network-timeout')
    assert run_until_terminal(max_attempts=3).attempt == 3
```

- [ ] **Step 2: Verify absent recovery semantics**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_checkpoint.py tests/test_reproduction/test_orchestrator.py -q`

Expected: FAIL because checkpoint/orchestrator APIs do not exist.

- [ ] **Step 3: Implement strict recovery controller**

Acquire one campaign `flock`; select only a checkpoint that `torch.load`s and matches repo/config/data/env provenance. Record attempt/fingerprint/next retry persistently, map transient to 75 and permanent to 78, cap retries, watch structured progress/checkpoint age, terminate then kill a confirmed hung child, and never run evaluation concurrently with training.

- [ ] **Step 4: Run deterministic interruption/recovery tests**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_checkpoint.py tests/test_reproduction/test_orchestrator.py -q`

Expected: PASS for interruption, corrupt latest fallback, incompatible provenance stop, finite retries, stall escalation, and exclusive-GPU ordering.

- [ ] **Step 5: Commit**

Run: `git add mambapose_repro/checkpoint.py mambapose_repro/orchestrator.py tools/reproduction/run_campaign.py tests/test_reproduction/test_checkpoint.py tests/test_reproduction/test_orchestrator.py && git commit -m "feat: orchestrate recoverable paper runs"`

### Task 6: Non-Controlling Observer and User Service

**Files:**
- Create: `mambapose_repro/observe.py`
- Create: `tools/reproduction/observe.py`
- Create: `systemd/mambapose-reproduction.service`
- Create: `systemd/mambapose-observer.service`
- Create: `systemd/mambapose-observer.timer`
- Create: `tools/reproduction/install_user_service.sh`
- Test: `tests/test_reproduction/test_observer.py`
- Test: `tests/test_reproduction/test_systemd_units.py`

**Interfaces:**
- Consumes: state/events/logs/checkpoints, process metadata, `nvidia-smi`, disk stats.
- Produces: atomic `status.json`, health JSONL, enabled user service/timer.

- [ ] **Step 1: Write observer-authority and unit tests**

```python
def test_observer_cannot_change_campaign_state(tmp_campaign):
    before = (tmp_campaign / 'state.json').read_bytes()
    observe(tmp_campaign)
    assert (tmp_campaign / 'state.json').read_bytes() == before

def test_permanent_exit_is_not_restarted():
    unit = Path('systemd/mambapose-reproduction.service').read_text()
    assert 'RestartPreventExitStatus=78' in unit
```

- [ ] **Step 2: Confirm units/observer are absent**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_observer.py tests/test_reproduction/test_systemd_units.py -q`

Expected: FAIL because observer and units do not exist.

- [ ] **Step 3: Implement observation and installation**

Observer derives health from state plus dynamic progress/checkpoint deadlines and writes only status/history. Service uses absolute repo paths, `PYTHONNOUSERSITE=1`, `Restart=on-failure`, `RestartPreventExitStatus=78`, `KillMode=control-group`, and no shell interpolation. Installer verifies user manager, records `Linger`, installs under `~/.config/systemd/user`, daemon-reloads, enables service/timer, and reports full-logout durability as blocked when linger is off.

- [ ] **Step 4: Verify units and disconnect behavior with a fixture campaign**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_observer.py tests/test_reproduction/test_systemd_units.py -q`

Run: `bash tools/reproduction/install_user_service.sh --fixture-smoke`

Expected: PASS; a detached fixture continues after the launching shell exits, observer becomes fresh, permanent exit does not loop, and `Linger=no` is recorded without a false reboot claim.

- [ ] **Step 5: Commit**

Run: `git add mambapose_repro/observe.py tools/reproduction/observe.py systemd tools/reproduction/install_user_service.sh tests/test_reproduction/test_observer.py tests/test_reproduction/test_systemd_units.py && git commit -m "ops: persist and observe MambaPose campaign"`

### Task 7: Test-Dev and Result Artifact Validators

**Files:**
- Create: `mambapose_repro/results.py`
- Create: `tools/reproduction/validate_results.py`
- Test: `tests/test_reproduction/test_results.py`

**Interfaces:**
- Consumes: MMPose metric JSON/logs, COCO test-dev predictions, paper target matrix.
- Produces: validated normalized `results.json`, deltas, and submission-validation evidence.

- [ ] **Step 1: Write metric and submission schema tests**

```python
def test_testdev_predictions_have_coco_keypoint_shape(testdev_json):
    report = validate_testdev(testdev_json, expected_image_ids={1, 2})
    assert report.valid
    assert all(len(row['keypoints']) == 51 for row in report.rows)

def test_missing_local_ap_is_not_fabricated(testdev_json):
    normalized = normalize_testdev(testdev_json)
    assert 'AP' not in normalized
    assert normalized['evaluation'] == 'submission_only'
```

- [ ] **Step 2: Confirm result validators are absent**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_results.py -q`

Expected: FAIL because result normalization/validation APIs do not exist.

- [ ] **Step 3: Implement strict artifact validators**

Require finite numbers, known metric keys, paper-target linkage, checkpoint/config hashes, correct keypoint cardinality, valid image IDs, and deterministic normalized JSON. Mark local reproduction only when an actual evaluation artifact exists; mark test-dev `submission_only` until user supplies CodaLab output.

- [ ] **Step 4: Run validation tests**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction/test_results.py -q`

Expected: PASS for valid fixtures and fail-closed behavior for NaN, unknown IDs, wrong keypoint length, missing hashes, or fabricated AP.

- [ ] **Step 5: Commit**

Run: `git add mambapose_repro/results.py tools/reproduction/validate_results.py tests/test_reproduction/test_results.py && git commit -m "feat: validate paper result artifacts"`
