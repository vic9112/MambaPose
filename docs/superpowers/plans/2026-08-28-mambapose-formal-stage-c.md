# MambaPose Formal Stage C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and execute a fail-closed formal Stage C comparison of full S-V1 against structurally pruned no-PIF S-V1, using matched 300-epoch seeds and an optional no-PIF+W8 derivative without exceeding the approved accuracy gate.

**Architecture:** A new strict formal schema separates immutable initialization authority from produced checkpoints. A deterministic trainer records loader order and bounded resume lineage; a paired controller serializes all CUDA work through the canonical shared lock. A bounded seed-0 no-PIF+W8 screen runs before six primary trainings. Formal comparison owns paired confidence intervals and fixed seed escalation, while an independent audit produces the only hardware handoff.

**Tech Stack:** Python 3.11, PyTorch 2.7.1/CUDA 12.8, MMEngine 0.10.7, MMPose 1.3.1, pytest, JSON/Markdown evidence, user systemd, NVIDIA RTX 5090.

**Spec:** `docs/superpowers/specs/2026-08-27-mambapose-hardware-friendly-model-optimization-design.md`

## Global Constraints

- Implement only in `/home/vicchen/workspace/MambaPose/.worktrees/algo-accuracy-first` on branch `algo/accuracy-first`. Do not edit the structural or numeric worktrees.
- The approved structural reference is exact commit `a6adf6d84f9b51b133b0fca7272c51f728bfb7ac`; the approved numeric W8 reference is exact frozen W8 commit `5794d0abdea64757b95b3eaf6f5a925cbc65ac3a`. Transplant only reviewed no-PIF export and W8 weight-only behavior. Do not import W8A8, PWL, Binary Q/K, observer, or calibration behavior.
- The common initialization is `pretrained/vssm_tiny_0230_ckpt_epoch_262.pth`, SHA-256 `09739f6d95638e5caf0d33fcbca85b7cff62b8ca16ec2926d781109939c6b201`. It is a backbone initialization, never a trained pose checkpoint.
- Tracked source, configs, manifests, and produced outputs must remain inside the implementation or frozen runtime worktree. Read-only runtime assets are a separate authority: only lexical canonical relative paths below `pretrained`, `data`, and `work_dirs/reproduction` may traverse the exact declared links into the canonical main-checkout asset roots. Bind the link target root, asset-relative path, and file SHA; reject arbitrary external links, traversal, alternate same-byte trees, link-target drift, primary-link drift, and hash drift. A relocated frozen worktree must reconstruct and revalidate these same declared asset bindings.
- The bounded W8 screen receives its historical input only through a read-only repository-relative `prior-stage-b` link to the exact detached structural artifact bundle. The manifest binds structural commit `a6adf6d...`, train/profile/evaluation/latency artifacts, pruned runtime checkpoint `5797ce...`, the unpruned `28cd024...` parent lineage, and independent audit SHA `b7382d...`; runtime validation rehashes the bundle and rejects an alternate same-byte location or link drift. The fresh formal campaign never links the whole historical `work_dirs/optimization` tree.
- The primary formal arms are `baseline` and `no_pif`, with fixed seeds `0, 1, 2`; both train for 300 epochs from the same initialization authority. Seeds `3, 4` are predeclared and dormant unless the three-seed paired 95% confidence interval intersects `0.1` AP.
- Run one bounded seed-0 no-PIF+W8 conversion/export/full-val screen before any of the six primary trainings. A failed W8 screen disables only the derivative and does not remove the mandatory no-PIF formal arm.
- All CUDA-consuming work uses the existing canonical main-checkout lock at `/home/vicchen/workspace/MambaPose/work_dirs/optimization/gpu.lock`. No route-specific or frozen-worktree-local GPU lock is permitted.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=0`, and `MAMBAPOSE_PHYSICAL_DEVICE_INDEX=0` must be set by a standard-library-only launcher before importing Torch, MMEngine, MMPose, or any module that imports them.
- All datasets, detections, initializations, prior artifacts, and environment packages are local and authenticated before launch. The formal service runs with network access denied; no training/evaluation stage may download or mutate a dependency.
- Never set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. Checkpoint admission must use source-attested, hash-bound loading; runtime artifacts reject unsafe pickle globals and never deserialize an untrusted path.
- Determinism requires legal integer root seeds; Python, NumPy, Torch, and CUDA seeding; deterministic algorithms; deterministic cuDNN; fixed workers; `persistent_workers=False`; independently seeded workers; a seeded sampler; and 300 per-epoch sample-order hashes.
- Formal evaluation uses complete COCO val2017, the paper-comparable person detections, both `flip_test=True` and `flip_test=False`, and the unchanged evaluator. GPU fake-QDQ latency is software evidence only, not FPGA or integer-kernel evidence.
- Every implementation task is TDD: add failing tests, demonstrate RED, implement only enough for GREEN, run focused and adjacent CPU-safe tests, commit, then obtain independent review. A task advances only with `0 Critical / 0 Important` findings.
- CPU-safe commands use `PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B -m pytest -p no:cacheprovider`. Never use `pytest -m not gpu`, because unmarked tests may still initialize CUDA.
- No GPU campaign, systemd start, checkpoint deserialization, or long training may begin until Tasks 1-6 are committed, the implementation worktree is clean, a fresh detached runtime root is frozen at the reviewed commit, and a launch audit reports `0 Critical / 0 Important`.

---

### Task 1: Add the Strict Paired Manifest, Config, and Init/Output Contract

**Ownership:** Primary implementation agent in `algo/accuracy-first`; an independent reviewer checks strictness and exact baseline/candidate symmetry before Task 2.

**Files:**
- Create: `mambapose_opt/formal_schema.py`
- Create: `optimization/formal_stage_c.json`
- Create: `configs/optimization/formal_stage_c/_base_/paired_300ep.py`
- Create: `configs/optimization/formal_stage_c/full_seed0.py`
- Create: `configs/optimization/formal_stage_c/full_seed1.py`
- Create: `configs/optimization/formal_stage_c/full_seed2.py`
- Create: `configs/optimization/formal_stage_c/full_seed3.py`
- Create: `configs/optimization/formal_stage_c/full_seed4.py`
- Create: `configs/optimization/formal_stage_c/no_pif_seed0.py`
- Create: `configs/optimization/formal_stage_c/no_pif_seed1.py`
- Create: `configs/optimization/formal_stage_c/no_pif_seed2.py`
- Create: `configs/optimization/formal_stage_c/no_pif_seed3.py`
- Create: `configs/optimization/formal_stage_c/no_pif_seed4.py`
- Create: `tools/optimization/prepare_formal_prior_bundle.py`
- Create: `tools/optimization/build_formal_manifest.py`
- Create: `tests/test_optimization/test_formal_schema.py`

**Interfaces:**
- `FormalManifestError(ValueError)`
- `InitializationAuthority(path: Path, sha256: str, kind: Literal['vmamba-backbone'])`
- `PriorArtifactAuthority(link_root: Path, source_commit: str, artifacts: Mapping[str, FileAuthority], audit_sha256: str)`
- `FormalProtocol(epochs: int, worker_count: int, persistent_workers: bool, per_device_batch_size: int, world_size: int, accumulation_steps: int, effective_batch_size: int, primary_seeds: tuple[int, ...], conditional_seeds: tuple[int, ...])`
- `FormalRunSpec(run_id: str, role: Literal['baseline', 'no_pif'], seed: int, conditional: bool, config: Path, config_sha256: str, initialization_id: str, output_root: Path)`
- `FormalStageCManifest.from_dict(value: Mapping[str, Any], *, repository_root: Path) -> FormalStageCManifest`
- `load_formal_manifest(path: Path | str, repository_root: Path) -> FormalStageCManifest`
- `FormalRunInit.from_dict(...)` and `FormalTrainResult.from_dict(...)`, with an explicit trusted root for every path-bearing parser, exact input/output lineage, and no mixed initialization/checkpoint fields.

- [ ] **Step 1: Write strict failing schema tests**

Add tests proving that the parser rejects unknown or missing keys, duplicate run IDs, duplicate `(role, seed)` rows, duplicate output roots/configs, absolute/traversing paths, non-hex hashes, booleans as seeds, duplicate values inside either protocol seed list, any primary set other than `(0, 1, 2)`, any conditional set other than `(3, 4)`, epochs other than 300, persistent workers, baseline/no-PIF config asymmetry outside the declared PIF mode and run identity, and an initialization path placed in an output field. Each seed 0-4 must appear exactly once in each role; the required baseline/no-PIF pairing is not a duplicate-seed error. Effective batch must be derived as `per_device_batch_size * world_size * accumulation_steps` and is fixed to `128 * 1 * 1 = 128`; reject accumulation, world-size, auto-scale-LR, or pair asymmetry. Path tests must also admit the exact declared `pretrained`, `data`, and `work_dirs/reproduction` links into the canonical main-checkout asset roots, while rejecting an alternate same-byte tree, changed link targets, primary-link drift, hash drift, traversal, and every unapproved external root. The `prior-stage-b` fixture must bind the exact structural commit, train/profile/evaluation/latency artifacts, pruned and unpruned checkpoint roles, and audit SHA; reject a missing artifact, alternate same-byte bundle, swapped checkpoint roles, or changed link target. Builder tests must prove that the tracked manifest is stable across the commit that adds it: it contains closure content hashes and asset authorities but never its own containing Git commit, so `--check` has no self-referential commit/SHA cycle.

```python
def test_manifest_requires_exact_paired_seed_matrix(valid_document, repo):
    document = copy.deepcopy(valid_document)
    document['runs'].pop()
    with pytest.raises(FormalManifestError, match='paired seed matrix'):
        FormalStageCManifest.from_dict(document, repository_root=repo)

def test_train_result_cannot_rebind_initialization(valid_result):
    document = copy.deepcopy(valid_result)
    document['initialization']['sha256'] = '0' * 64
    with pytest.raises(FormalManifestError, match='initialization authority'):
        FormalTrainResult.from_dict(document)
```

- [ ] **Step 2: Run the schema test and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B -m pytest -p no:cacheprovider tests/test_optimization/test_formal_schema.py -q
```

Expected: collection fails because `mambapose_opt.formal_schema` does not exist.

- [ ] **Step 3: Implement exact schemas and safe paths**

Use frozen dataclasses and explicit key sets. Lexically normalize every relative path before access. Tracked source/config/manifest paths and produced output roots must resolve under the active worktree. Asset paths use a separate explicit authority that accepts only the declared `pretrained`, `data`, and `work_dirs/reproduction` link roots, verifies that their resolved targets are the exact canonical main-checkout asset roots, and binds target root, asset-relative path, and SHA. `prior-stage-b` is a fourth, purpose-specific read-only authority whose exact detached source worktree, structural commit, artifact set, checkpoint roles, and audit SHA are fixed independently; it is never a general external-root escape. Never authorize an arbitrary external symlink or an alternate same-byte tree. Require the exact initialization SHA above. Require ten run rows: baseline/no-PIF for seeds 0-4, with only seeds 3-4 marked conditional. The tracked manifest records expected output roots, never pre-invents checkpoint hashes. The public validator must reconstruct these bindings after relocation and fail closed on link or target drift.

`FormalRunInit` must bind manifest SHA, source commit, clean-tree assertion, config closure SHA, resolved-config SHA, environment inventory SHA, data/detection hashes, role, seed, initialization path/hash, 300 epochs, effective batch, worker policy, and output root. `FormalTrainResult` must bind its init document SHA plus best checkpoint and exactly two latest valid resume checkpoints, structured log SHA, 300 order hashes, final epoch, and completion status.

- [ ] **Step 4: Write paired configs and the manifest builder**

The common base inherits `configs/reproduction/coco_s_v1.py`, fixes `deterministic=True`, worker count `2`, `persistent_workers=False`, the seeded sampler/worker-init contracts, 300 epochs, per-device batch `128`, world size `1`, accumulation `1`, derived effective batch `128`, no auto-scale LR, evaluator, detections, and TTA. Each leaf overrides only seed, experiment/run ID, work directory, and `pif_mode` (`full` or `disabled`). Seeds 3-4 exist in Git but remain conditional in the manifest.

First, a standard-library-only bundle producer copies the exact approved structural train/runtime checkpoint/profile/evaluation/latency files, source config/manifest, and `task-4-final-artifact-audit.md` into `/home/vicchen/workspace/MambaPose/work_dirs/optimization/formal-stage-c-prior/no-pif-seed0`. It writes a canonical `bundle.json` with every source path/hash/role and source commit through atomic staging-to-rename, refuses an existing mismatch, never deserializes the checkpoint, and makes the completed files read-only. The active worktree exposes only `work_dirs/optimization/prior-stage-b` as an ignored read-only link to that bundle; it never links the whole historical optimization tree.

The builder hashes the entire config inheritance closure, dataset/annotation/detection files, initialization, and the exact files exposed through `prior-stage-b`. The tracked manifest deliberately does not contain its own source commit: doing so would become stale at the commit that adds the manifest. It may build deterministically from the current tracked/staged file contents while the Task-1 tree is dirty, and atomically writes canonical sorted JSON. After the implementation is reviewed and frozen, `FormalRunInit` binds the clean runtime HEAD, tracked manifest SHA, and recomputed config closure. Neither bundle producer nor manifest builder may import Torch or load a checkpoint.

- [ ] **Step 5: Run focused and adjacent GREEN tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B -m pytest -p no:cacheprovider tests/test_optimization/test_formal_schema.py tests/test_optimization/test_schema.py tests/test_reproduction/test_configs.py -q
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B tools/optimization/prepare_formal_prior_bundle.py --check work_dirs/optimization/prior-stage-b/bundle.json
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B tools/optimization/build_formal_manifest.py --check optimization/formal_stage_c.json
git diff --check
```

Expected: all tests pass; `--check` reports the tracked manifest is canonical and current without rewriting it, and a post-commit replay demonstrates that no containing-commit field can make it stale.

- [ ] **Step 6: Commit and review the contract**

```bash
git add mambapose_opt/formal_schema.py optimization/formal_stage_c.json configs/optimization/formal_stage_c tools/optimization/prepare_formal_prior_bundle.py tools/optimization/build_formal_manifest.py tests/test_optimization/test_formal_schema.py
git commit -m "feat: add formal paired experiment contract"
```

Review gate: independently verify exact schema rejection probes, ten-row seed matrix, same initialization authority, only PIF-mode/run-path differences across paired configs, no unsafe loader, and `0 Critical / 0 Important`.

---

### Task 2: Build the Deterministic 300-Epoch Trainer, Resume, and Order Hashes

**Ownership:** Primary implementation agent; independent determinism reviewer reproduces order traces in fresh processes before Task 3.

**Files:**
- Create: `mambapose_opt/formal_environment.py`
- Create: `mambapose_opt/formal_checkpoint.py`
- Create: `mambapose_opt/formal_determinism.py`
- Create: `mambapose_opt/formal_training.py`
- Create: `tools/optimization/capture_formal_environment.py`
- Create: `tools/optimization/formal_stage_c_entrypoint.py`
- Create: `tools/optimization/trace_formal_order.py`
- Create: `tools/optimization/preflight_formal_model.py`
- Create: `tools/optimization/train_formal_candidate.py`
- Create: `tests/test_optimization/test_formal_determinism.py`
- Create: `tests/test_optimization/test_formal_environment.py`
- Create: `tests/test_optimization/test_formal_checkpoint.py`
- Create: `tests/test_optimization/test_formal_repeatability.py`
- Create: `tests/test_optimization/test_formal_training.py`

**Interfaces:**
- `EnvironmentAuthority.capture(repository_root: Path) -> EnvironmentAuthority`
- `validate_environment_authority(authority: EnvironmentAuthority, repository_root: Path) -> None`
- `load_formal_tensor_checkpoint(path: Path, authority: FileAuthority, expected_keys: Mapping[str, TensorShape]) -> Mapping[str, Tensor]`
- `configure_root_determinism(seed: int) -> RootDeterminism`
- `seed_worker(worker_id: int) -> None`
- `build_seeded_sampler(dataset: Sized, seed: int, epoch: int) -> Sampler[int]`
- `hash_sample_order(indices: Iterable[int]) -> str`
- `trace_epoch_orders(spec: FormalRunSpec, epochs: int) -> tuple[str, ...]`
- `run_formal_model_preflight(pair_seed: int, role: str) -> FormalRepeatabilityResult`
- `compare_paired_initial_state(baseline: FormalRepeatabilityResult, no_pif: FormalRepeatabilityResult) -> PairedInitializationResult`
- `validate_resume_checkpoint(path: Path, expected: FormalRunInit) -> ResumeState`
- `train_formal_candidate(init_path: Path, resume_path: Path | None) -> FormalTrainResult`
- Standard-library-only `formal_stage_c_entrypoint.py` sets process environment and then uses `os.execve` to invoke the requested worker module.

- [ ] **Step 1: Write determinism and resume tests first**

Tests must prove legal seeds include 0 and reject bool/negative/out-of-range values; root configuration sets Python/NumPy/Torch/CUDA seeds, deterministic algorithms, `cudnn.deterministic=True`, `cudnn.benchmark=False`; candidate seed is never hard-coded to zero; worker and sampler order changes by seed/epoch but repeats exactly for the same pair; two fresh subprocesses emit identical trace hashes; and no Torch-bearing module is imported before the launcher sets `CUBLAS_WORKSPACE_CONFIG`.

Environment tests bind the `.venv` link root, resolved interpreter, interpreter SHA, lock/requirements evidence, exact package inventory, and native CUDA-extension files. They reject an alternate same-byte environment, executable/package/native-module drift, traversal, a symlink swap, and any stage whose fresh observation differs from `FormalRunInit`. Safe-loader tests include a hostile-pickle fixture and require `weights_only=True` or a stricter tensor-only format, exact path/SHA/key/shape/dtype validation, recursive finite tensors, and rejection of symlink or source-authority drift. No supported profile/evaluation/training path may fall back to `Backbone_VSSM.load_pretrained` or generic unsafe `init_model` checkpoint loading.

Repeatability unit tests are mock-based but require a public artifact with role/seed/config/source/environment/custom-scan identity, complete initial-state tensor digest, common-tensor digest, forward/loss/backward/gradient/optimizer-update digests, and predeclared numeric tolerances. A baseline/no-PIF pair must have equal common-tensor digests; each role's complete digest must repeat across fresh processes. Missing custom-scan evidence, unexpected drift, a changed tolerance, non-finite gradients, or a digest from a different source fails admission.

Resume tests use synthetic, source-attested tensors only. They reject mismatched manifest/init/config/seed/role/order prefix, non-finite optimizer state, missing RNG state, unvalidated checkpoint, and a fourth retained checkpoint.

```python
def test_trace_repeats_but_seed_changes_order(tmp_path):
    a = run_trace_subprocess(seed=2, epoch=17, root=tmp_path)
    b = run_trace_subprocess(seed=2, epoch=17, root=tmp_path)
    c = run_trace_subprocess(seed=1, epoch=17, root=tmp_path)
    assert a == b
    assert a != c

def test_resume_rejects_wrong_pair_member(valid_resume, run_init):
    valid_resume['formal']['role'] = 'baseline'
    with pytest.raises(FormalTrainingError, match='role'):
        validate_resume_document(valid_resume, expected=replace(run_init, role='no_pif'))
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B -m pytest -p no:cacheprovider tests/test_optimization/test_formal_environment.py tests/test_optimization/test_formal_checkpoint.py tests/test_optimization/test_formal_determinism.py tests/test_optimization/test_formal_repeatability.py tests/test_optimization/test_formal_training.py -q
```

Expected: collection fails because the formal determinism/training modules do not exist.

- [ ] **Step 3: Implement process-first determinism**

The launcher must be import-safe with standard library only. Before every stage it reconstructs and rehashes `EnvironmentAuthority`, validates the exact approved `.venv` link/interpreter/package/native-module inventory, validates exact environment values, sets them if absent, rejects conflicting values, and then execs the authority-bound `.venv/bin/python -B -m <worker> ...`. Inside the worker, call `configure_root_determinism(spec.seed)` before config loading, model construction, or dataloader construction.

Create a 2-epoch preflight that traces the complete sampler order twice in separate processes. Admission requires byte-identical per-epoch hashes. The real trainer records all 300 hashes and checks each runtime epoch against its preflight-derived deterministic rule.

After implementation review and before any 300-epoch run, execute a real MambaPose pair preflight twice in fresh processes for each role/seed. It covers backbone custom selective scan, forward, supervised loss, backward, finite gradients, and one optimizer update. Record complete initial-state and common-state digests before the update; the baseline/no-PIF common tensors must be bit-identical for a seed, while each role's complete state must replay. Compare output/loss/gradient/update digests under predeclared tolerances from the tracked protocol. Unexpected drift stops launch; any accepted residual kernel nondeterminism is quantified in the artifact and final limitation rather than hidden. This public preflight artifact is a controller dependency, not optional commentary.

- [ ] **Step 4: Implement authenticated training and bounded resume**

Use the manifest/init schema from Task 1. Implement a formal checkpoint loader rather than calling the current `Backbone_VSSM.load_pretrained`, which uses plain `torch.load`. Load the exact VMamba initialization with `weights_only=True` (or a stricter tensor-only representation), verify path/SHA/key/shape/dtype/finite tensors, construct with implicit `pretrained`/`init_cfg` neutralized, and inject only validated compatible backbone tensors. The same safe interface loads produced pose tensors for profile/evaluation; generic checkpoint initialization is not an allowed fallback. Never enable broad unsafe deserialization. Save optimizer, scheduler, scaler if used, Python/NumPy/Torch/CUDA RNG states, completed epoch, order-hash prefix, init-document SHA, and config/source identities.

Write checkpoints atomically. Retain best plus the two newest validated resume checkpoints. A resume may continue only the same run ID at the next epoch. On completion, emit one strict `FormalTrainResult`; downstream stages consume its best-checkpoint path/hash from this result rather than a manifest placeholder.

Add a CPU synthetic 2-epoch smoke mode for tests; production configs remain exactly 300 epochs and cannot use smoke flags.

- [ ] **Step 5: Run fresh-process reproduction and adjacent tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B -m pytest -p no:cacheprovider tests/test_optimization/test_formal_environment.py tests/test_optimization/test_formal_checkpoint.py tests/test_optimization/test_formal_determinism.py tests/test_optimization/test_formal_repeatability.py tests/test_optimization/test_formal_training.py tests/test_optimization/test_determinism.py tests/test_optimization/test_source.py -q
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -B tools/optimization/formal_stage_c_entrypoint.py trace --manifest optimization/formal_stage_c.json --run-id full-seed0 --epochs 2 --cpu-only --repeat 2
git diff --check
```

Expected: tests pass; the two fresh trace documents have identical order hashes and distinct process IDs; no CUDA is initialized.

- [ ] **Step 6: Commit and determinism review**

```bash
git add mambapose_opt/formal_environment.py mambapose_opt/formal_checkpoint.py mambapose_opt/formal_determinism.py mambapose_opt/formal_training.py tools/optimization/capture_formal_environment.py tools/optimization/formal_stage_c_entrypoint.py tools/optimization/trace_formal_order.py tools/optimization/preflight_formal_model.py tools/optimization/train_formal_candidate.py tests/test_optimization/test_formal_environment.py tests/test_optimization/test_formal_checkpoint.py tests/test_optimization/test_formal_determinism.py tests/test_optimization/test_formal_repeatability.py tests/test_optimization/test_formal_training.py
git commit -m "feat: add deterministic formal training producer"
```

Review gate: independently inspect environment authority and per-stage drift rejection, import order, safe tensor-only checkpoint admission with no generic-loader fallback, seed propagation, 300-hash contract, paired common-state identity, custom-scan/full-model repeatability evidence and tolerance policy, best-plus-two retention, and two-process replay. Require `0 Critical / 0 Important`.

---

### Task 3: Add the Paired Controller and Durable User-systemd Units

**Ownership:** Primary implementation agent owns controller/state schema; an independent launch reviewer owns unit verification and CUDA-lock admission review.

**Files:**
- Create: `mambapose_opt/formal_controller.py`
- Create: `mambapose_opt/formal_observe.py`
- Create: `tools/optimization/run_formal_stage_c.py`
- Create: `tools/optimization/observe_formal_stage_c.py`
- Create: `tools/optimization/install_formal_stage_c_service.sh`
- Create: `systemd/mambapose-formal-stage-c.service`
- Create: `systemd/mambapose-formal-stage-c-observer.service`
- Create: `systemd/mambapose-formal-stage-c-observer.timer`
- Create: `tests/test_optimization/test_formal_controller.py`
- Create: `tests/test_optimization/test_formal_systemd.py`

**Interfaces:**
- `FormalStage(name, run_id, attempt, cuda, conditional, dependencies)`
- `FormalCampaignState.from_dict(...) -> FormalCampaignState`
- `build_formal_dag(manifest) -> FormalDAG`, which always contains primary, W8, escalation, and final branches and has one immutable SHA
- `CampaignControllerLock.acquire_nonblocking(path: Path) -> ContextManager[CampaignControllerLock]`
- `FormalStageCController.run_next() -> StageOutcome`
- `FormalStageCController.revalidate_running_stage() -> LeaseDisposition`
- `observe_formal_campaign(root: Path) -> Mapping[str, Any]`

- [ ] **Step 1: Write controller/state/unit RED tests**

Tests assert one immutable DAG/SHA predeclares bounded W8, exactly six primary baseline/no-PIF seed0-2 trainings, both indivisible seed3/4 pairs, W8 derivatives, comparison, and final audit; runtime results change activation/disposition state, never the plan. Task 3 does not fabricate the Task-4 screen schemas or producers: until Task 4 registers those exact validators/runners, attempting the screen returns permanent protocol failure and cannot activate W8. Dormant seed3/4 stages cannot run without a validated escalation admission. One non-blocking canonical controller lock prevents CLI/service or two services from mutating the campaign concurrently. The observer is read-only. State/event writes are atomic and append-only. Exit 75 is transient contention; exit 78 is permanent protocol failure; retry counters and deadlines persist across systemd restarts. Every stage revalidates the Task-2 `EnvironmentAuthority`; drift stops before process launch.

Contention tests introduce a foreign GPU owner after admission. The controller must revalidate the lease on every heartbeat, signal the worker to stop cooperatively, accept only an atomic validated resume checkpoint at a safe boundary, record the pre-contention lineage, exit transiently, and later resume from that checkpoint. A worker that cannot produce a valid resume stops permanently rather than continuing contaminated training. Any latency interval overlapping contention is invalid. Tests also cover controller-lock overlap, restart-persistent retry counters, immutable-DAG mutation, and activation of only predeclared branches.

Unit tests parse files and require exact frozen-root substitution, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, no `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`, network denial after preflight, `Restart=on-failure`, `RestartPreventExitStatus=78`, `KillMode=control-group`, bounded `TimeoutStopSec`, `UMask=0077`, timer `Persistent=true`, and the canonical shared GPU-lock and controller-lock arguments.

```python
def test_primary_train_dag_is_exactly_six_and_immutable(manifest):
    dag = build_formal_dag(manifest)
    train = [(s.run_id, s.name) for s in dag.stages if s.name == 'train' and not s.conditional]
    assert train == [
        ('full-seed0', 'train'), ('no-pif-seed0', 'train'),
        ('full-seed1', 'train'), ('no-pif-seed1', 'train'),
        ('full-seed2', 'train'), ('no-pif-seed2', 'train'),
    ]
```

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B -m pytest -p no:cacheprovider tests/test_optimization/test_formal_controller.py tests/test_optimization/test_formal_systemd.py -q
```

Expected: collection fails because the formal controller and units do not exist.

- [ ] **Step 3: Implement immutable orchestration and shared locking**

Keep this controller separate from the Stage A/B six-stage generic runner. Every CUDA stage acquires `/home/vicchen/workspace/MambaPose/work_dirs/optimization/gpu.lock` through `canonical_gpu_lock_path`; admission also rejects foreign live GPU owners. No stage may pass a route-local lock.

The immutable DAG order is: W8 bounded screen, its independent disposition gate, primary order/full-model repeatability preflights, six interleaved trainings, exports/profiles/evaluations, three-seed comparison, dormant seed3/4 pair trainings, final comparison, and audit. Task 3 owns immutable stage/dependency names and explicit unregistered-runner failure only; Task 4 supplies and wires the strict screen result/disposition producers and validators before launch. A validated candidate failure marks W8 `disabled` and proceeds to no-PIF formal training, while a missing runner/schema/provenance failure is permanent and cannot be reclassified as a candidate result. A primary deterministic/provenance failure stops the campaign. Hold one canonical campaign-controller lock for every state mutation. During CUDA work, revalidate the GPU lease/foreign owners on every heartbeat and cooperatively stop at an atomic validated resume point if contention appears; invalidate affected latency. Before each stage, rehash the exact `.venv`/interpreter/package/native-module authority and refuse any environment drift.

- [ ] **Step 4: Implement durable service and read-only observer**

The install script takes `--frozen-root` and refuses a branch-attached, dirty, or wrong-commit root. The service invokes only the standard-library launcher, denies network access, and uses the canonical controller lock. The observer can read service/campaign/artifact state and append no controller state; it cannot start, stop, retry, or complete a stage. Units are installed disabled; enabling/starting is a separate audited runtime action.

- [ ] **Step 5: Run focused, adjacent, and unit-verification GREEN checks**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B -m pytest -p no:cacheprovider tests/test_optimization/test_formal_controller.py tests/test_optimization/test_formal_systemd.py tests/test_optimization/test_controller.py tests/test_optimization/test_gpu_guard.py tests/test_optimization/test_observer.py tests/test_optimization/test_systemd.py -q
systemd-analyze verify systemd/mambapose-formal-stage-c.service systemd/mambapose-formal-stage-c-observer.service systemd/mambapose-formal-stage-c-observer.timer
git diff --check
```

Expected: all tests and unit verification pass without starting or enabling a unit.

- [ ] **Step 6: Commit and launch-contract review**

```bash
git add mambapose_opt/formal_controller.py mambapose_opt/formal_observe.py tools/optimization/run_formal_stage_c.py tools/optimization/observe_formal_stage_c.py tools/optimization/install_formal_stage_c_service.sh systemd/mambapose-formal-stage-c.service systemd/mambapose-formal-stage-c-observer.service systemd/mambapose-formal-stage-c-observer.timer tests/test_optimization/test_formal_controller.py tests/test_optimization/test_formal_systemd.py
git commit -m "feat: add durable formal paired controller"
```

Review gate: verify immutable full DAG, one controller and one shared GPU lock, mid-stage foreign-owner stop/resume and latency invalidation, restart-persistent retry state, dormant escalation, screen-failure continuation, service environment-before-import and network denial, disabled installation, observer non-mutation, and `0 Critical / 0 Important`.

---

### Task 4: Integrate Pruned no-PIF with Bounded W8 and Complete Export

**Ownership:** Primary integration agent. Structural and numeric reviewers independently compare transplanted semantics to commits `a6adf6d...` and `5794d0a...`; neither route branch is modified.

**Files:**
- Create: `mambapose_opt/export.py`
- Create: `mambapose_opt/formal_runtime.py`
- Create: `mambapose_opt/formal_w8.py`
- Create: `configs/optimization/formal_stage_c/no_pif_pruned.py`
- Create: `configs/optimization/formal_stage_c/no_pif_w8.py`
- Create: `tools/optimization/convert_formal_w8.py`
- Create: `tools/optimization/export_formal_w8.py`
- Create: `tools/optimization/smoke_formal_w8.py`
- Create: `tools/optimization/profile_formal_candidate.py`
- Create: `tools/optimization/evaluate_formal_candidate.py`
- Create: `tools/optimization/measure_formal_latency.py`
- Create: `tools/optimization/decide_formal_w8_screen.py`
- Create: `tests/test_optimization/test_formal_no_pif_export.py`
- Create: `tests/test_optimization/test_formal_no_pif_w8.py`
- Create: `tests/test_optimization/test_formal_w8_stage_a.py`
- Create: `tests/test_optimization/test_formal_w8_screen.py`
- Create: `mmpose/models/utils/hardware_friendly/fake_quant.py`
- Create: `mmpose/models/utils/hardware_friendly/__init__.py`
- Modify: `mmpose/models/heads/heatmap_heads/mamba_token_head.py`
- Modify: `mmpose/models/heads/heatmap_heads/tokenbase.py`
- Modify: `mambapose_opt/formal_controller.py`
- Modify: `tests/test_optimization/test_formal_controller.py`

**Interfaces:**
- `export_pruned_no_pif(model: nn.Module, destination_factory: Callable[[], nn.Module]) -> nn.Module`
- `build_no_pif_w8_policy(model: nn.Module) -> QuantPolicy`
- `convert_no_pif_w8(model: nn.Module, parent_result: FormalTrainResult) -> NoPIFW8Result`
- `export_complete_no_pif_w8(model, conversion, destination: Path) -> CompleteW8Export`
- `load_complete_no_pif_w8(export: Path, destination_factory) -> nn.Module`
- `FormalNoPIFExportResult.from_dict(...) -> FormalNoPIFExportResult`
- `FormalW8StageAResult.from_dict(...) -> FormalW8StageAResult`
- `FormalProfileResult.from_dict(...) -> FormalProfileResult`
- `FormalW8ScreenResult.from_dict(...) -> FormalW8ScreenResult`
- `FormalW8Disposition.from_dict(...) -> FormalW8Disposition`
- `run_formal_w8_stage_a(parent: PriorArtifactAuthority, policy: QuantPolicy) -> FormalW8StageAResult`
- `decide_formal_w8_screen(screen: FormalW8ScreenResult) -> FormalW8Disposition`

- [ ] **Step 1: Write structural and W8 RED tests**

Structural tests require bit-exact disabled-to-pruned heatmaps, no `PoseInteraction` child, no `pose_interaction.*` state, strict reachable-state equality, and rejection of full-mode sources. W8 tests require exact live-role discovery after pruning, zero PIF roles, per-output-channel symmetric int8 weights, `>=0.8` coverage of all live Linear/Conv MAC-bearing weights, deterministic conversion, and no activation QDQ.

Complete-export tests require every tensor to appear exactly once as packed int8 weight+FP32 scale/bias or remaining FP32 tensor; total file bytes and tensor-payload bytes are measured separately; all files and the canonical manifest have SHA-256; reconstruction is numerically identical to the fake-W8 runtime; relative paths only; parent train result/config/policy/source hashes must match.

Stage-A tests use a mockable producer plus a real-model contract: exact authenticated pruned parent and W8 policy/source/environment, `[1,3,256,192] -> [1,17,64,48]`, finite forward/loss/backward, finite nonzero gradients for every W8 target, Adam parameter/state update, identity-mode equivalence, tensor-only serialization/resume, and clean export/reload. Tampering with parent/policy/role/hash/path/tensor/shape/gradient/optimizer/export must fail closed. Screen tests require strict schemas and public validators for profile/range/operation inventory, dual-mode full-val evaluation, latency, reconstruction, disposition, and the transitive SHA of the exact Stage-A result. They verify 5,000-image COCO/detection authority and controller activation only from an admitted disposition. Profile tests require an authenticated fixed train2017-only sample list/order/count, tensor/operator roles, shapes/dtypes, min/max and declared range statistics, precision/accumulator semantics, unsupported-operator inventory, and claim limits; compound source/order/range/runtime tampering fails closed.

```python
def test_no_pif_w8_policy_contains_no_removed_roles(pruned_model):
    policy = build_no_pif_w8_policy(pruned_model)
    assert policy.allow
    assert all('pose_interaction' not in role for role in policy.allow)

def test_complete_export_accounts_for_every_state_tensor(converted, tmp_path):
    report = export_complete_no_pif_w8(converted.model, converted.report, tmp_path)
    assert report.accounted_state_keys == tuple(converted.model.state_dict())
    assert report.low_bit_coverage >= 0.8
    assert report.integer_kernel_latency_claimed is False
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B -m pytest -p no:cacheprovider tests/test_optimization/test_formal_no_pif_export.py tests/test_optimization/test_formal_no_pif_w8.py tests/test_optimization/test_formal_w8_stage_a.py tests/test_optimization/test_formal_w8_screen.py -q
```

Expected: collection fails because export/W8 interfaces do not exist.

- [ ] **Step 3: Transplant only approved structural and W8 semantics**

Inspect exact provenance before coding:

```bash
git show a6adf6d84f9b51b133b0fca7272c51f728bfb7ac:mambapose_opt/export.py
git show a6adf6d84f9b51b133b0fca7272c51f728bfb7ac:tests/test_optimization/test_no_pif_export.py
git show 5794d0abdea64757b95b3eaf6f5a925cbc65ac3a:mmpose/models/utils/hardware_friendly/fake_quant.py
```

Port the audited `pif_export_pruned` construction and strict reachable-state mapping without importing static-neighbor experiments. Port only `QuantSpec`, `QuantPolicy`, fake-W8 Linear/Conv wrappers, deterministic conversion, and int8 packing. The package `__init__.py` must not import observer, activation calibration, PWL, or Binary Q/K modules.

- [ ] **Step 4: Implement no-PIF-specific policy and complete export**

Discover exact eligible roles from the pruned runtime and bind the sorted role list plus policy SHA to the artifact. Reject missing, extra, duplicate, PIF, or unsupported roles. The denominator is all live Linear/Conv weights, not only the allow list.

Extend the approved partial packing into a complete deployment package: packed int8 values and FP32 scales for admitted weights; FP32 biases; every remaining FP32 state tensor; shapes/dtypes; config/source/parent hashes; operation/precision manifest; coverage numerator/denominator; total payload and on-disk bytes. Do not claim an integer kernel or FPGA latency.

- [ ] **Step 5: Implement and validate the integrated full-model Stage A**

Run the authenticated pruned no-PIF+W8 model through a real one-batch forward/loss/backward/Adam step under the shared launcher/lock contract. Prove every target gradient and optimizer update, identity mode, restricted tensor-only save/resume, clean export/reload, exact tensor shapes, and parent/policy/source/environment binding. `convert`, export, profile, evaluation, latency, and disposition validators must all require the exact Stage-A path/SHA; a different later valid smoke cannot retroactively validate an earlier result.

- [ ] **Step 6: Implement bounded screen producers, validators, and admission**

Bind the screen through `PriorArtifactAuthority` and the approved structural train artifact to the seed-0 pruned runtime checkpoint SHA `5797ceaffbf7d369d8eaf8f47b548a79bd43d66dd603933571fa3b88ff28e3db`. That structural artifact must also attest that its unpruned trained-parent input lineage is checkpoint SHA `28cd02405e58d619896a0430f91936684084ef7e3423d507c7a10b743759a5fb`; the unpruned parent is never the W8 conversion input. Implement explicit formal profile/range/operation-inventory, full COCO val flip/no-flip evaluation, bounded software latency, reconstruction, screen-result, and disposition producers plus public validators; do not reuse old `CandidateSpec` tools as implicit formal consumers. The profile observes a predeclared fixed authenticated train2017-only sample list (default 512, never val2017) and records exact order/count, tensor/operator roles, shapes/dtypes/ranges, precision/accumulator semantics, and unsupported operations. The same profile producer runs for every trained baseline, no-PIF, and admitted W8 derivative before evaluation, so Task 6 never infers missing ranges by deserializing a checkpoint. Bind all files transitively to the same parent, Stage-A result where applicable, reconstructed complete export, source/environment, canonical train and 5,000-image COCO/detection authorities, and runtime checkpoint. Admission requires valid artifacts, no-PIF structural proof, complete export, `>=0.8` live Linear/Conv coverage, and flip AP drop `<=0.3` relative to its approved pruned no-PIF parent. Failure writes a terminal W8-disabled disposition and leaves formal no-PIF enabled. Modify the Task-3 controller so only this public disposition activates W8 derivation; missing or invalid artifacts fail closed.

- [ ] **Step 7: Run focused and adjacent GREEN tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B -m pytest -p no:cacheprovider tests/test_optimization/test_formal_no_pif_export.py tests/test_optimization/test_formal_no_pif_w8.py tests/test_optimization/test_formal_w8_stage_a.py tests/test_optimization/test_formal_w8_screen.py tests/test_optimization/test_formal_controller.py tests/test_optimization/test_integration_policy.py tests/test_optimization/test_inventory.py -q
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B -m pytest -p no:cacheprovider tests/test_optimization/test_evaluation.py tests/test_optimization/test_latency.py -q
git diff --check
```

Expected: all tests pass; no PIF, W8A8, PWL, Binary, observer, or activation-scale role is admitted.

- [ ] **Step 8: Commit and dual-route review**

```bash
git add mambapose_opt/export.py mambapose_opt/formal_runtime.py mambapose_opt/formal_w8.py configs/optimization/formal_stage_c/no_pif_pruned.py configs/optimization/formal_stage_c/no_pif_w8.py tools/optimization/convert_formal_w8.py tools/optimization/export_formal_w8.py tools/optimization/smoke_formal_w8.py tools/optimization/profile_formal_candidate.py tools/optimization/evaluate_formal_candidate.py tools/optimization/measure_formal_latency.py tools/optimization/decide_formal_w8_screen.py tests/test_optimization/test_formal_no_pif_export.py tests/test_optimization/test_formal_no_pif_w8.py tests/test_optimization/test_formal_w8_stage_a.py tests/test_optimization/test_formal_w8_screen.py mmpose/models/utils/hardware_friendly/fake_quant.py mmpose/models/utils/hardware_friendly/__init__.py mmpose/models/heads/heatmap_heads/mamba_token_head.py mmpose/models/heads/heatmap_heads/tokenbase.py mambapose_opt/formal_controller.py tests/test_optimization/test_formal_controller.py
git commit -m "feat: integrate pruned no-pif weight-only export"
```

Review gate: structural reviewer proves exact deletion/export semantics; numeric reviewer proves W8-only semantics and complete byte accounting; shared reviewer checks screen parent binding and fail-closed disposition. Require `0 Critical / 0 Important`.

---

### Task 5: Implement Formal Paired Comparison, Confidence Interval, and Escalation

**Ownership:** Primary comparison agent; an independent statistics/provenance reviewer validates synthetic fixtures and exact artifact bindings.

**Files:**
- Create: `mambapose_opt/formal_compare.py`
- Create: `tools/optimization/compare_formal_stage_c.py`
- Create: `tests/test_optimization/test_formal_compare.py`

**Interfaces:**
- `PairedSeedResult(seed: int, baseline: MetricsAuthority, candidate: MetricsAuthority, drop: float)`
- `PairedComparison(seeds, drops, mean_drop, sample_stddev, ci95_lower, ci95_upper, max_drop, accuracy_pass, needs_escalation)`
- `compare_formal_pairs(manifest, result_paths) -> PairedComparison`
- `build_escalation_admission(comparison: PairedComparison) -> EscalationAdmission`
- `write_escalation_admission(admission: EscalationAdmission, destination: Path) -> FileAuthority`

- [ ] **Step 1: Write exact statistics and provenance RED tests**

Tests cover negative drops, strict `mean_drop < 0.1`, strict `max_drop < 0.3`, sample standard deviation, individual differences, and paired two-sided 95% Student-t confidence intervals. Use fixed critical values `4.302652729911275` for three seeds and `2.7764451051977987` for five seeds. Intersection is inclusive: `ci95_lower <= 0.1 <= ci95_upper`.

Reject unpaired/missing/duplicate seeds; baseline/candidate seed mismatch; different initialization/config protocol/data/detections/evaluator/TTA/order-policy identities; incomplete full-val authority; only one flip mode; changed source; W8 result without a parent no-PIF formal result; and any historical Stage-B/rejected W8A8 artifact.

```python
def test_three_seed_ci_intersection_admits_only_fixed_seeds_3_and_4(fixtures):
    result = compare_formal_pairs(fixtures.for_drops([0.00, 0.10, 0.20]))
    assert result.needs_escalation is True
    admission = build_escalation_admission(result)
    assert admission.seeds == (3, 4)

def test_gate_is_strict_at_boundaries(fixtures):
    assert not compare_formal_pairs(fixtures.for_drops([0.1, 0.1, 0.1])).accuracy_pass
    assert not compare_formal_pairs(fixtures.for_drops([0.3, -0.1, -0.1])).accuracy_pass
```

- [ ] **Step 2: Run comparison test and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B -m pytest -p no:cacheprovider tests/test_optimization/test_formal_compare.py -q
```

Expected: collection fails because `mambapose_opt.formal_compare` does not exist.

- [ ] **Step 3: Implement fail-closed comparison**

The primary paired point estimate is full S-V1 AP minus no-PIF AP in paper-comparable flip mode. Always emit each seed, mean, sample standard deviation, maximum, and 95% CI; also report no-flip AP plus AP50/AP75/APM/APL/AR without substituting them for the primary gate.

For three seeds, emit exactly one of: `formal_pass`, `formal_fail`, or `needs_seed3_4`. If the CI intersects 0.1, the controller validates the comparison SHA and activates both predeclared seed3/4 pairs. It may not choose one seed, retry for favorable metrics, or discard a completed seed. Recompute the same strict mean/max rules over all five seeds.

If W8 passed its bounded screen, derive and evaluate W8 from each completed no-PIF checkpoint and report a separate paired derivative row. No additional 300-epoch W8 training arm is created.

- [ ] **Step 4: Run synthetic statistics, mutation, and adjacent GREEN tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B -m pytest -p no:cacheprovider tests/test_optimization/test_formal_compare.py tests/test_optimization/test_formal_schema.py tests/test_optimization/test_formal_controller.py -q
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B -m pytest -p no:cacheprovider tests/test_optimization/test_evaluation.py tests/test_optimization/test_artifacts.py -q
git diff --check
```

Expected: all tests pass, including forged metric/config/order/source mutations and exact boundary cases.

- [ ] **Step 5: Commit and statistics review**

```bash
git add mambapose_opt/formal_compare.py tools/optimization/compare_formal_stage_c.py tests/test_optimization/test_formal_compare.py
git commit -m "feat: add formal paired accuracy gate"
```

Review gate: independently recompute fixture statistics, verify inclusive CI intersection and strict accuracy boundaries, prove seed3/4 are fixed and indivisible, and require `0 Critical / 0 Important`.

---

### Task 6: Add Final Audit, Hardware Handoff, and Frozen Runtime Gates

**Ownership:** Primary agent implements the auditor and templates. A different independent reviewer performs launch and final artifact audits; the implementation agent may not self-approve results.

**Files:**
- Create: `mambapose_opt/formal_audit.py`
- Create: `tools/optimization/audit_formal_stage_c.py`
- Create: `tests/test_optimization/test_formal_audit.py`
- Create: `docs/optimization/formal-stage-c-results.md`
- Create: `docs/optimization/formal-stage-c-hardware-handoff.md`

**Interfaces:**
- `audit_formal_campaign(root: Path, manifest: Path) -> FormalAuditReport`
- `build_hardware_handoff(audit: FormalAuditReport) -> HardwareHandoff`
- `FormalAuditReport.to_json() -> Mapping[str, Any]`

- [ ] **Step 1: Write audit/handoff RED tests**

Audit tests reject dirty/wrong source, missing stages, nonzero/restarted service result, held shared lock, incomplete/non-validated 300-epoch results, wrong checkpoint retention, missing order hashes, different paired initialization/protocol, incomplete COCO authority, missing flip mode, malformed comparison/escalation, derivative without admission, missing or drifted formal profile/range/operation authority for any evaluated runtime, and any absolute public artifact path.

Handoff tests require tensor names/shapes/dtypes/ranges, input/output shapes, precision by operator, static absence of PIF, W8 scales/accumulator declaration where applicable, remaining Transformer softmax/nonlinearities, VMamba custom scan hazards, unsupported operation inventory, model bytes, and explicit claim limits. It must reject FPGA resource, latency, throughput, power, integer-kernel, or external-paper speedup claims without measured hardware evidence.

- [ ] **Step 2: Run audit test and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B -m pytest -p no:cacheprovider tests/test_optimization/test_formal_audit.py -q
```

Expected: collection fails because `mambapose_opt.formal_audit` does not exist.

- [ ] **Step 3: Implement whole-lineage audit and relative public result**

The auditor follows hashes from formal manifest to run init, training, best/resume checkpoints, export, formal profile/range/operation artifact, dual-mode evaluation, W8 derivative, comparison, and escalation admission. It recomputes file hashes without checkpoint deserialization. It verifies the profile collection's authenticated train-only sample source/order/count, tensor/operator ranges and precision, unsupported operations, campaign completion, no live lock holder, and service exit evidence. Public CandidateResult paths are repository/campaign-relative and replay from either the original or relocated campaign root.

The final result document distinguishes `formal_pass`, `formal_fail`, and `statistically_unresolved`; a failed candidate remains a valid completed experiment. Only a formal accuracy pass plus structural PIF deletion enters the Pareto set. W8 storage is additional evidence only after complete-export validation. Fake-QDQ GPU latency remains a tie-breaker and never an FPGA prediction.

- [ ] **Step 4: Implement hardware handoff and result templates**

The Markdown templates are populated only from validated JSON. They include all individual seed results, paired CI, secondary metrics, checkpoint/export SHAs, parameters/FP32 bytes/export bytes/coverage, operation changes, latency protocol/distributions/repeat variability, remaining hazards, and claim limits. The handoff is algorithm-only and explicitly defers board resource/clock/latency trade-offs to a later FPGA phase.

- [ ] **Step 5: Run focused, adjacent, and full CPU-safe tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B -m pytest -p no:cacheprovider tests/test_optimization/test_formal_audit.py tests/test_optimization/test_formal_compare.py tests/test_optimization/test_formal_training.py tests/test_optimization/test_formal_controller.py -q
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B -m pytest -p no:cacheprovider tests/test_optimization tests/test_reproduction/test_configs.py tests/test_reproduction/test_results.py -q
git diff --check
find . -type d -name __pycache__ -not -path './.venv/*' -print
```

Expected: all CPU-safe tests pass; `git diff --check` is clean; `find` prints nothing outside `.venv`.

- [ ] **Step 6: Commit and obtain implementation closure review**

```bash
git add mambapose_opt/formal_audit.py tools/optimization/audit_formal_stage_c.py tests/test_optimization/test_formal_audit.py docs/optimization/formal-stage-c-results.md docs/optimization/formal-stage-c-hardware-handoff.md
git commit -m "feat: add formal stage c audit and handoff"
git status --short --branch
git log --oneline --decorate -6
```

Review gate: review the full Stage C implementation range, run the focused and full CPU-safe suites from a fresh checkout, execute malformed-artifact probes, and require `0 Critical / 0 Important`. Do not launch if any Important remains.

- [ ] **Step 7: Freeze and audit the runtime root before any CUDA action**

Create a new detached runtime worktree under `/home/vicchen/workspace/MambaPose/.worktrees/` at the exact approved implementation commit. It must be clean and fresh, contain no campaign state or bytecode, and use canonical read-only links for `.venv`, `data`, `pretrained`, and `work_dirs/reproduction`. Create only the ignored read-only sublink `work_dirs/optimization/prior-stage-b` to `/home/vicchen/workspace/MambaPose/work_dirs/optimization/formal-stage-c-prior/no-pif-seed0`; do not link the whole historical `work_dirs/optimization`. The remaining new campaign root is local while the GPU lock remains canonical and shared.

The independent launch audit must verify exact commit/detached/clean state, every link including the prior bundle, manifest/config/prior-artifact hashes, the captured `.venv` real interpreter/package/native-module authority, service/timer paths and environment, user lingering, old optimization units inactive/disabled, no live shared-lock holder, no external GPU owner, and no campaign state. Read-only `systemctl --user show/is-active/is-enabled` and `systemd-analyze verify` are allowed; launch audit must not start a unit, initialize CUDA, or deserialize a checkpoint.

- [ ] **Step 8: Execute the approved runtime sequence**

Only after launch approval:

1. Run the bounded seed-0 no-PIF+W8 convert/export/reconstruction/full-val screen under the shared lock.
2. Obtain independent screen artifact review. Record W8 `admitted` or `disabled`; proceed in both cases.
3. Run loader preflight repeats for all six primary runs.
4. Train exactly six 300-epoch runs in paired order: baseline/no-PIF for seeds 0, 1, and 2. Resume only from validated best-plus-two lineage.
5. Export/evaluate each no-PIF checkpoint and, only if admitted, derive/export/evaluate W8 from that same checkpoint.
6. Compare seeds 0-2. If and only if the CI intersects 0.1, activate both baseline/no-PIF seeds 3 and 4, then recompute over all five.
7. Run the final whole-lineage audit and independent artifact replay. Publish results only with `0 Critical / 0 Important`.

- [ ] **Step 9: Final handoff/merge gate**

Before proposing merge to `main`, require a clean implementation branch, exact report/artifact SHA inventory, all public paths relative, all required services inactive/disabled, released GPU lock, formal comparison terminal, and independent `0 Critical / 0 Important`. Checkpoints remain runtime/release artifacts and are not added to Git. A later FPGA deployment plan starts from the validated hardware-handoff contract and must not reinterpret fake-QDQ GPU latency as U50 evidence.

---

## Plan Completeness Self-Review

- [ ] Every Task-7 Important maps to an implementation task: paired schema/config (Task 1), trainer/resume/determinism (Task 2), paired durable controller (Task 3), no-PIF+W8/full export (Task 4), statistics/escalation (Task 5), audit/handoff (Task 6).
- [ ] Every task lists ownership, files, interfaces, RED, GREEN, exact tests, commit command, and an independent `0 Critical / 0 Important` gate.
- [ ] Bounded W8 screen precedes exactly six primary 300-epoch trainings; seeds 3/4 are conditional and indivisible.
- [ ] Shared GPU lock, safe loading, environment-before-import, complete order hashes, complete export, full COCO dual-mode evaluation, strict AP thresholds, and claim limits are explicit.
- [ ] The plan contains no deferred implementation placeholder, unsafe-load environment assignment, W8A8/PWL/Binary admission, or instruction to start CUDA/systemd before launch approval.

Run this plan-only check before handing off implementation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -B - <<'PY'
from pathlib import Path
import re

path = Path('docs/superpowers/plans/2026-08-28-mambapose-formal-stage-c.md')
text = path.read_text(encoding='utf-8')
required = (
    '### Task 1:', '### Task 2:', '### Task 3:',
    '### Task 4:', '### Task 5:', '### Task 6:',
    'CUBLAS_WORKSPACE_CONFIG=:4096:8',
    '/home/vicchen/workspace/MambaPose/work_dirs/optimization/gpu.lock',
    'Seeds `3, 4`', 'exactly six 300-epoch runs',
    '0 Critical / 0 Important',
)
missing = [item for item in required if item not in text]
forbidden = (
    'T' + 'BD', 'TO' + 'DO', 'implement ' + 'later',
    'Environment=TORCH_FORCE_' + 'NO_WEIGHTS_ONLY_LOAD=1',
)
present = [item for item in forbidden if item in text]
assert not missing, missing
assert not present, present
assert len(re.findall(r'^\*\*Files:\*\*$', text, re.MULTILINE)) == 6
assert len(re.findall(r'^\*\*Interfaces:\*\*$', text, re.MULTILINE)) == 6
assert len(re.findall(r'^- \[ \] \*\*Step 2: .*verify RED\*\*$', text, re.MULTILINE)) == 6
assert len(re.findall(r'^git commit -m "', text, re.MULTILINE)) == 6
print('formal Stage C plan completeness: PASS')
PY
git diff --check -- docs/superpowers/plans/2026-08-28-mambapose-formal-stage-c.md
sha256sum docs/superpowers/plans/2026-08-28-mambapose-formal-stage-c.md
git status --short --branch
```
