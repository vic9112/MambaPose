# MambaPose Experiment Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute, monitor, evaluate, and report the complete five-run primary matrix, four ablations, and two COCO test-dev exports on one RTX 5090.

**Architecture:** Admission is sequential: verified environment/native stack, verified data/weights, real-data smoke/resume, calibrated effective batches, then one systemd-owned GPU job at a time. Each completed run is evaluated from a hashed checkpoint and config; the report is generated only from validated artifacts.

**Tech Stack:** MMPose train/test entrypoints, RTX 5090, user systemd, JSON evidence, Markdown report

**Spec:** `docs/superpowers/specs/2026-08-18-mambapose-paper-reproduction-design.md`

## Global Constraints

- Do not bypass a failed environment, native, checkpoint-load, dataset, real-batch, or resume gate.
- Preserve the repository effective batch by gradient accumulation if the measured FP32 micro-batch is smaller.
- Primary runs remain FP32 unless a measured practical failure forces a recorded AMP deviation.
- Only one training/evaluation process owns the GPU at once.
- At most one rerun follows a concrete correction; never run a blind seed search.
- AP within 0.5 is a direct primary reproduction; ablations must preserve paper direction.

---

### Task 1: Acquire and Admit Data and VMamba Weights

**Files:**
- Runtime: `data/**`
- Runtime: `pretrained/vssm_tiny_0230_ckpt_epoch_262.pth`
- Evidence: `data/inventory.json`
- Evidence: `pretrained/inventory.json`
- Evidence: `work_dirs/reproduction/evidence/preflight.json`

**Interfaces:**
- Consumes: `reproduction/sources.json`, downloader, preflight, and official/derived source endpoints.
- Produces: complete COCO/CrowdPose layouts plus a validated VMamba-T tensor-match report.

- [ ] **Step 1: Resolve all source records without prompting**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/prepare_data.py --dry-run --json work_dirs/reproduction/evidence/source-resolution.json`

Expected: every source is classified as downloadable, already valid, or `manual_review_required`; no unknown URL/path remains.

- [ ] **Step 2: Download and atomically extract unattended sources**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/prepare_data.py --download-unattended`

Expected: downloads resume from `.part`, hashes/archive tests pass, and targets are atomically published.

- [ ] **Step 3: Resolve any officially interactive CrowdPose distribution gate**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/prepare_data.py --status`

Expected: either images are valid or the campaign records the exact official manual gate; it must never claim a complete dataset while gated.

- [ ] **Step 4: Run full path/schema/image/checkpoint preflight**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m mambapose_repro.preflight --all --output work_dirs/reproduction/evidence/preflight.json`

Expected: exit 0, exact image/annotation/detection paths pass, and VMamba load audit shows the expected backbone substantially loaded.

- [ ] **Step 5: Snapshot immutable inventory hashes**

Run: `sha256sum data/inventory.json pretrained/inventory.json work_dirs/reproduction/evidence/preflight.json > work_dirs/reproduction/evidence/preflight.sha256`

### Task 2: Real-Data Smoke, Interruption, and Batch Calibration

**Files:**
- Evidence: `work_dirs/reproduction/evidence/smoke/**`
- Evidence: `work_dirs/reproduction/evidence/calibration.json`
- Generated: `work_dirs/reproduction/resolved_configs/**`

**Interfaces:**
- Consumes: admitted environment/native/data/weight gates and five primary configs.
- Produces: real-batch train/eval proof, resume proof, per-model micro-batch/accumulation choices.

- [ ] **Step 1: Run one real FP32 train/eval batch for each architecture/dataset path**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/run_campaign.py --gate real-data-smoke`

Expected: finite loss/gradients, optimizer step, decode, and metric path for COCO S-V1/S-V2/B and CrowdPose S-V1/S-V2.

- [ ] **Step 2: Prove interruption and resume from a validated epoch checkpoint**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/run_campaign.py --gate resume-smoke --interrupt-after-checkpoint`

Expected: resumed epoch/iteration advances without replaying completed work; checkpoint/config/data/env hashes match.

- [ ] **Step 3: Calibrate largest stable FP32 micro-batch**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/run_campaign.py --gate calibrate --dtype fp32`

Expected: calibrated micro-batches, peak allocated/reserved VRAM, and no uncaught OOM.

- [ ] **Step 4: Materialize effective-batch-preserving configs**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/run_campaign.py --resolve-configs`

Expected: `micro_batch * accumulation == repository effective batch` for each run; LR remains `1e-3`.

- [ ] **Step 5: Re-run config and one-step gates on resolved files**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/run_campaign.py --gate resolved-smoke`

Expected: all resolved configs load, train one step, and carry provenance hashes.

### Task 3: Install and Start the Durable Campaign

**Files:**
- Runtime: `~/.config/systemd/user/mambapose-reproduction.service`
- Runtime: `~/.config/systemd/user/mambapose-observer.{service,timer}`
- Runtime: `work_dirs/reproduction/state.json`

**Interfaces:**
- Consumes: admitted manifest and resolved configs.
- Produces: enabled background campaign and observer, independent of Codex/terminal/network connection while user manager remains active.

- [ ] **Step 1: Install units and record linger scope**

Run: `bash tools/reproduction/install_user_service.sh`

Expected: units validate with `systemd-analyze --user verify`, service/timer are enabled, and current `Linger` is written to evidence.

- [ ] **Step 2: Start campaign and observer**

Run: `systemctl --user start mambapose-reproduction.service mambapose-observer.timer`

Expected: active service, timer scheduled, campaign lock acquired, first stage identified.

- [ ] **Step 3: Run launch-shell disconnect smoke**

Run: `systemctl --user show mambapose-reproduction.service -p MainPID -p ActiveState -p SubState`

Expected: service remains active after the launching shell exits; observer status remains fresh. Do not claim last-session/reboot survival while `Linger=no`.

- [ ] **Step 4: Confirm one GPU owner and valid health**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/observe.py --check`

Expected: one declared child process, current progress/checkpoint deadline, finite GPU telemetry, sufficient disk.

- [ ] **Step 5: Save service admission evidence**

Run: `systemctl --user status --no-pager mambapose-reproduction.service > work_dirs/reproduction/evidence/service-start.txt`

### Task 4: Monitor to Terminal Completion

**Files:**
- Runtime: `work_dirs/reproduction/status.json`
- Runtime: `work_dirs/reproduction/health.jsonl`
- Runtime: `work_dirs/reproduction/events.jsonl`
- Runtime: `work_dirs/reproduction/results.jsonl`

**Interfaces:**
- Consumes: systemd service, observer artifacts, stage validators.
- Produces: five primary checkpoints, four ablation checkpoints, evaluations, and two test-dev JSON files.

- [ ] **Step 1: Verify observer freshness at every review**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/observe.py --check`

Expected: `running` with progress within dynamic deadline, or `complete`; never accept PID-only health.

- [ ] **Step 2: Let automatic bounded recovery handle only transient failures**

Run: `systemctl --user show mambapose-reproduction.service -p NRestarts -p ExecMainStatus -p Result`

Expected: transient exits retry within persisted budget; permanent exit 78 stops and retains diagnosis.

- [ ] **Step 3: Diagnose any permanent failure from evidence order**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/run_campaign.py --diagnose-current`

Expected: diagnosis names paper/repo/assumption evidence, failure fingerprint, last valid artifact, and the smallest concrete correction before one controlled restart.

- [ ] **Step 4: Continue until every manifest stage validates**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/run_campaign.py --status --require-complete`

Expected: exit 0 only when five primary runs, four ablations, evaluations, and two submission JSONs validate.

- [ ] **Step 5: Stop restart ownership after terminal completion**

Run: `systemctl --user is-active mambapose-reproduction.service; systemctl --user is-enabled mambapose-reproduction.service`

Expected: service is inactive after success and cannot rerun completed stages; observer timer may remain active for final status.

### Task 5: Independent Result Audit and Final Report

**Files:**
- Create: `docs/reproduction/results.md`
- Evidence: `work_dirs/reproduction/evidence/final-verification.json`
- Evidence: `work_dirs/reproduction/submissions/*.json`

**Interfaces:**
- Consumes: validated metrics/checkpoints/configs/inventories and paper target matrix.
- Produces: target/measured/delta table, reproduced/deviation/unavailable classification, objective reviewer signoff.

- [ ] **Step 1: Regenerate normalized results only from artifacts**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/validate_results.py --all --output work_dirs/reproduction/evidence/final-verification.json`

Expected: finite metric set, correct hashes, target linkage, ablation directions, and test-dev marked `submission_only`.

- [ ] **Step 2: Generate the target/measured/delta report**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/analysis_tools/get_flops.py configs/reproduction/coco_s_v1.py`

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/analysis_tools/get_flops.py configs/reproduction/coco_s_v2.py`

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/analysis_tools/get_flops.py configs/reproduction/coco_b.py`

Run: `PYTHONNOUSERSITE=1 .venv/bin/python tools/reproduction/validate_results.py --render docs/reproduction/results.md`

Expected: all paper metrics, measured FLOPs, AP deltas, checkpoint/config hashes, and deviations appear; no repo-local path is confused with a paper claim.

- [ ] **Step 3: Run the complete verification suite from a clean process**

Run: `PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/test_reproduction -q`

Run: `git diff --check`

Expected: PASS and no whitespace errors.

- [ ] **Step 4: Request independent objective review**

Provide the reviewer only the spec, plans, paper/PDF, git diff/log, `final-verification.json`, service evidence, and results report. Require them to check paper matrix fidelity, artifact provenance, numerical/runtime gates, recovery behavior, and every reproduction claim.

- [ ] **Step 5: Address verified findings, rerun gates, and commit report**

Run: `git add docs/reproduction/results.md && git commit -m "docs: report MambaPose paper reproduction"`

Expected: final commit is created only after the independent reviewer has no unresolved correctness finding and all final verification commands pass.
