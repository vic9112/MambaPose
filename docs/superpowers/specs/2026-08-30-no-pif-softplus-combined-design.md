# No-PIF + Tail-Aware Softplus PWL Combined Candidate

## Decision

The executable base is PWL commit
`f71fcff33bebe32241d0825eb2b736605a926183`. The independently evolved
accuracy-first history is not merged into the PWL controller. Instead, the
tracked parent-authority record binds the exact no-PIF source commit
`d938530bad75fa6a8ac515b2c313792d8150da49`, its config and checkpoint, and the
PWL source commit, selected Softplus config, and isolated v6 artifact hashes.
The six v6 JSON artifacts and the full-S-V1 evaluation are committed as
byte-identical, SHA-bound snapshots, so authority does not depend on retained
sibling worktrees.

This preserves both parents without combining two controller implementations.
The combined checkpoint parent is the published no-PIF S-V1 checkpoint. The
full S-V1 candidate is declared as the evaluation comparator.

## Algorithm contract

- The no-PIF checkpoint is loaded strictly before the bypassed PIF children
  are replaced by a parameter-free identity. This removes the PIF runtime
  modules and operations without weakening checkpoint tensor validation.
- Exactly five VMamba SS2D transition roles receive 16-segment Softplus PWL on
  `[-8, 8]` with `continuous-asymptotic-tail-v1` handling.
- Calibration uses COCO train2017 and the exact no-PIF source candidate, not
  full S-V1.
- The stage order is `calibrate`, `convert`, `smoke-stage-a`, `profile`,
  `evaluate`, `compare`, `latency`. The CPU-only compare stage authenticates
  the frozen full-S-V1 evaluation, consumes the combined evaluation, verifies
  their COCO protocols and model identities, and recomputes all flip/no-flip
  metric differences. The prior four-function selection stage is not rerun;
  conversion is admitted by the hash-bound combined parent record.
- Stage-A and every downstream stage retain the existing PWL fit,
  installation, runtime-config, checkpoint, GPU-lease, and artifact validators.
  Latency binds the comparison artifact, so the terminal result transitively
  retains both evaluations.

## Claim boundary

The isolated Softplus result and the isolated no-PIF result must not be added.
The combined accuracy and interaction effect remain unmeasured until this
candidate completes evaluation against full S-V1. PyTorch latency is not FPGA
latency or resource evidence, and no FPGA speedup is claimed.
The direct single-checkpoint comparison is only a preliminary isolated screen:
it reports both `candidate_minus_full` and `drop_full_minus_candidate`, and it
cannot claim the multi-seed formal paired pass.

## CPU preflight

From a clean checkout of this branch:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  tools/optimization/audit_combined_candidate.py
```

The audit strictly loads the no-PIF checkpoint on CPU, verifies both parent
bindings, confirms PIF pruning, and installs the exact five tail-aware
Softplus topology probes. It does not start a campaign or use the GPU.
