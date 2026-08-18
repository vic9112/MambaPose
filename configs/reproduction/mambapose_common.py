"""Paper-wide settings and decision provenance for MambaPose reproduction."""

decision_authority = ['paper', 'repository', 'assumption']
paper_defaults = dict(
    input_size=[192, 256],
    heatmap_size=[48, 64],
    transformer_depth=6,
    epochs=300,
    learning_rate=1e-3,
    milestones=[200, 260],
    pretrained='pretrained/vssm_tiny_0230_ckpt_epoch_262.pth')
assumptions = dict(seed=0, seed_search=False)

