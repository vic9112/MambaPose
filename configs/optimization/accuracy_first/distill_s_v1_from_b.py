_base_ = ['../../reproduction/coco_s_v1.py']

student_model = _base_.model

model = dict(
    _delete_=True,
    type='MambaPoseHeatmapDistiller',
    teacher='configs/reproduction/coco_b.py',
    student=student_model,
    teacher_checkpoint=(
        'work_dirs/reproduction/runs/coco-b/'
        'best_coco_AP_epoch_290.pth'),
    teacher_checkpoint_sha256=(
        '38b5e5b1bccfdf7b8b153d91837f7f1bfe371a14f12f6e417a52ebb893efb9b2'),
    student_checkpoint=(
        'work_dirs/reproduction/runs/coco-s-v1/'
        'best_coco_AP_epoch_300.pth'),
    student_checkpoint_sha256=(
        'a6f76dae86db4d92c445f26a428b61911e8b42c4de2119348997e9537cc7cdd2'),
    # DWPose stage-2 uses unit-weight output distillation for 20% of its base
    # schedule. Apply the same output-loss ratio to aligned heatmaps.
    heatmap_loss_weight=1.0,
    data_preprocessor=student_model.data_preprocessor)

# DWPose stage-2 runs for 60 epochs. Starting from the reproduced 300-epoch
# S-V1 checkpoint, hold the learning rate at the MambaPose base schedule's
# final (post-260) value to preserve the strict AP budget.
train_cfg = dict(max_epochs=60, val_interval=5)
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-5))
param_scheduler = []

experiment_id = 'distill-s-v1-from-coco-b'
optimization_route = 'accuracy-first'
distillation_provenance = dict(
    teacher_candidate_id='coco-b-teacher',
    student_candidate_id='full-s-v1',
    teacher_ap=74.895,
    student_ap=72.83223444297235,
    target='heatmaps',
    inference_cost='student-only')
