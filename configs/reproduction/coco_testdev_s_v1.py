_base_ = ['./coco_s_v1.py']

experiment_id = 'coco-testdev-s-v1'
checkpoint_from = 'coco-s-v1'
paper_target = dict(
    dataset='coco',
    split='test-dev2017',
    metric='AP',
    value=72.4,
    metrics=dict(AP=72.4, APM=69.5, APL=77.7),
    local_status='submission_only')
test_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/coco/',
        data_mode='topdown',
        ann_file='annotations/image_info_test-dev2017.json',
        bbox_file=(
            'data/coco/person_detection_results/'
            'COCO_test-dev2017_detections_AP_H_609_person.json'),
        data_prefix=dict(img='test2017/'),
        test_mode=True,
        pipeline={{_base_.val_pipeline}}))
test_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    format_only=True,
    outfile_prefix=(
        'work_dirs/reproduction/submissions/coco-testdev-s-v1/predictions'))

