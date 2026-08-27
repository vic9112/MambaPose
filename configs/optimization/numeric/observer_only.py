_base_ = ['../coco_s_v1_deterministic.py']

numeric_optimization = dict(
    schema_version=1, route='ssm-quant-pwl', candidate_kind='observer',
    stage_order=('calibrate',),
    calibration=dict(
        source_candidate='full-s-v1', split='train2017', shuffle=False,
        worker_count=0, sample_count=512,
        artifact_binding='runtime-sha256-required'),
)
