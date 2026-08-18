from pathlib import Path

from mmengine.config import Config
import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / 'configs/reproduction'
PRETRAIN = 'pretrained/vssm_tiny_0230_ckpt_epoch_262.pth'


@pytest.mark.parametrize(
    ('name', 'experiment_id', 'depths', 'batch', 'keypoints'),
    [
        ('coco_s_v1.py', 'coco-s-v1', [1, 1, 2, 1], 128, 17),
        ('coco_s_v2.py', 'coco-s-v2', [1, 2, 3, 2], 128, 17),
        ('coco_b.py', 'coco-b', [2, 2, 5, 2], 64, 17),
        ('crowdpose_s_v1.py', 'crowdpose-s-v1', [1, 1, 2, 1], 64, 14),
        ('crowdpose_s_v2.py', 'crowdpose-s-v2', [1, 2, 3, 2], 128, 14),
    ])
def test_primary_paper_matrix(
        name, experiment_id, depths, batch, keypoints):
    cfg = Config.fromfile(CONFIG_DIR / name)
    assert cfg.experiment_id == experiment_id
    assert list(cfg.model.backbone.depths) == depths
    assert cfg.model.backbone.pretrained == PRETRAIN
    assert cfg.model.head.num_joints == keypoints
    assert cfg.model.head.tokenpose_cfg.depth == 6
    assert cfg.model.head.tokenpose_cfg.pif_mode == 'full'
    assert cfg.train_dataloader.batch_size == batch
    assert cfg.randomness.seed == 0
    assert cfg.default_hooks.checkpoint.interval == 1
    assert cfg.default_hooks.checkpoint.max_keep_ckpts >= 2
    assert cfg.train_cfg.max_epochs == 300
    assert cfg.optim_wrapper.optimizer.type == 'Adam'
    assert cfg.optim_wrapper.optimizer.lr == pytest.approx(1e-3)
    scheduler = [s for s in cfg.param_scheduler if s.type == 'MultiStepLR'][0]
    assert list(scheduler.milestones) == [200, 260]


@pytest.mark.parametrize('name', ['crowdpose_s_v1.py', 'crowdpose_s_v2.py'])
def test_crowdpose_paths_resolve_under_data_root(name):
    cfg = Config.fromfile(CONFIG_DIR / name)
    for loader in (cfg.train_dataloader, cfg.val_dataloader,
                   cfg.test_dataloader):
        assert loader.dataset.data_root == 'data/crowdpose/'
    assert cfg.val_dataloader.dataset.bbox_file == (
        'data/crowdpose/annotations/det_for_crowd_test_0.1_0.5.json')
    assert cfg.val_evaluator.ann_file == (
        'data/crowdpose/annotations/mmpose_crowdpose_test.json')


@pytest.mark.parametrize(
    ('name', 'base_id'),
    [
        ('coco_testdev_s_v1.py', 'coco-s-v1'),
        ('coco_testdev_s_v2.py', 'coco-s-v2'),
    ])
def test_testdev_is_submission_only(name, base_id):
    cfg = Config.fromfile(CONFIG_DIR / name)
    assert cfg.checkpoint_from == base_id
    assert cfg.test_dataloader.dataset.ann_file == (
        'annotations/image_info_test-dev2017.json')
    assert cfg.test_dataloader.dataset.data_prefix.img == 'test2017/'
    assert cfg.test_dataloader.dataset.bbox_file.endswith(
        'COCO_test-dev2017_detections_AP_H_609_person.json')
    assert cfg.test_evaluator.type == 'CocoMetric'
    assert cfg.test_evaluator.format_only is True
    assert cfg.test_evaluator.outfile_prefix.startswith(
        'work_dirs/reproduction/submissions/')
    assert 'ann_file' not in cfg.test_evaluator


@pytest.mark.parametrize(
    ('name', 'mode', 'dataset'),
    [
        ('coco_s_v1_no_pif.py', 'disabled', 'coco'),
        ('crowdpose_s_v1_no_pif.py', 'disabled', 'crowdpose'),
        ('crowdpose_s_v1_no_prior.py', 'no_prior', 'crowdpose'),
        ('crowdpose_s_v1_no_cycling.py', 'no_cycling', 'crowdpose'),
    ])
def test_ablation_matrix_changes_only_declared_pif_mode(name, mode, dataset):
    cfg = Config.fromfile(CONFIG_DIR / 'ablations' / name)
    assert cfg.model.head.tokenpose_cfg.pif_mode == mode
    assert cfg.paper_target.dataset == dataset
    assert cfg.paper_target.metric == 'AP'
    assert cfg.paper_target.value > 0


def test_common_records_paper_then_repository_authority():
    cfg = Config.fromfile(CONFIG_DIR / 'mambapose_common.py')
    assert cfg.decision_authority == ['paper', 'repository', 'assumption']
    assert cfg.paper_defaults.input_size == [192, 256]
    assert cfg.paper_defaults.transformer_depth == 6
    assert cfg.paper_defaults.epochs == 300
