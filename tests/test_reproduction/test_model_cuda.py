import gc

from mmengine.config import Config
import pytest
import torch


@pytest.mark.parametrize('embed_dim', [384, 768], ids=['vim-small-width', 'vim-base-width'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_vision_mamba_width_profiles_complete_cuda_train_step(
        embed_dim, dtype):
    from mmpose.models.backbones.Vim.vim.models_mamba import VisionMamba

    torch.manual_seed(41)
    model = VisionMamba(
        img_size=32,
        patch_size=16,
        stride=16,
        depth=4,
        embed_dim=embed_dim,
        num_classes=0,
        if_cls_token=True,
        use_middle_cls_token=True,
        bimamba_type='v2').cuda().train()
    image = torch.randn(1, 3, 32, 32, device='cuda')
    context = (torch.autocast('cuda', dtype=torch.bfloat16)
               if dtype == torch.bfloat16 else torch.autocast(
                   'cuda', enabled=False))
    with context:
        output = model(image)
        loss = output.float().square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters())
    del model, image, output, loss
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_paper_s_v1_model_completes_end_to_end_cuda_train_step(dtype):
    from mmpose.registry import MODELS
    from mmpose.utils import register_all_modules

    register_all_modules()
    cfg = Config.fromfile('configs/reproduction/coco_s_v1.py')
    cfg.model.backbone.pretrained = None
    model = MODELS.build(cfg.model).cuda().train()
    image = torch.randn(1, 3, 256, 192, device='cuda')
    context = (torch.autocast('cuda', dtype=torch.bfloat16)
               if dtype == torch.bfloat16 else torch.autocast(
                   'cuda', enabled=False))
    with context:
        features = model.extract_feat(image)
        heatmaps = model.head(features)
        loss = heatmaps.float().square().mean()
    loss.backward()
    assert heatmaps.shape == (1, 17, 64, 48)
    assert torch.isfinite(heatmaps).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters())
    del model, image, features, heatmaps, loss
    gc.collect()
    torch.cuda.empty_cache()
