import pytest
import torch


@pytest.mark.parametrize('num_keypoints', [14, 17])
@pytest.mark.parametrize('mode', ['full', 'disabled', 'no_prior', 'no_cycling'])
def test_pif_modes_preserve_keypoint_output_shape(num_keypoints, mode):
    from mmpose.models.heads.heatmap_heads.pif import PoseInteraction

    layer = PoseInteraction(
        dim=256, num_keypoints=num_keypoints, mode=mode).cuda().eval()
    tokens = torch.randn(2, num_keypoints, 256, device='cuda')
    output = layer(tokens)
    assert output.shape == tokens.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize('num_keypoints', [14, 17])
@pytest.mark.parametrize('mode', ['full', 'no_cycling'])
def test_scan_indices_reconstruct_each_token_once(num_keypoints, mode):
    from mmpose.models.heads.heatmap_heads.pif import build_scan_indices

    scan, restore = build_scan_indices(num_keypoints, mode)
    assert int(scan.min()) >= 0
    assert int(scan.max()) <= num_keypoints
    assert torch.equal(scan[restore], torch.arange(num_keypoints + 1))


def _author_full_path(layer, keypoint_tokens):
    """Literal copy of the repository's active PIF body for regression."""
    topk_num = 5
    similarity = torch.matmul(
        keypoint_tokens, keypoint_tokens.transpose(1, 2))
    topk_indices = torch.topk(similarity, k=topk_num, dim=-1).indices
    batch_indices = torch.arange(
        keypoint_tokens.size(0), device=keypoint_tokens.device).view(-1, 1, 1)
    expanded = keypoint_tokens[batch_indices, topk_indices]
    expanded = expanded.view(-1, topk_num, keypoint_tokens.size(2))
    result, _ = layer.Mamba_selfScanBlock(
        torch.flip(expanded, [1]), None, inference_params=None)
    result = result.view(
        keypoint_tokens.size(0), -1, keypoint_tokens.size(2))
    expanded_new = torch.flip(result, [1])[:, 0::topk_num, :]
    keypoint_tokens = keypoint_tokens + (
        layer.param * layer.within_dropout_2(expanded_new))
    keypoint_tokens = layer.within_norm_2(keypoint_tokens)
    mean_token = keypoint_tokens.mean(dim=1, keepdim=True)
    keypoint_tokens = torch.cat((mean_token, keypoint_tokens), dim=1)
    scan, restore = layer.scan_indices, layer.restore_indices
    hidden = keypoint_tokens[:, scan, :]
    normal, _ = layer.MambaBlock(hidden, None, inference_params=None)
    reverse, _ = layer.MambaBlock2(
        torch.flip(hidden, [1]), None, inference_params=None)
    hidden = normal + torch.flip(reverse, [1])
    hidden = hidden[:, restore, :]
    keypoint_tokens = layer.within_norm(
        keypoint_tokens + layer.within_dropout(hidden))
    return keypoint_tokens[:, 1:, :]


@pytest.mark.parametrize('num_keypoints', [14, 17])
def test_full_mode_matches_the_author_active_path(num_keypoints):
    from mmpose.models.heads.heatmap_heads.pif import PoseInteraction

    torch.manual_seed(31)
    layer = PoseInteraction(
        dim=256, num_keypoints=num_keypoints, mode='full').cuda().eval()
    tokens = torch.randn(
        2, num_keypoints, 256, device='cuda', requires_grad=True)
    actual = layer(tokens)
    expected = _author_full_path(layer, tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient, = torch.autograd.grad(actual.sum(), tokens)
    assert torch.isfinite(gradient).all()


def test_disabled_mode_maps_transformer_tokens_directly():
    from mmpose.models.heads.heatmap_heads.pif import PoseInteraction

    layer = PoseInteraction(dim=256, num_keypoints=17, mode='disabled')
    tokens = torch.randn(2, 17, 256)
    assert layer(tokens) is tokens


def test_unknown_pif_mode_is_rejected():
    from mmpose.models.heads.heatmap_heads.pif import PoseInteraction

    with pytest.raises(ValueError, match='pif mode'):
        PoseInteraction(dim=256, num_keypoints=17, mode='invented')
