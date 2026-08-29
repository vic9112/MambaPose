import pytest
import torch


def test_binary_qk_signed_dot_reference_and_deterministic_zero():
    from mmpose.models.utils.hardware_friendly import binary_qk_logits, ste_sign

    q = torch.tensor([[[[1.0, -2.0, 3.0, -4.0]]]])
    k = torch.tensor([[[[-1.0, -2.0, 3.0, 4.0]]]])

    assert binary_qk_logits(q, k).item() == 0.0
    assert ste_sign(torch.tensor([0.0])).item() == 1.0


def test_binary_qk_ste_has_gradient():
    from mmpose.models.utils.hardware_friendly import binary_qk_logits

    q = torch.randn(1, 2, 3, 4, requires_grad=True)
    k = torch.randn(1, 2, 3, 4, requires_grad=True)
    binary_qk_logits(q, k).sum().backward()

    assert q.grad is not None
    assert k.grad is not None


def test_binary_scaled_qk_matches_per_batch_head_abs_mean_formula():
    from mmpose.models.utils.hardware_friendly import (
        binary_qk_logits,
        binary_scaled_qk_logits,
    )

    q = torch.tensor([[[[1.0, -3.0], [5.0, -7.0]]]])
    k = torch.tensor([[[[-2.0, 4.0], [6.0, -8.0]]]])
    q_scale = q.abs().mean(dim=-2, keepdim=True).mean(
        dim=-1, keepdim=True)
    k_scale = k.abs().mean(dim=-2, keepdim=True).mean(
        dim=-1, keepdim=True)
    expected = binary_qk_logits(q, k) * q_scale * k_scale

    torch.testing.assert_close(
        binary_scaled_qk_logits(q, k), expected, rtol=0, atol=0)


def test_binary_scaled_qk_has_finite_qk_gradients():
    from mmpose.models.utils.hardware_friendly import (
        binary_scaled_qk_logits,
    )

    q = torch.randn(2, 3, 5, 4, requires_grad=True)
    k = torch.randn(2, 3, 5, 4, requires_grad=True)
    binary_scaled_qk_logits(q, k).square().mean().backward()

    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert q.grad.abs().sum() > 0
    assert k.grad.abs().sum() > 0


def test_float_qk_mode_keeps_checkpoint_keys_and_operation_exact():
    from mmpose.models.heads.heatmap_heads.tokenbase import Attention

    torch.manual_seed(7)
    original = Attention(16, heads=4, dropout=0.0, scale_with_head=True).eval()
    configurable = Attention(
        16, heads=4, dropout=0.0, scale_with_head=True,
        qk_mode='float').eval()
    configurable.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(2, 11, 16)

    assert configurable.state_dict().keys() == original.state_dict().keys()
    original_result = original(x)
    configured_result = configurable(x)
    for actual, expected in zip(configured_result, original_result):
        if actual is None:
            assert expected is None
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_binary_scaled_mode_adds_no_checkpoint_parameters():
    from mmpose.models.heads.heatmap_heads.tokenbase import Attention

    original = Attention(
        16, heads=4, dropout=0.0, scale_with_head=True).eval()
    scaled = Attention(
        16, heads=4, dropout=0.0, scale_with_head=True,
        qk_mode='binary_scaled').eval()

    scaled.load_state_dict(original.state_dict(), strict=True)
    assert scaled.state_dict().keys() == original.state_dict().keys()


def test_transformer_binary_mode_is_restricted_to_its_six_attention_layers():
    from mmpose.models.heads.heatmap_heads.tokenbase import Transformer

    transformer = Transformer(
        dim=16, depth=6, heads=4, mlp_dim=32, dropout=0.0,
        num_keypoints=2, qk_mode='binary')
    modes = [layer[0].fn.fn.qk_mode for layer in transformer.layers]

    assert modes == ['binary'] * 6
    scaled = Transformer(
        dim=16, depth=6, heads=4, mlp_dim=32, dropout=0.0,
        num_keypoints=2, qk_mode='binary_scaled')
    assert [layer[0].fn.fn.qk_mode for layer in scaled.layers] \
        == ['binary_scaled'] * 6
    with pytest.raises(ValueError, match='qk_mode'):
        Transformer(dim=16, depth=6, heads=4, mlp_dim=32, dropout=0.0,
                    qk_mode='vmamba')
