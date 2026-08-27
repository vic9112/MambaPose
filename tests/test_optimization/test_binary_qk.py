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


def test_transformer_binary_mode_is_restricted_to_its_six_attention_layers():
    from mmpose.models.heads.heatmap_heads.tokenbase import Transformer

    transformer = Transformer(
        dim=16, depth=6, heads=4, mlp_dim=32, dropout=0.0,
        num_keypoints=2, qk_mode='binary')
    modes = [layer[0].fn.fn.qk_mode for layer in transformer.layers]

    assert modes == ['binary'] * 6
    with pytest.raises(ValueError, match='qk_mode'):
        Transformer(dim=16, depth=6, heads=4, mlp_dim=32, dropout=0.0,
                    qk_mode='vmamba')
