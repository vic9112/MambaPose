import pytest
import torch
import torch.nn.functional as F


def test_pwl_rejects_unsorted_or_discontinuous_segments():
    from mmpose.models.utils.hardware_friendly import PiecewiseLinearApproximation

    with pytest.raises(ValueError, match='strictly increasing'):
        PiecewiseLinearApproximation([0.0, -1.0, 1.0], [1.0, 1.0], [0.0, 0.0])
    with pytest.raises(ValueError, match='continuous'):
        PiecewiseLinearApproximation([-1.0, 0.0, 1.0], [1.0, 2.0], [0.0, 1.0])


def test_pwl_selects_segments_and_saturates_at_declared_domain():
    from mmpose.models.utils.hardware_friendly import PiecewiseLinearApproximation

    approximation = PiecewiseLinearApproximation(
        [-1.0, 0.0, 1.0], [1.0, 2.0], [0.0, 0.0])
    actual = approximation(torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0]))

    torch.testing.assert_close(
        actual, torch.tensor([-1.0, -0.5, 0.0, 1.0, 2.0]))
    assert approximation.domain == (-1.0, 1.0)
    assert approximation.segments == 2
    assert approximation.saturation == 'clamp'


@pytest.mark.parametrize('name, reference, domain', [
    ('silu', F.silu, (-6.0, 6.0)),
    ('gelu', F.gelu, (-5.0, 5.0)),
    ('softplus', F.softplus, (-8.0, 8.0)),
    ('exp', torch.exp, (-4.0, 4.0)),
])
def test_fit_pwl_is_deterministic_continuous_and_declares_error(
        name, reference, domain):
    from mmpose.models.utils.hardware_friendly import fit_pwl

    first = fit_pwl(reference, domain, segments=16, grid_points=4097,
                    function_name=name)
    second = fit_pwl(reference, domain, segments=16, grid_points=4097,
                     function_name=name)

    assert first.function_name == name
    assert first.state_dict().keys() == second.state_dict().keys()
    for key in first.state_dict():
        torch.testing.assert_close(first.state_dict()[key], second.state_dict()[key])
    assert first.max_error >= first.mean_error >= 0
    x = torch.linspace(*domain, 101, requires_grad=True)
    first(x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_disabled_pwl_is_bit_exact_and_has_no_state_keys():
    from mmpose.models.utils.hardware_friendly import PiecewiseLinearApproximation

    approximation = PiecewiseLinearApproximation.identity()
    x = torch.randn(17)

    assert approximation.state_dict() == {}
    torch.testing.assert_close(approximation(x), x, rtol=0, atol=0)
