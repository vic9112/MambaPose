import io

import pytest
import torch


def test_token_observer_keeps_one_range_per_token():
    from mmpose.models.utils.hardware_friendly import ActivationRangeObserver

    observer = ActivationRangeObserver('token')
    observer(torch.tensor([[[1.0, -2.0], [4.0, -3.0]]]))

    assert observer.max_abs.tolist() == [[2.0, 4.0]]


def test_channel_observer_uses_nchw_channel_axis_and_merges_batches():
    from mmpose.models.utils.hardware_friendly import ActivationRangeObserver

    observer = ActivationRangeObserver('channel')
    value = torch.tensor([[[[1.0, -2.0]], [[3.0, -4.0]], [[5.0, -6.0]]],
                          [[[7.0, -8.0]], [[9.0, -10.0]], [[11.0, -12.0]]]])
    observer(value)

    assert observer.max_abs.tolist() == [8.0, 10.0, 12.0]


def test_channel_observer_merges_batches_and_preserves_zeros():
    from mmpose.models.utils.hardware_friendly import ActivationRangeObserver

    first = ActivationRangeObserver('channel')
    second = ActivationRangeObserver('channel')
    first(torch.tensor([[[0.0, -2.0], [3.0, 0.0]]]))
    second(torch.tensor([[[4.0, 1.0], [0.0, -5.0]]]))
    first.merge(second)

    assert first.max_abs.tolist() == [4.0, 5.0]
    summary = first.summary(percentiles=(0.5, 0.99))
    assert summary['sample_count'] == 8
    assert summary['zero_count'] == 3
    assert summary['algorithm'] == 'fixed-log2-histogram-v1'
    assert summary['relative_error_bound'] > 0


def test_observer_rejects_nonfinite_before_mutating_state():
    from mmpose.models.utils.hardware_friendly import ActivationRangeObserver

    observer = ActivationRangeObserver('tensor')
    observer(torch.tensor([1.0]))
    before = {name: value.clone() for name, value in observer.state_dict().items()
              if isinstance(value, torch.Tensor)}

    with pytest.raises(ValueError, match='finite'):
        observer(torch.tensor([float('nan')]))

    for name, value in before.items():
        torch.testing.assert_close(observer.state_dict()[name], value)


def test_token_observer_rejects_identity_or_shape_drift():
    from mmpose.models.utils.hardware_friendly import ActivationRangeObserver

    observer = ActivationRangeObserver('token')
    observer(torch.ones(1, 2, 3), token_ids=('left', 'right'))

    with pytest.raises(ValueError, match='token identity'):
        observer(torch.ones(1, 2, 3), token_ids=('right', 'left'))
    with pytest.raises(ValueError, match='token shape'):
        observer(torch.ones(2, 2, 3), token_ids=('left', 'right'))


def test_observer_state_dict_resume_is_exact():
    from mmpose.models.utils.hardware_friendly import ActivationRangeObserver

    observer = ActivationRangeObserver('token')
    observer(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
             token_ids=('left', 'right'))
    stream = io.BytesIO()
    torch.save(observer.state_dict(), stream)
    stream.seek(0)

    resumed = ActivationRangeObserver('token')
    resumed.load_state_dict(torch.load(stream, weights_only=True))

    assert resumed.token_ids == ('left', 'right')
    assert resumed.summary() == observer.summary()
    torch.testing.assert_close(resumed.max_abs, observer.max_abs)
