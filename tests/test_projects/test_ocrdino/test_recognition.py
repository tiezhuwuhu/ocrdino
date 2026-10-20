# Copyright (c) OpenMMLab. All rights reserved.
"""Behavioral tests for width selection, CTC and shared-feature gradients."""

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('mmcv._ext')
pytest.importorskip('mmengine')

from projects.OCRDINO.ocrdino import recognition  # noqa: E402
from projects.OCRDINO.ocrdino.recognition import LightCTCHead  # noqa: E402


def make_head(**kwargs):
    args = dict(
        dictionary=['A', 'B'],
        in_channels=4,
        hidden_channels=8,
        lstm_hidden_size=4,
        gn_groups=4,
        width_buckets=(8, 16, 32),
        width_scale=8.)
    args.update(kwargs)
    return LightCTCHead(**args)


def test_geometry_selects_width_and_caps_maximum():
    head = make_head()
    rois = torch.tensor([
        [0, 0, 0, 10, 10],  # 8 -> 8
        [0, 0, 0, 11, 10],  # 9 -> 16
        [0, 0, 0, 20, 10],  # 16 stays 16
        [0, 0, 0, 21, 10],  # 17 -> 32
        [0, 0, 0, 80, 10],  # 64 capped at 32
    ], dtype=torch.float32)
    assert head.roi_widths(rois).tolist() == [8, 16, 16, 32, 32]
    assert head.roi_widths(rois[:0]).shape == (0, )
    with pytest.raises(ValueError, match='positive area'):
        head.roi_widths(torch.tensor([[0., 2., 0., 1., 1.]]))


def test_repeated_characters_require_extra_ctc_steps():
    head = make_head(width_buckets=(4, 8), width_scale=4.)
    assert head.required_ctc_steps('AB') == 2
    assert head.required_ctc_steps('AAB') == 4
    assert head.required_ctc_steps('AABB') == 6
    c2 = torch.randn(1, 4, 10, 10)
    rois = torch.tensor([[0., 0., 0., 40., 40.]])
    with pytest.raises(ValueError, match='needs at least 6 CTC steps'):
        head.loss(c2, rois, ['AABB'])
    with pytest.raises(ValueError):
        head.loss(c2, rois, ['Z'])
    with pytest.raises(ValueError, match='nonempty string'):
        head.loss(c2, rois, [''])


def test_cpu_roi_loss_updates_head_and_shared_c2():
    torch.manual_seed(3)
    head = make_head()
    c2 = torch.randn(2, 4, 16, 40, requires_grad=True)
    rois = torch.tensor([[0., 0., 0., 32., 32.],
                         [1., 8., 4., 88., 36.]])
    loss = head.loss(c2, rois, ['AB', 'AAB'])['loss_rec_ctc']
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert c2.grad is not None and c2.grad.abs().sum() > 0
    assert torch.isfinite(c2.grad).all()
    for parameter in head.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_empty_rank_participates_in_reduction_and_all_parameter_gradients(
        monkeypatch):
    head = make_head()
    c2 = torch.randn(1, 4, 8, 8, requires_grad=True)
    calls = []

    def fake_reduce_mean(local_count):
        calls.append(local_count.item())
        return local_count.new_tensor(0.5)  # One region across two ranks.

    monkeypatch.setattr(recognition, 'reduce_mean', fake_reduce_mean)
    loss = head.loss(c2, torch.empty(0, 5), [])['loss_rec_ctc']
    assert calls == [0.]
    assert loss.item() == 0. and torch.isfinite(loss)
    loss.backward()
    assert c2.grad is not None
    for parameter in head.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0
    texts, scores = head.predict(c2, torch.empty(0, 5))
    assert texts == [] and scores.shape == (0, )


def test_global_normalization_does_not_clamp_positive_fractional_count(
        monkeypatch):
    torch.manual_seed(2)
    head = make_head()
    c2 = torch.randn(1, 4, 10, 10)
    rois = torch.tensor([[0., 0., 0., 32., 32.]])
    local_loss = head.loss(c2, rois, ['AB'])['loss_rec_ctc']
    monkeypatch.setattr(recognition, 'reduce_mean',
                        lambda count: count.new_tensor(0.5))
    ddp_loss = head.loss(c2, rois, ['AB'])['loss_rec_ctc']
    torch.testing.assert_close(ddp_loss, local_loss * 2)


def test_greedy_ctc_preserves_repeats_separated_by_blank_and_roi_order(
        monkeypatch):
    head = make_head()
    c2 = torch.randn(1, 4, 8, 16)
    rois = torch.tensor([[0., 0., 0., 32., 32.],
                         [0., 0., 0., 64., 32.]])

    def fake_logits(_c2, _rois, _widths):
        # Return bucket groups out of RoI order. A A blank A B B -> AAB.
        for index, path in [(1, [1, 1, 0, 1, 2, 2]),
                            (0, [0, 0, 0, 0, 0, 0])]:
            logits = torch.full((1, len(path), 3), -8.)
            for time_step, char in enumerate(path):
                logits[0, time_step, char] = 8.
            yield torch.tensor([index]), logits

    monkeypatch.setattr(head, '_iter_logits', fake_logits)
    texts, scores = head.predict(c2, rois)
    assert texts == ['', 'AAB']
    assert scores[0] == 0.
    assert 0.99 < scores[1] <= 1.


def test_checkpoint_cannot_silently_reorder_same_size_dictionary():
    original = make_head(dictionary=['A', 'B'])
    reordered = make_head(dictionary=['B', 'A'])
    with pytest.raises(RuntimeError, match='dictionary mismatch'):
        reordered.load_state_dict(original.state_dict(), strict=False)
    make_head().load_state_dict(original.state_dict())


def test_mmengine_strict_false_cannot_ignore_dictionary_mismatch():
    from mmengine.runner.checkpoint import load_state_dict
    original = make_head(dictionary=['A', 'B'])
    reordered = make_head(dictionary=['B', 'A'])
    with pytest.raises(RuntimeError, match='dictionary mismatch'):
        load_state_dict(reordered, original.state_dict(), strict=False)


def test_mmengine_strict_false_requires_recognition_dictionary_hash():
    from mmengine.runner.checkpoint import load_state_dict
    state_dict = make_head().state_dict()
    state_dict.pop('dictionary_hash')
    with pytest.raises(RuntimeError, match='no dictionary_hash'):
        load_state_dict(make_head(), state_dict, strict=False)


def test_recognition_weights_require_dictionary_fingerprint():
    head = make_head()
    state_dict = head.state_dict()
    state_dict.pop('dictionary_hash')
    with pytest.raises(RuntimeError, match='no dictionary_hash'):
        make_head().load_state_dict(state_dict, strict=False)
    # Detection-only initialization does not have recognition parameters.
    make_head().load_state_dict({}, strict=False)


@pytest.mark.parametrize('kwargs', [
    dict(width_buckets=(16, 8)),
    dict(width_buckets=(8, 8)),
    dict(roi_height=16),
    dict(width_scale=0.),
    dict(gn_groups=3),
])
def test_invalid_configuration_fails_early(kwargs):
    with pytest.raises(ValueError):
        make_head(**kwargs)
