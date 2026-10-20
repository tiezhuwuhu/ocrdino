# Copyright (c) OpenMMLab. All rights reserved.
"""Real tensor/ops tests for parallel Character Query recognition.

These tests require real PyTorch, MMEngine and compiled MMCV ops. There are
no substituted attention/ROI kernels. Small dimensions keep CPU tests quick;
separate assertions verify the production default architecture.
"""

import copy

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('mmcv._ext')
pytest.importorskip('mmengine')

from mmdet.registry import MODELS  # noqa: E402
from projects.OCRDINO.ocrdino import recognition_query as recognition  # noqa: E402
from projects.OCRDINO.ocrdino.recognition_query import (  # noqa: E402
    CharacterDecoderLayer, CharacterQueryHead, sinusoidal_positions)


def make_head(**kwargs):
    args = dict(dictionary=['A', 'B'], in_channels=4, hidden_channels=8,
                gn_groups=4, embed_dims=16, num_layers=2, num_heads=4,
                ffn_dim=32, dropout=0., width_buckets=(8, 16, 32),
                width_scale=8., query_buckets=(8, 16, 32),
                query_width_scale=8., chunk_size=2)
    args.update(kwargs)
    return CharacterQueryHead(**args)


def fixture_inputs():
    c2 = torch.randn(2, 4, 16, 40, requires_grad=True)
    rois = torch.tensor([[0., 0., 0., 32., 32.],
                         [1., 8., 4., 88., 36.],
                         [0., 4., 0., 48., 32.]])
    return c2, rois, ['AB', 'AAB', 'BA']


def test_registered_head_has_required_production_defaults():
    head = MODELS.build(dict(type='CharacterQueryHead',
                             dictionary=[chr(i) for i in range(32, 127)]))
    assert isinstance(head, CharacterQueryHead)
    assert len(head.decoder) == 3 and head.embed_dims == 256
    assert head.decoder[0].self_attn.num_heads == 8
    assert head.decoder[0].linear1.out_features == 1024
    assert head.decoder[0].self_attn.dropout == .1
    assert head.classifier.out_features == 96
    assert head.dictionary.eos_idx == 95
    assert not hasattr(head.dictionary, 'blank_idx')
    assert not hasattr(head.dictionary, 'pad_idx')
    assert not any(isinstance(module, (torch.nn.LSTM, torch.nn.CTCLoss))
                   for module in head.modules())
    assert head.width_buckets == head.query_buckets == (32, 64, 128, 256, 500)
    assert head.chunk_size == 8 and head.with_cp


def test_positions_and_shared_content_define_distinct_ordered_queries():
    head = make_head()
    position = sinusoidal_positions(7, 16)
    assert position.shape == (1, 7, 16)
    assert torch.equal(position[0, 0, 0::2], torch.zeros(8))
    assert torch.equal(position[0, 0, 1::2], torch.ones(8))
    assert not torch.equal(position[:, 0], position[:, 1])
    queries = head.character_queries(3, 7)
    torch.testing.assert_close(queries, (head.shared_content + position).expand(3, -1, -1))
    assert head.shared_content.numel() == 16


def test_cross_attention_reads_positioned_keys_but_unmodified_visual_values():
    torch.manual_seed(7)
    layer = CharacterDecoderLayer(16, 4, 32, 0.)
    queries = torch.randn(2, 8, 16, requires_grad=True)
    visual = torch.randn(2, 12, 16, requires_grad=True)
    positions = sinusoidal_positions(12, 16)
    captured = []

    def capture(_module, args, kwargs):
        captured.append((args, kwargs))

    handle = layer.cross_attn.register_forward_pre_hook(capture, with_kwargs=True)
    output = layer(queries, visual, positions)
    handle.remove()
    args, kwargs = captured[0]
    torch.testing.assert_close(args[1], visual + positions)
    torch.testing.assert_close(args[2], visual)
    assert kwargs['need_weights'] is False
    assert kwargs.get('attn_mask') is None
    assert output.shape == (2, 8, 16)
    output.square().mean().backward()
    assert visual.grad is not None and visual.grad.abs().sum() > 0
    assert queries.grad is not None


def test_decoder_parallel_shape_and_visual_gradient():
    head = make_head()
    visual = torch.randn(2, 13, 16, requires_grad=True)
    logits = head.decode_visual(visual, 8)
    assert logits.shape == (2, 8, 3)
    logits.sum().backward()
    assert visual.grad.abs().sum() > 0
    assert head.shared_content.grad.abs().sum() > 0


def test_hello_targets_have_one_eos_no_pad_class_and_keep_ll():
    head = make_head(dictionary=['H', 'E', 'L', 'O'])
    targets, lengths = head.encode_targets(['HELLO'], 8)
    assert targets.tolist() == [[0, 1, 2, 2, 3, 4, -100, -100]]
    assert lengths.tolist() == [5]
    assert head.dictionary.decode([0, 1, 2, 2, 3, 4, 0]) == 'HELLO'
    assert head.required_output_positions('HELLO') == 6
    assert head.required_output_positions('LL') == 3


def test_region_loss_matches_character_mean_plus_eos():
    head = make_head(lambda_eos=.7)
    targets, lengths = head.encode_targets(['A', 'ABB'], 8)
    logits = torch.randn(2, 8, 3)
    actual = head._region_losses(logits, targets, lengths)
    expected = []
    for index, length in enumerate(lengths.tolist()):
        char_ce = torch.nn.functional.cross_entropy(
            logits[index, :length], targets[index, :length])
        eos_ce = torch.nn.functional.cross_entropy(
            logits[index, length:length + 1], targets[index, length:length + 1])
        expected.append(char_ce + .7 * eos_ce)
    torch.testing.assert_close(actual, torch.stack(expected))


def test_slots_after_eos_have_exactly_zero_loss_gradient():
    head = make_head()
    targets, lengths = head.encode_targets(['AB'], 8)
    logits = torch.randn(1, 8, 3, requires_grad=True)
    loss = head._region_losses(logits, targets, lengths).sum()
    changed = logits.detach().clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 100.
    torch.testing.assert_close(loss.detach(), head._region_losses(changed, targets, lengths).sum())
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, 3:]) == 0
    assert logits.grad[:, :3].abs().sum() > 0


def test_all_eos_is_penalized_more_than_correct_characters_and_eos():
    head = make_head(dictionary=['H', 'E', 'L', 'O'])
    targets, lengths = head.encode_targets(['HELLO'], 8)
    correct = torch.full((1, 8, 5), -10.)
    for index, target in enumerate(targets[0, :6]):
        correct[0, index, target] = 10.
    all_eos = torch.full_like(correct, -10.)
    all_eos[:, :, head.dictionary.eos_idx] = 10.
    good = head._region_losses(correct, targets, lengths)
    bad = head._region_losses(all_eos, targets, lengths)
    assert good.item() < 1e-5
    assert bad.item() > 10. and bad.item() > good.item()


def test_geometry_buckets_are_independent_and_identical_in_train_eval():
    head = make_head(width_buckets=(8, 32), query_buckets=(8, 16, 32))
    rois = torch.tensor([[0, 0, 0, 10, 10], [0, 0, 0, 11, 10],
                         [0, 0, 0, 20, 10], [0, 0, 0, 21, 10],
                         [0, 0, 0, 80, 10]], dtype=torch.float32)
    head.train()
    widths, counts = head.roi_widths(rois), head.roi_query_counts(rois)
    assert widths.tolist() == [8, 32, 32, 32, 32]
    assert counts.tolist() == [8, 16, 16, 32, 32]
    head.eval()
    torch.testing.assert_close(widths, head.roi_widths(rois))
    torch.testing.assert_close(counts, head.roi_query_counts(rois))
    assert head.roi_query_counts(rois[:0]).shape == (0,)
    with pytest.raises(ValueError, match='positive area'):
        head.roi_query_counts(torch.tensor([[0., 2., 0., 1., 1.]]))


def test_capacity_and_unknown_text_raise_before_sampling():
    head = make_head(query_buckets=(4, 8), query_width_scale=4.)
    rois = torch.tensor([[0., 0., 0., 32., 32.]])
    assert head.validate_targets(rois, ['AAA']) == [[0, 0, 0]]
    with pytest.raises(ValueError, match="'AAAA'.*needs 5.*bucket is 4"):
        head.validate_targets(rois, ['AAAA'])
    with pytest.raises(ValueError, match='not present'):
        head.validate_targets(rois, ['Z'])
    with pytest.raises(ValueError, match='empty transcription'):
        head.validate_targets(rois, [''])
    with pytest.raises(ValueError, match='One transcription'):
        head.validate_targets(rois, [])


def test_real_roi_convolution_shapes_and_loss_gradients_reach_shared_c2():
    torch.manual_seed(3)
    head = make_head(with_cp=False)
    c2, rois, texts = fixture_inputs()
    input_shapes, output_shapes = [], []
    pre = head.convs.register_forward_pre_hook(
        lambda _module, args: input_shapes.append(tuple(args[0].shape)))
    post = head.convs.register_forward_hook(
        lambda _module, _args, result: output_shapes.append(tuple(result.shape)))
    loss = head.loss(c2, rois, texts)['loss_rec']
    pre.remove()
    post.remove()
    assert torch.isfinite(loss) and loss > 0
    assert input_shapes == [(1, 8, 8, 8), (1, 8, 8, 16), (1, 8, 8, 32)]
    assert output_shapes == [(1, 8, 1, 8), (1, 8, 1, 16), (1, 8, 1, 32)]
    loss.backward()
    assert c2.grad is not None and c2.grad.abs().sum() > 0
    assert torch.isfinite(c2.grad).all()
    for parameter in head.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_chunking_and_nonreentrant_checkpoint_keep_loss_and_gradients():
    torch.manual_seed(4)
    full = make_head(chunk_size=8, with_cp=False)
    chunked = make_head(chunk_size=1, with_cp=True)
    chunked.load_state_dict(full.state_dict())
    c2, rois, texts = fixture_inputs()
    # Two same-bucket regions exercise actual splitting, different buckets
    # catch accidental late-bound closure values during backward recompute.
    rois = torch.cat([rois, rois[:1]], dim=0)
    texts.append('BB')
    other = c2.detach().clone().requires_grad_()
    first = full.loss(c2, rois, texts)['loss_rec']
    second = chunked.loss(other, rois, texts)['loss_rec']
    torch.testing.assert_close(first, second, rtol=2e-5, atol=2e-6)
    first.backward()
    second.backward()
    torch.testing.assert_close(c2.grad, other.grad, rtol=2e-4, atol=2e-6)
    for (name1, param1), (name2, param2) in zip(full.named_parameters(), chunked.named_parameters()):
        assert name1 == name2
        torch.testing.assert_close(param1.grad, param2.grad, rtol=2e-4, atol=2e-6)


def test_checkpoint_recomputation_preserves_dropout_rng():
    torch.manual_seed(8)
    plain = make_head(dropout=.1, chunk_size=1, with_cp=False)
    checked = make_head(dropout=.1, chunk_size=1, with_cp=True)
    checked.load_state_dict(plain.state_dict())
    c2, rois, texts = fixture_inputs()
    other = c2.detach().clone().requires_grad_()
    torch.manual_seed(100)
    first = plain.loss(c2, rois, texts)['loss_rec']
    first.backward()
    torch.manual_seed(100)
    second = checked.loss(other, rois, texts)['loss_rec']
    second.backward()
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(c2.grad, other.grad, rtol=2e-4, atol=2e-6)


def test_checkpoint_reduces_saved_regional_activation_storage():
    torch.manual_seed(11)
    head = make_head(chunk_size=1, width_buckets=(32,), query_buckets=(32,))
    c2, rois, texts = fixture_inputs()

    def saved_bytes(with_cp):
        head.with_cp = with_cp
        storage = {}

        def pack(tensor):
            block = tensor.untyped_storage()
            storage[block.data_ptr()] = block.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss = head.loss(c2, rois, texts)['loss_rec']
        assert loss.requires_grad
        return sum(storage.values())

    assert saved_bytes(True) < saved_bytes(False)


def test_empty_rank_still_reduces_and_touches_all_parameters(monkeypatch):
    head = make_head()
    c2 = torch.randn(1, 4, 8, 8, requires_grad=True)
    counts = []

    def fake_reduce(count):
        counts.append(count.item())
        return count.new_tensor(.5)

    monkeypatch.setattr(recognition, 'reduce_mean', fake_reduce)
    loss = head.loss(c2, torch.empty(0, 5), [])['loss_rec']
    assert counts == [0.] and loss.item() == 0.
    loss.backward()
    assert c2.grad is not None
    for parameter in head.parameters():
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
    texts, scores, terminated = head.predict(c2, torch.empty(0, 5))
    assert texts == [] and scores.shape == terminated.shape == (0,)


def test_fractional_ddp_mean_is_not_clamped_to_one(monkeypatch):
    head = make_head(with_cp=False)
    c2, rois, texts = fixture_inputs()
    single = head.loss(c2, rois[:1], texts[:1])['loss_rec']
    monkeypatch.setattr(recognition, 'reduce_mean', lambda count: count.new_tensor(.5))
    distributed = head.loss(c2, rois[:1], texts[:1])['loss_rec']
    torch.testing.assert_close(distributed, single * 2)


def test_decode_preserves_repeats_eos_and_box_order(monkeypatch):
    head = make_head(dictionary=['H', 'E', 'L', 'O'])
    c2 = torch.randn(1, 4, 8, 16)
    rois = torch.tensor([[0., 0., 0., 32., 32.],
                         [0., 0., 0., 64., 32.],
                         [0., 0., 0., 48., 32.]])

    def fake_logits(_c2, _rois, _widths, _counts):
        for index, path in [(1, [0, 1, 2, 2, 3, 4, 0]),
                            (2, [2, 2, 2, 2]), (0, [4, 0, 0])]:
            logits = torch.full((1, len(path), 5), -8.)
            for pos, char in enumerate(path):
                logits[0, pos, char] = 8.
            yield torch.tensor([index]), logits

    monkeypatch.setattr(head, '_iter_logits', fake_logits)
    texts, scores, terminated = head.predict(c2, rois)
    assert texts == ['', 'HELLO', 'LLLL']
    assert terminated.tolist() == [True, True, False]
    assert scores[0] == 0. and (scores[1:] > .99).all()


def test_checkpoint_dictionary_and_architecture_guards_also_reject_strict_false():
    from mmengine.runner.checkpoint import load_state_dict
    original = make_head()
    state = original.state_dict()
    make_head().load_state_dict(state)
    with pytest.raises(RuntimeError, match='dictionary mismatch'):
        load_state_dict(make_head(dictionary=['B', 'A']), state, strict=False)
    for key in ('dictionary_hash', 'recognition_version'):
        broken = copy.copy(state)
        del broken[key]
        with pytest.raises(RuntimeError, match='no ' + key):
            load_state_dict(make_head(), broken, strict=False)
    old = {'lstm.weight_ih_l0': torch.zeros(16, 8),
           'dictionary_hash': original.dictionary_hash.clone()}
    with pytest.raises(RuntimeError, match='Old CTC'):
        load_state_dict(make_head(), old, strict=False)
    incompatible = copy.copy(state)
    incompatible['recognition_version'] = torch.tensor(999)
    with pytest.raises(RuntimeError, match='recognition_version is incompatible'):
        make_head().load_state_dict(incompatible, strict=False)
    make_head().load_state_dict({}, strict=False)  # COCO detection-only init.


@pytest.mark.parametrize('kwargs', [
    dict(width_buckets=(16, 8)), dict(query_buckets=(8, 8)),
    dict(roi_height=16), dict(width_scale=0.), dict(query_width_scale=float('nan')),
    dict(gn_groups=3), dict(chunk_size=0), dict(embed_dims=15),
    dict(lambda_eos=-1.), dict(dropout=1.),
])
def test_invalid_configurations_fail_early(kwargs):
    with pytest.raises(ValueError):
        make_head(**kwargs)


def test_500_query_capacity_reserves_one_slot_for_eos():
    head = make_head(
        dictionary=['A'], query_buckets=(32, 64, 128, 256, 500),
        query_width_scale=16.)
    rois = torch.tensor([[0., 0., 0., 640., 32.]])  # Requested 320 -> 500.
    assert head.roi_query_counts(rois).tolist() == [500]
    assert len(head.validate_targets(rois, ['A' * 499])[0]) == 499
    targets, lengths = head.encode_targets(['A' * 499], 500)
    assert targets.shape == (1, 500) and lengths.item() == 499
    assert (targets[0, :499] == 0).all()
    assert targets[0, 499].item() == head.dictionary.eos_idx
    with pytest.raises(ValueError, match='needs 501.*bucket is 500'):
        head.validate_targets(rois, ['A' * 500])

