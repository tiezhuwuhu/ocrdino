"""Integration contracts for DN correspondence and image-space recognition."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('mmcv._ext')
pytest.importorskip('mmengine')

from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from projects.OCRDINO.ocrdino.detector import CappedCdnQueryGenerator, OCRDINO
from projects.OCRDINO.ocrdino.heads import OCRDINOHead


def sample_with_boxes(count):
    boxes = torch.tensor([[i + 1., 2., i + 5., 6.] for i in range(count)]).reshape(-1, 4)
    sample = DetDataSample(metainfo=dict(img_shape=(64, 64), scale_factor=(2., 2.)))
    sample.gt_instances = InstanceData(
        bboxes=boxes, labels=torch.zeros(count, dtype=torch.long),
        texts=['A'] * count, rec_ignore=torch.zeros(count, dtype=torch.bool))
    return sample


def make_head():
    return OCRDINOHead(
        num_classes=1, embed_dims=8, num_pred_layer=2, as_two_stage=True,
        loss_cls=dict(type='FocalLoss', use_sigmoid=True),
        train_cfg=dict(assigner=dict(
            type='HungarianAssigner',
            match_costs=[dict(type='BBoxL1Cost', weight=5., box_format='xywh')])))


def test_dn_cap_targets_correspond_to_exact_sampled_gt():
    generator = CappedCdnQueryGenerator(
        num_classes=1, embed_dims=8, num_matching_queries=10,
        max_queries=4, group_cfg=dict(dynamic=True, num_dn_queries=4))
    sample = sample_with_boxes(7)
    labels, boxes, mask, meta = generator([sample])
    assert labels.shape == (1, 4, 8)
    assert boxes.shape == (1, 4, 4)
    assert mask.shape == (14, 14)
    assert len(sample.gt_instances) == 7  # Main GT never truncated.
    selected = meta['dn_gt_indices'][0]
    assert len(selected) == 2
    head = make_head()
    targets = head.get_dn_targets(
        [sample.gt_instances], [sample.metainfo], meta)
    normalized_targets = targets[2][0]
    for slot, gt_index in enumerate(selected.tolist()):
        x1, y1, x2, y2 = sample.gt_instances.bboxes[gt_index]
        expected = torch.stack([(x1 + x2) / 128, (y1 + y2) / 128,
                                (x2 - x1) / 64, (y2 - y1) / 64])
        torch.testing.assert_close(normalized_targets[slot], expected)


def test_empty_gt_dn_keeps_valid_empty_shapes():
    generator = CappedCdnQueryGenerator(
        num_classes=1, embed_dims=8, num_matching_queries=10,
        max_queries=4, group_cfg=dict(dynamic=True, num_dn_queries=4))
    labels, boxes, mask, meta = generator([sample_with_boxes(0)])
    assert labels.shape == (1, 0, 8)
    assert boxes.shape == (1, 0, 4)
    targets = make_head().get_dn_targets(
        [sample_with_boxes(0).gt_instances], [dict(img_shape=(64, 64))], meta)
    assert targets[0][0].numel() == 0


def test_ignored_region_and_incomplete_image_do_not_teach_background():
    head = make_head()
    empty = sample_with_boxes(0).gt_instances
    logits = torch.tensor([[2.], [2.]])
    predicted = torch.tensor([[.25, .25, .2, .2], [.75, .75, .1, .1]])
    meta = dict(img_shape=(100, 100),
                ocr_ignored_bboxes=torch.tensor([[10., 10., 40., 40.]]))
    targets = head._get_targets_single(logits, predicted, empty, meta)
    assert targets[1].tolist() == [0., 1.]
    assert targets[-1].tolist() == [1]
    meta['annotation_complete'] = False
    targets = head._get_targets_single(logits, predicted, empty, meta)
    assert targets[1].tolist() == [0., 0.]
    assert targets[-1].numel() == 0


def test_inference_crops_resized_coordinates_then_rescales_only_output():
    class BoxHead:
        def predict(self, **kwargs):
            assert kwargs['rescale'] is False
            return [InstanceData(
                bboxes=torch.tensor([[10., 20., 50., 40.], [1., 1., 4., 4.]]),
                labels=torch.zeros(2, dtype=torch.long), scores=torch.tensor([.8, .1]))]

    class Recognition:
        def predict(self, c2, rois):
            torch.testing.assert_close(rois, torch.tensor([[0., 10., 20., 50., 40.]]))
            return ['AB'], torch.tensor([.9])

    class Harness:
        bbox_head = BoxHead()
        recognition_head = Recognition()
        rec_test_cfg = dict(score_thr=.3, max_per_img=10)
        def _extract_shared(self, inputs):
            return (), torch.zeros(1, 256, 16, 16)
        def forward_transformer(self, features, samples):
            return {}
        def add_pred_to_datasample(self, samples, predictions):
            return predictions

    result = OCRDINO.predict(Harness(), None, [sample_with_boxes(0)])[0]
    torch.testing.assert_close(result.bboxes, torch.tensor([[5., 10., 25., 20.]]))
    assert result.texts == ['AB']
