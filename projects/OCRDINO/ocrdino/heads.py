"""DINO losses with bounded-DN correspondence and ignored-region support."""
import torch

from mmdet.models.dense_heads.dino_head import DINOHead
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps


@MODELS.register_module()
class OCRDINOHead(DINOHead):
    def __init__(self, *args, ignore_iof_thr=.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_iof_thr = ignore_iof_thr

    def loss(self, hidden_states, references, enc_outputs_class,
             enc_outputs_coord, batch_data_samples, dn_meta):
        metas = []
        for sample in batch_data_samples:
            meta = dict(sample.metainfo)
            ignored = sample.get('ignored_instances', None)
            if ignored is not None and 'bboxes' in ignored:
                meta['ocr_ignored_bboxes'] = ignored.bboxes
            metas.append(meta)
        scores, boxes = self(hidden_states, references)
        return self.loss_by_feat(
            scores, boxes, enc_outputs_class, enc_outputs_coord,
            [s.gt_instances for s in batch_data_samples], metas, dn_meta)

    def _get_targets_single(self, cls_score, bbox_pred, gt_instances, img_meta):
        output = super()._get_targets_single(
            cls_score, bbox_pred, gt_instances, img_meta)
        labels, weights, targets, box_weights, positives, negatives = output
        if not img_meta.get('annotation_complete', True):
            weights[negatives] = 0
        ignored = img_meta.get('ocr_ignored_bboxes')
        if ignored is not None and len(ignored) and len(negatives):
            h, w = img_meta['img_shape'][:2]
            pixel_boxes = bbox_cxcywh_to_xyxy(bbox_pred[negatives]) * bbox_pred.new_tensor(
                [w, h, w, h])
            overlap = bbox_overlaps(pixel_boxes, ignored.to(pixel_boxes), mode='iof')
            weights[negatives[overlap.amax(dim=1) >= self.ignore_iof_thr]] = 0
        # Excluded negatives must not dilute the classification normalization.
        negatives = negatives[weights[negatives] > 0]
        return labels, weights, targets, box_weights, positives, negatives

    def get_dn_targets(self, batch_gt_instances, batch_img_metas, dn_meta):
        indices = dn_meta.get('dn_gt_indices')
        if indices is not None:
            batch_gt_instances = [
                gt[selected] for gt, selected in zip(batch_gt_instances, indices)]
        return super().get_dn_targets(
            batch_gt_instances, batch_img_metas, dn_meta)
