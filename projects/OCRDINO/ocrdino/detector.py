"""DINO text detection and a shared-C2 region recognizer, without core edits."""
import copy

import torch
from mmengine import MessageHub
from mmengine.structures import InstanceData

from mmdet.models.detectors.dino import DINO
from mmdet.models.layers import CdnQueryGenerator
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps


class CappedCdnQueryGenerator(CdnQueryGenerator):
    """Bound actual DN tokens; preserve a sampled-GT map for DN supervision."""

    def __init__(self, *args, max_queries=128, **kwargs):
        if max_queries < 2:
            raise ValueError('max_queries must allow at least one DN pair')
        super().__init__(*args, **kwargs)
        self.max_queries = int(max_queries)

    def get_num_groups(self, max_num_target=None):
        if not max_num_target:
            return 1
        return max(1, self.max_queries // (2 * max_num_target))

    def __call__(self, batch_data_samples):
        samples, indices = [], []
        for sample in batch_data_samples:
            gt = sample.gt_instances
            device = gt.labels.device
            if len(gt) > self.max_queries // 2:
                selected = torch.randperm(len(gt), device=device)[
                    :self.max_queries // 2].sort().values
            else:
                selected = torch.arange(len(gt), device=device)
            # Only DN sees this subset. The detector and recognizer retain GT.
            dn_sample = copy.copy(sample)
            dn_sample.gt_instances = gt[selected]
            samples.append(dn_sample)
            indices.append(selected)
        labels, boxes, mask, meta = super().__call__(samples)
        meta['dn_gt_indices'] = indices
        return labels, boxes, mask, meta


@MODELS.register_module()
class OCRDINO(DINO):
    """A single backbone, four-scale DINO, and one final-stage parallel character-query head."""

    def __init__(self, *args, recognition_head, dn_max_queries=128,
                 rec_train_cfg=None, rec_test_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        if tuple(self.backbone.out_indices) != (0, 1, 2, 3):
            raise ValueError('OCRDINO needs backbone.out_indices=(0,1,2,3)')
        if self.bbox_head.num_classes != 1:
            raise ValueError('Detection has one text category; characters belong to the region recognizer')
        from .heads import OCRDINOHead
        if not isinstance(self.bbox_head, OCRDINOHead):
            raise TypeError('Use OCRDINOHead for aligned capped-DN supervision')
        self.recognition_head = MODELS.build(recognition_head)
        original = self.dn_query_generator
        self.dn_query_generator = CappedCdnQueryGenerator(
            num_classes=self.bbox_head.num_classes,
            embed_dims=self.embed_dims, num_matching_queries=self.num_queries,
            label_noise_scale=original.label_noise_scale,
            box_noise_scale=original.box_noise_scale,
            group_cfg=dict(dynamic=True, num_dn_queries=dn_max_queries),
            max_queries=dn_max_queries)
        self.rec_train_cfg = dict(
            max_rois=128, pred_box_prob=0., pred_box_start_epoch=5,
            min_pred_iou=.7)
        self.rec_train_cfg.update(rec_train_cfg or {})
        self.rec_test_cfg = dict(score_thr=.3, max_per_img=100)
        self.rec_test_cfg.update(rec_test_cfg or {})
        if not 0 <= self.rec_train_cfg['pred_box_prob'] <= 1:
            raise ValueError('pred_box_prob must be in [0,1]')
        if self.rec_train_cfg['max_rois'] is not None and self.rec_train_cfg['max_rois'] < 1:
            raise ValueError('max_rois must be positive or None')
        if self.rec_test_cfg['max_per_img'] < 1:
            raise ValueError('max_per_img must be positive')

    def _extract_shared(self, batch_inputs):
        features = self.backbone(batch_inputs)
        # No second backbone and no stride-4 tokens in the DINO encoder.
        return self.neck(features[1:]), features[0]

    def extract_feat(self, batch_inputs):
        # The framework tensor mode returns raw detection tensors.
        return self._extract_shared(batch_inputs)[0]

    @torch.no_grad()
    def _matched_boxes(self, head_inputs, samples):
        """Optional late-training crops: match final predictions to GT."""
        cls, boxes = self.bbox_head(
            head_inputs['hidden_states'], head_inputs['references'])
        dn = head_inputs['dn_meta']['num_denoising_queries']
        cls, boxes = cls[-1, :, dn:], boxes[-1, :, dn:]
        results = []
        for logits, normalized, sample in zip(cls, boxes, samples):
            h, w = sample.img_shape[:2]
            predicted = bbox_cxcywh_to_xyxy(normalized) * normalized.new_tensor(
                [w, h, w, h])
            predicted[:, 0::2].clamp_(0, w)
            predicted[:, 1::2].clamp_(0, h)
            gt = sample.gt_instances
            crops = gt.bboxes.clone()
            if len(gt):
                match = self.bbox_head.assigner.assign(
                    pred_instances=InstanceData(scores=logits, bboxes=predicted),
                    gt_instances=gt, img_meta=sample.metainfo)
                positive = torch.where(match.gt_inds > 0)[0]
                assigned = match.gt_inds[positive] - 1
                overlaps = bbox_overlaps(
                    predicted[positive], gt.bboxes[assigned], is_aligned=True)
                # Keep GT if prediction would lose too much of the text region.
                coverage = bbox_overlaps(
                    gt.bboxes[assigned], predicted[positive], mode='iof',
                    is_aligned=True)
                candidate_rois = torch.cat([
                    predicted.new_zeros((len(positive), 1)), predicted[positive]], dim=1)
                sizes = candidate_rois[:, 3:5] - candidate_rois[:, 1:3]
                valid_geometry = (sizes > 0).all(dim=1) & torch.isfinite(candidate_rois).all(dim=1)
                candidate_slots = torch.zeros_like(positive)
                candidate_slots[valid_geometry] = self.recognition_head.roi_query_counts(
                    candidate_rois[valid_geometry])
                required = candidate_slots.new_tensor([
                    self.recognition_head.required_output_positions(gt.texts[i])
                    if not gt.rec_ignore[i] else 0 for i in assigned.tolist()])
                use = ((overlaps >= self.rec_train_cfg['min_pred_iou']) &
                       (coverage >= .95) &
                       valid_geometry &
                       (candidate_slots >= required) &
                       (torch.rand_like(overlaps) < self.rec_train_cfg['pred_box_prob']))
                crops[assigned[use]] = predicted[positive[use]]
            results.append(crops)
        return results

    def _training_rois(self, samples, c2, head_inputs):
        # Check EVERY valid GT before optional predicted crops or max_rois
        # sampling: an oversized label must never disappear by random chance.
        gt_rois, texts, locations = [], [], []
        for batch_idx, sample in enumerate(samples):
            gt = sample.gt_instances
            if not hasattr(gt, 'texts') or not hasattr(gt, 'rec_ignore'):
                raise ValueError('Use OCRLoadAnnotations and OCRPackDetInputs')
            for index in torch.where(~gt.rec_ignore.bool())[0].tolist():
                box = gt.bboxes[index].detach().to(device=c2.device, dtype=torch.float32)
                gt_rois.append(torch.cat([box.new_tensor([batch_idx]), box]))
                texts.append(gt.texts[index])
                locations.append((batch_idx, index))
        if not gt_rois:
            return c2.new_zeros((0, 5), dtype=torch.float32), []
        rois = torch.stack(gt_rois)
        self.recognition_head.validate_targets(rois, texts)
        epoch = MessageHub.get_current_instance().get_info('epoch') or 0
        use_pred = (self.rec_train_cfg['pred_box_prob'] > 0 and
                    epoch >= self.rec_train_cfg['pred_box_start_epoch'])
        if use_pred:
            boxes = self._matched_boxes(head_inputs, samples)
            rois = torch.stack([
                torch.cat([rois[row, :1], boxes[batch_idx][index].detach().to(rois)])
                for row, (batch_idx, index) in enumerate(locations)])
            self.recognition_head.validate_targets(rois, texts)
        maximum = self.rec_train_cfg['max_rois']
        if maximum is not None and len(rois) > maximum:
            selected = torch.randperm(len(rois), device=rois.device)[:maximum]
            rois = rois[selected]
            texts = [texts[index] for index in selected.tolist()]
        return rois, texts

    def loss(self, batch_inputs, batch_data_samples):
        for sample in batch_data_samples:
            if len(sample.gt_instances) > self.num_queries:
                raise ValueError('More text GT than detection queries; increase num_queries')
        detection_features, c2 = self._extract_shared(batch_inputs)
        inputs = self.forward_transformer(detection_features, batch_data_samples)
        losses = self.bbox_head.loss(
            **inputs, batch_data_samples=batch_data_samples)
        rois, texts = self._training_rois(batch_data_samples, c2, inputs)
        losses.update(self.recognition_head.loss(c2, rois, texts))
        # An empty local GT batch still participates in DDP for the DN embedding.
        losses['loss_dn_embedding_zero'] = self.dn_query_generator.label_embedding.weight.sum() * 0
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        detection_features, c2 = self._extract_shared(batch_inputs)
        inputs = self.forward_transformer(detection_features, batch_data_samples)
        predictions = self.bbox_head.predict(
            **inputs, batch_data_samples=batch_data_samples, rescale=False)
        filtered, rois = [], []
        for index, instances in enumerate(predictions):
            boxes = instances.bboxes
            valid = ((instances.scores >= self.rec_test_cfg['score_thr']) &
                     torch.isfinite(boxes).all(dim=1) &
                     (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
            instances = instances[valid]
            order = instances.scores.argsort(descending=True)[
                :self.rec_test_cfg['max_per_img']]
            instances = instances[order]
            filtered.append(instances)
            rois.append(torch.cat([
                instances.bboxes.new_full((len(instances), 1), index),
                instances.bboxes], dim=1))
        text, confidence, terminated = self.recognition_head.predict(c2, torch.cat(rois))
        offset = 0
        for instances, sample in zip(filtered, batch_data_samples):
            count = len(instances)
            instances.texts = text[offset:offset + count]
            instances.text_scores = confidence[offset:offset + count]
            instances.ocr_scores = instances.scores * instances.text_scores
            instances.text_terminated = terminated[offset:offset + count]
            if rescale:
                scale = instances.bboxes.new_tensor(sample.scale_factor)
                if scale.numel() == 2:
                    scale = scale.repeat(2)
                if scale.numel() != 4:
                    raise ValueError('scale_factor must have 2 or 4 elements')
                instances.bboxes = instances.bboxes / scale
            offset += count
        return self.add_pred_to_datasample(batch_data_samples, filtered)
