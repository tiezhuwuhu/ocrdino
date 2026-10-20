"""Real CPU full-model smoke; synthetic inputs, no CUDA/accuracy claim."""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='projects/OCRDINO/configs/ocrdino_r50_4scale_query.py')
    parser.add_argument('--checkpoint', default='checkpoints/ocrdino_init.pth')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', default='projects/OCRDINO/reports/query_validation/cpu_model_smoke.json')
    args = parser.parse_args()
    import torch
    import mmcv
    import mmengine
    from mmengine import Config
    from mmengine.logging import MMLogger
    from mmengine.runner import load_checkpoint
    from mmengine.structures import InstanceData
    from mmdet.registry import MODELS
    from mmdet.structures import DetDataSample
    from mmdet.utils import register_all_modules
    torch.set_num_threads(args.threads)
    torch.manual_seed(3407)
    register_all_modules()
    logger = MMLogger.get_current_instance()
    logger.setLevel(logging.WARNING)
    cfg = Config.fromfile(args.config)
    started = time.perf_counter()
    model = MODELS.build(cfg.model)
    model.init_weights()
    load_checkpoint(model, args.checkpoint, map_location='cpu', logger=logger)
    assert model.num_queries == 900 and model.num_feature_levels == 4
    assert len(model.encoder.layers) == 6 and len(model.decoder.layers) == 6
    assert len(model.recognition_head.decoder) == 3
    assert model.backbone.with_cp
    # Exercise recognition even though random new text scores can be below .3.
    # This is only a smoke postprocessing override, not a training/model change.
    model.rec_test_cfg.update(score_thr=0., max_per_img=3)
    build_seconds = time.perf_counter() - started

    def make_batch(empty=False):
        sample = DetDataSample(metainfo=dict(
            img_id=0, img_path='synthetic_tensor',
            ori_shape=(512, 512), img_shape=(256, 256), scale_factor=(.5, .5),
            annotation_complete=True))
        boxes = torch.empty(0, 4) if empty else torch.tensor([
            [16., 24., 176., 56.], [40., 120., 168., 152.]])
        texts = [] if empty else ['HELLO', 'BOOK']
        sample.gt_instances = InstanceData(
            bboxes=boxes, labels=torch.zeros(len(boxes), dtype=torch.long),
            texts=texts, rec_ignore=torch.zeros(len(boxes), dtype=torch.bool))
        sample.ignored_instances = InstanceData(
            bboxes=torch.empty(0, 4), labels=torch.empty(0, dtype=torch.long))
        return model.data_preprocessor(dict(
            inputs=[torch.rand(3, 256, 256) * 255], data_samples=[sample]),
            training=True)

    reports = []
    for empty in (False, True):
        started = time.perf_counter()
        model.train()
        model.zero_grad(set_to_none=True)
        batch = make_batch(empty)
        losses = model(**batch, mode='loss')
        total, logs = model.parse_losses(losses)
        assert torch.isfinite(total)
        total.backward()
        norms = {}
        for name, module in [
            ('backbone', model.backbone), ('encoder', model.encoder),
            ('decoder', model.decoder), ('recognition', model.recognition_head)]:
            gradients = [p.grad for p in module.parameters() if p.requires_grad]
            assert all(g is not None and torch.isfinite(g).all() for g in gradients), name
            norms[name] = float(torch.sqrt(sum(g.square().sum() for g in gradients)))
        if empty:
            assert losses['loss_rec'].item() == 0. and norms['recognition'] == 0.
        else:
            assert losses['loss_rec'].item() > 0. and norms['recognition'] > 0.
        assert model.dn_query_generator.label_embedding.weight.grad is not None
        reports.append(dict(
            empty_gt=empty, loss=float(total.detach()),
            loss_rec=float(losses['loss_rec'].detach()), gradient_norms=norms,
            input_shape=list(batch['inputs'].shape),
            seconds=time.perf_counter() - started))
        print(json.dumps(reports[-1]), flush=True)

    model.zero_grad(set_to_none=True)
    model.train()
    started = time.perf_counter()
    recognition_only = model(**make_batch(False), mode='loss')['loss_rec']
    recognition_only.backward()
    stem_gradient = model.backbone.conv1.weight.grad
    assert stem_gradient is not None and torch.isfinite(stem_gradient).all()
    assert stem_gradient.abs().sum() > 0
    recognition_only_proof = dict(
        loss_rec=float(recognition_only.detach()),
        backbone_stem_gradient_norm=float(stem_gradient.norm()),
        seconds=time.perf_counter() - started)
    model.zero_grad(set_to_none=True)
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        prediction = model(**make_batch(False), mode='predict')[0].pred_instances
    count = len(prediction)
    assert count == 3
    assert len(prediction.texts) == len(prediction.text_scores) == len(prediction.text_terminated) == count
    torch.testing.assert_close(prediction.ocr_scores, prediction.scores * prediction.text_scores)
    assert torch.isfinite(prediction.bboxes).all()
    assert (prediction.bboxes >= 0).all() and (prediction.bboxes <= 512).all()
    report = dict(
        status='passed', device='cpu', torch=torch.__version__, mmcv=mmcv.__version__,
        mmengine=mmengine.__version__, cuda_tested=False, amp_tested=False,
        data='synthetic RGB tensors and HELLO/BOOK bbox+text; not a dataset accuracy test',
        checkpoint=str(Path(args.checkpoint).resolve()), build_seconds=build_seconds,
        structure=dict(backbone='ResNet50', scales=4, detection_queries=900,
                       encoder_layers=6, decoder_layers=6, character_decoder_layers=3),
        smoke_postprocessing_override=dict(score_thr=0., max_per_img=3),
        training=reports, recognition_only_backward=recognition_only_proof,
        prediction_count=count,
        prediction_seconds=time.perf_counter() - started,
        outputs=dict(bboxes=prediction.bboxes.tolist(), texts=prediction.texts,
                     text_terminated=prediction.text_terminated.tolist()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

