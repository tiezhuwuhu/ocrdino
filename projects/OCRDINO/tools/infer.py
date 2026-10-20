"""Run a single shared-backbone OCRDINO checkpoint and write pixel-space JSON."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('image')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float)
    args = parser.parse_args()
    output = Path(args.out).resolve()
    if output in [Path(path).resolve() for path in (args.image, args.checkpoint, args.config)]:
        parser.error('Output must not overwrite image, checkpoint or config')
    from mmengine import Config
    from mmengine.dataset import Compose
    from mmdet.apis import inference_detector, init_detector
    cfg = Config.fromfile(args.config)
    if args.score_thr is not None:
        cfg.model.rec_test_cfg.score_thr = args.score_thr
    # This project intentionally has no validation/test dataset configured.
    # Avoid init_detector's palette lookup and inference's dataset pipeline lookup.
    model = init_detector(cfg, args.checkpoint, device=args.device, palette='random')
    prediction = inference_detector(
        model, args.image, test_pipeline=Compose(cfg.test_pipeline)).pred_instances.cpu()
    rows = []
    for index, text in enumerate(prediction.texts):
        rows.append(dict(
            bbox=prediction.bboxes[index].tolist(), text=text,
            detection_score=float(prediction.scores[index]),
            text_score=float(prediction.text_scores[index]),
            ocr_score=float(prediction.ocr_scores[index]),
            text_terminated=bool(prediction.text_terminated[index])))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(
        image=str(Path(args.image).resolve()), bbox_format='xyxy',
        coordinate_space='original_image_pixels', predictions=rows),
        ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved {} text regions to {}'.format(len(rows), output))


if __name__ == '__main__':
    main()
