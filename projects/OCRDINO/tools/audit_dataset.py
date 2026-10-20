"""Audit schema, dictionary, query capacity and CTC alignment before training."""
import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    '_ocrdino_audit_prepare', Path(__file__).with_name('prepare_dataset.py'))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def audit_dataset(document, dictionary, image_root=None, width_buckets=(32, 64, 128),
                  width_scale=16, max_queries=900):
    buckets = tuple(width_buckets)
    if not buckets or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                          for value in buckets) or list(buckets) != sorted(set(buckets)):
        raise ValueError('width_buckets must be increasing, unique positive integers')
    if not math.isfinite(width_scale) or width_scale <= 0 or max_queries <= 0:
        raise ValueError('width_scale and max_queries must be positive')
    report = dict(schema_version='ocrdino.audit.v1', images=0, total_annotations=0,
                  totalGT=0, validrec=0, maxchars=0, maxroisperimage=0, ignored=0,
                  recognition_ignored=0, incomplete=0, widthbucketcounts={}, errors=[])
    try:
        images = prepare.schema.validate_document(document)
    except ValueError as error:
        report['errors'].append(dict(kind='schema', reason=str(error)))
        report['ok'] = False
        return report
    report.update(images=len(images), total_annotations=len(document['annotations']),
                  incomplete=sum(not image.get('annotation_complete', True) for image in images.values()))
    counts, widths = Counter(), Counter()
    for annotation in document['annotations']:
        if annotation.get('ignore', False):
            report['ignored'] += 1
            continue
        counts[annotation['image_id']] += 1
        report['totalGT'] += 1
        if annotation.get('rec_ignore', False):
            report['recognition_ignored'] += 1
            continue
        text = annotation['text']
        details = dict(annotation_id=annotation['id'], image_id=annotation['image_id'],
                       bbox=annotation['bbox'], text=text)
        valid_text = True
        try:
            dictionary.encode(text)
        except ValueError as error:
            valid_text = False
            report['errors'].append(dict(kind='transcription', reason=str(error), **details))
        length = len(text)
        report['maxchars'] = max(report['maxchars'], length)
        requested_width = math.ceil(annotation['bbox'][2] / annotation['bbox'][3] * width_scale)
        width = next((bucket for bucket in buckets if bucket >= requested_width), buckets[-1])
        widths[str(width)] += 1
        required = length + sum(left == right for left, right in zip(text, text[1:]))
        if required > width:
            report['errors'].append(dict(kind='ctc_alignment', input_steps=width,
                                         required_steps=required,
                                         reason='CTC needs len(text) plus adjacent equal pairs; '
                                                'increase width buckets/scale or inspect annotation', **details))
        elif valid_text:
            report['validrec'] += 1
    report['maxroisperimage'] = max(counts.values(), default=0)
    report['widthbucketcounts'] = dict(sorted(widths.items(), key=lambda item: int(item[0])))
    for image_id, count in counts.items():
        if count > max_queries:
            report['errors'].append(dict(kind='query_capacity', image_id=image_id,
                                         ground_truth_regions=count, max_queries=max_queries,
                                         reason='More supervised text instances than detection queries'))
    if image_root is not None:
        for image in images.values():
            try:
                actual = prepare.image_size(Path(image_root) / image['file_name'])
                expected = (image['width'], image['height'])
                if actual != expected:
                    raise ValueError('Image dimensions {} disagree with annotation {}'.format(actual, expected))
            except (OSError, ValueError) as error:
                report['errors'].append(dict(kind='image', image_id=image['id'],
                                             file_name=image['file_name'], reason=str(error)))
    report['ok'] = not report['errors']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotation', help='Canonical ocrdino.text.v1 training JSON')
    parser.add_argument('--dictionary', required=True)
    parser.add_argument('--image-root', help='Optional: also verify files and image dimensions')
    parser.add_argument('--width-buckets', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--width-scale', type=float, default=16.)
    parser.add_argument('--max-queries', type=int, default=900)
    parser.add_argument('--output', help='Optional audit report JSON')
    args = parser.parse_args()
    prepare.guard_outputs([args.output], [args.annotation, args.dictionary], [args.image_root])
    dictionary = prepare.vocabulary.CharacterDictionary(args.dictionary)
    report = audit_dataset(prepare.read_json(args.annotation), dictionary, args.image_root,
                           args.width_buckets, args.width_scale, args.max_queries)
    if args.output:
        prepare.write_json(args.output, report)
    print(json.dumps(prepare.safe_json(report), ensure_ascii=True, indent=2))
    raise SystemExit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()
