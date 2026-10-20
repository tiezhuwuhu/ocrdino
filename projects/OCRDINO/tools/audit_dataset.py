"""Audit schema, dictionary and geometry-selected Character Query capacity."""
import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    '_ocrdino_audit_prepare', Path(__file__).with_name('prepare_dataset.py'))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
capacity = prepare._load_module('recognition_capacity')


def audit_dataset(document, dictionary, image_root=None,
                  query_buckets=capacity.DEFAULT_BUCKETS, query_width_scale=16.,
                  visual_width_buckets=capacity.DEFAULT_BUCKETS,
                  visual_width_scale=16., max_queries=900):
    """Check every unmasked GT, independently of max_rois training subsampling."""
    query_buckets = capacity.validate_buckets(query_buckets, 'query_buckets')
    visual_width_buckets = capacity.validate_buckets(
        visual_width_buckets, 'visual_width_buckets')
    query_width_scale = capacity.validate_width_scale(
        query_width_scale, 'query_width_scale')
    visual_width_scale = capacity.validate_width_scale(
        visual_width_scale, 'visual_width_scale')
    if isinstance(max_queries, bool) or not isinstance(max_queries, int) or max_queries <= 0:
        raise ValueError('max_queries must be a positive integer')
    report = dict(
        schema_version='ocrdino.query_audit.v1', images=0, total_annotations=0,
        totalGT=0, validrec=0, maxchars=0, maxroisperimage=0, ignored=0,
        recognition_ignored=0, incomplete=0, querybucketcounts={},
        visualwidthbucketcounts={}, errors=[], warnings=[],
        query_buckets=list(query_buckets), query_width_scale=query_width_scale,
        visual_width_buckets=list(visual_width_buckets),
        visual_width_scale=visual_width_scale, max_detection_queries=max_queries,
        capacity_rule='len(text) + 1 EOS; buckets selected from bbox geometry only')
    try:
        images = prepare.schema.validate_document(document)
    except ValueError as error:
        report['errors'].append(dict(kind='schema', reason=str(error)))
        report['ok'] = False
        return report
    report.update(
        images=len(images), total_annotations=len(document['annotations']),
        incomplete=sum(not image.get('annotation_complete', True) for image in images.values()))
    counts, queries, widths = Counter(), Counter(), Counter()
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
        report['maxchars'] = max(report['maxchars'], len(text))
        box_width, box_height = annotation['bbox'][2:]
        slots = capacity.select_geometry_bucket(
            box_width, box_height, query_buckets, query_width_scale)
        width = capacity.select_geometry_bucket(
            box_width, box_height, visual_width_buckets, visual_width_scale)
        queries[str(slots)] += 1
        widths[str(width)] += 1
        required = capacity.required_output_positions(text)
        if required > slots:
            report['errors'].append(dict(
                kind='character_query_capacity', query_bucket=slots,
                required_positions=required, visual_width=width,
                reason='Geometry-selected Query bucket cannot contain text plus EOS; '
                       'increase geometric capacity or inspect annotation. Text is not truncated.',
                **details))
        elif valid_text:
            report['validrec'] += 1
        if width < required:
            report['warnings'].append(dict(
                kind='visual_resolution', visual_width=width, required_positions=required,
                query_bucket=slots, reason='Fewer visual tokens than characters plus EOS; '
                'this is not a mathematical decoding limit, but visual detail may be insufficient.',
                **details))
    report['maxroisperimage'] = max(counts.values(), default=0)
    report['querybucketcounts'] = dict(sorted(queries.items(), key=lambda item: int(item[0])))
    report['visualwidthbucketcounts'] = dict(sorted(widths.items(), key=lambda item: int(item[0])))
    for image_id, count in counts.items():
        if count > max_queries:
            report['errors'].append(dict(
                kind='detection_query_capacity', image_id=image_id,
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
    parser.add_argument('--query-buckets', type=int, nargs='+', default=list(capacity.DEFAULT_BUCKETS))
    parser.add_argument('--query-width-scale', type=float, default=16.)
    parser.add_argument('--visual-width-buckets', type=int, nargs='+', default=list(capacity.DEFAULT_BUCKETS))
    parser.add_argument('--visual-width-scale', type=float, default=16.)
    parser.add_argument('--max-queries', type=int, default=900, help='DINO detection-query count')
    parser.add_argument('--output', help='Optional audit report JSON')
    args = parser.parse_args()
    prepare.guard_outputs([args.output], [args.annotation, args.dictionary], [args.image_root])
    dictionary = prepare.vocabulary.CharacterDictionary(args.dictionary)
    report = audit_dataset(
        prepare.read_json(args.annotation), dictionary, args.image_root,
        query_buckets=args.query_buckets, query_width_scale=args.query_width_scale,
        visual_width_buckets=args.visual_width_buckets,
        visual_width_scale=args.visual_width_scale, max_queries=args.max_queries)
    if args.output:
        prepare.write_json(args.output, report)
    print(json.dumps(prepare.safe_json(report), ensure_ascii=True, indent=2))
    raise SystemExit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()