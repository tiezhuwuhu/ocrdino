"""Convert the supplied MLT2017, SynText1/2 and Total-Text archives.

Uses their actual ASCII-index rec arrays, not string coercion or CTC decoding.
Original archives are read-only. Images are extracted only with --extract-images.
"""
import argparse
from collections import Counter
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from english_archives import DatasetSource


def load_helper(filename):
    path = Path(__file__).with_name(filename + '.py')
    spec = importlib.util.spec_from_file_location('_english_' + filename, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_helper('prepare_dataset')
DATASETS = ('mlt2017', 'syntext1', 'syntext2', 'totaltext')
TRAIN_SOURCES = ('mlt2017', 'syntext1', 'syntext2')
# Verified against source JPEG headers and annotation positions. Their boxes
# use raw JPEG pixels; the source JSON dimensions instead reflect EXIF display.
RAW_FRAME_REPAIRS = {
    ('totaltext', 'train', '0000894.jpg'): ((470, 400), (400, 470)),
    ('totaltext', 'test', '0000228.jpg'): ((294, 392), (392, 294)),
}


def decode_rec(rec, full_length_policy='rec-ignore'):
    """95 is nonrecoverable; 96 is trailing padding; 0..94 map to chr(id+32).

    Repeated letters are ordinary GT characters and must never be collapsed.
    A completely occupied 25-slot label may have been truncated upstream.
    """
    if full_length_policy not in ('rec-ignore', 'keep'):
        raise ValueError('Unknown full-length policy')
    if not isinstance(rec, list) or len(rec) != 25:
        return '', ['invalid_rec_length']
    if any(isinstance(code, bool) or not isinstance(code, int) or
           code < 0 or code > 96 for code in rec):
        return '', ['invalid_rec_code']
    reasons = []
    end = rec.index(96) if 96 in rec else len(rec)
    if any(code != 96 for code in rec[end:]):
        reasons.append('nontrailing_padding')
    codes = rec[:end]
    text = ''.join(chr(code + 32) if code < 95 else '\ufffd'
                   for code in codes)
    if 95 in codes:
        reasons.append('unrecoverable_code_95')
    if not text or not text.strip():
        reasons.append('empty_recognition')
    if end == 25 and full_length_policy == 'rec-ignore':
        reasons.append('full_25_slots_possible_truncation')
    return text, reasons


def clip_bbox(bbox, width, height):
    if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or
            any(isinstance(v, bool) or not isinstance(v, (float, int)) or
                not math.isfinite(v) for v in bbox)):
        raise ValueError('bbox must contain four finite numbers')
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError('bbox has nonpositive area')
    x1, y1 = max(0., x), max(0., y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    clipped = [x1, y1, x2 - x1, y2 - y1]
    prepare.schema.validate_bbox(clipped, width, height)
    return clipped


def chosen_width(bbox, buckets=(32, 64, 128), width_scale=16.):
    requested = math.ceil(bbox[2] / bbox[3] * width_scale)
    return next((width for width in buckets if width >= requested), buckets[-1])


def convert_document(source, dataset, split, full_length_policy='rec-ignore',
                     width_buckets=(32, 64, 128), width_scale=16.):
    """Convert in place to avoid retaining two large annotation object graphs."""
    if dataset not in DATASETS or split not in ('train', 'test'):
        raise ValueError('Unknown source dataset or split')
    if not isinstance(source, dict) or not isinstance(source.get('images'), list):
        raise ValueError('Source must contain images and annotations arrays')
    if not isinstance(source.get('annotations'), list):
        raise ValueError('Source must contain annotations array')
    summary = Counter()
    images, image_map, names, repairs = [], {}, set(), []
    for index, original in enumerate(source['images'], 1):
        old_id = original['id']
        name = original['file_name'].replace('\\', '/')
        if old_id in image_map or name in names:
            raise ValueError('Duplicate source image id/file_name')
        width, height = original['width'], original['height']
        image = dict(id=index, file_name='images/{}/{}/{}'.format(dataset, split, name),
                     width=width, height=height, annotation_complete=True,
                     source_dataset=dataset, source_split=split,
                     source_image_id=old_id, source_file_name=name)
        repair = RAW_FRAME_REPAIRS.get((dataset, split, name))
        if repair:
            expected, corrected = repair
            if (width, height) != expected:
                raise ValueError('Known EXIF case changed; audit source before repair: ' + name)
            image.update(width=corrected[0], height=corrected[1],
                         source_dimensions=[width, height],
                         image_repair='strip_exif_without_rotating_pixels')
            repairs.append(dict(file_name=image['file_name'], source_dimensions=list(expected),
                                corrected_dimensions=list(corrected),
                                operation=image['image_repair']))
        prepare.schema.validate_image(image)
        images.append(image)
        image_map[old_id] = image
        names.add(name)
    source['images'] = images
    invalid, annotation_ids, counts = [], set(), Counter()
    original_annotations = source['annotations']
    total_annotations = len(original_annotations)
    summary['source_images'] = len(images)
    summary['source_annotations'] = total_annotations
    summary['image_frame_repairs'] = len(repairs)
    next_id = 1
    errors = []
    for index in range(total_annotations):
        raw = original_annotations[index]
        source_id = raw.get('id')
        if source_id in annotation_ids:
            raise ValueError('Duplicate source annotation id: ' + str(source_id))
        annotation_ids.add(source_id)
        image = image_map.get(raw.get('image_id'))
        if image is None:
            invalid.append(dict(reason='unknown_image_id', raw=prepare.safe_json(raw)))
            summary['orphan_annotations'] += 1
            original_annotations[index] = None
            continue
        if raw.get('category_id') != 1:
            raise ValueError('Unexpected source category: ' + str(raw.get('category_id')))
        try:
            bbox = clip_bbox(raw.get('bbox'), image['width'], image['height'])
        except (ValueError, TypeError) as error:
            invalid.append(dict(image_id=image['id'], source_annotation_id=source_id,
                                reason=str(error), raw=prepare.safe_json(raw)))
            image['annotation_complete'] = False
            summary['invalid_geometry'] += 1
            original_annotations[index] = None
            continue
        text, reasons = decode_rec(raw.get('rec'), full_length_policy)
        ignored = prepare.source_flag(raw.get('iscrowd', 0), 'iscrowd')
        if 'ignore' in raw:
            ignored = ignored or prepare.source_flag(raw['ignore'], 'ignore')
        annotation = dict(id=next_id, image_id=image['id'], category_id=1,
                          bbox=bbox, text=text, ignore=ignored,
                          rec_ignore=bool(reasons) or ignored,
                          source_annotation_id=source_id)
        next_id += 1
        counts[image['id']] += int(not ignored)
        if list(raw['bbox']) != bbox:
            annotation['source_bbox'] = raw['bbox']
            summary['clipped_bboxes'] += 1
        if reasons:
            annotation['raw_rec'] = raw.get('rec')
            annotation['rec_ignore_reasons'] = reasons
            for reason in reasons:
                summary[reason] += 1
        if isinstance(raw.get('rec'), list) and 96 not in raw['rec']:
            summary['full_25_slot_labels'] += int(
                isinstance(raw.get('rec'), list) and len(raw['rec']) == 25)
        if text in ('###', '***'):
            # These synthetic literals have iscrowd=0. Do not invent an ignore
            # marker merely because other datasets use a similar convention.
            summary['literal_marker_transcriptions'] += 1
        if ignored:
            summary['ignored_regions'] += 1
        elif annotation['rec_ignore']:
            summary['recognition_ignored_regions'] += 1
        else:
            summary['recognition_supervised_regions'] += 1
            needed = len(text) + sum(a == b for a, b in zip(text, text[1:]))
            available = chosen_width(bbox, width_buckets, width_scale)
            if needed > available:
                errors.append(dict(kind='ctc_capacity', annotation_id=annotation['id'],
                                   source_annotation_id=source_id, image_id=image['id'],
                                   text=text, required_steps=needed, input_steps=available))
        # Replacing releases the original Bezier arrays and unused attributes.
        original_annotations[index] = annotation
        if index and index % 250000 == 0:
            print('{} {}: converted {:,}/{:,} labels'.format(
                dataset, split, index, total_annotations), flush=True)
    annotations = [item for item in original_annotations if item is not None]
    summary['output_annotations'] = len(annotations)
    summary['detection_supervised_regions'] = sum(counts.values())
    summary['images_with_no_detection_gt'] = sum(not counts[i['id']] for i in images)
    summary['max_detection_gt_per_image'] = max(counts.values(), default=0)
    if summary['max_detection_gt_per_image'] > 900:
        errors.append(dict(kind='query_capacity', max_gt=summary['max_detection_gt_per_image'],
                           configured_queries=900))
    document = dict(schema_version=prepare.schema.DATASET_VERSION,
                    categories=[dict(id=1, name='text')], images=images,
                    annotations=annotations, invalid_annotations=invalid)
    prepare.schema.validate_document(document)
    report = dict(dataset=dataset, split=split, summary=dict(summary),
                  full_length_policy=full_length_policy,
                  ctc_width_buckets=list(width_buckets), ctc_width_scale=width_scale,
                  image_repairs=repairs, errors=errors,
                  image_headers_checked=False, ok=not errors,
                  notes=[
                      'rec 0..94 maps to ASCII 32..126; 95 is unrecoverable; 96 is padding.',
                      'Source has no language field; ASCII-decodable is not an English language label.',
                      'Positive intersections are clipped; original boxes remain in source_bbox.',
                      'Full 25-slot text may be truncated upstream; missing characters cannot be recovered.',
                      'Image extraction and header validation occur only with --extract-images.'])
    return document, report


def verify_image(path, image):
    """Match what the configured OpenCV color decoder will display."""
    from PIL import Image
    with Image.open(path) as decoded:
        orientation = decoded.getexif().get(274, 1)
        size = decoded.size
        if orientation in (5, 6, 7, 8):
            size = size[::-1]
        if size != (image['width'], image['height']):
            raise ValueError('Image/annotation dimensions disagree: {} display={} JSON={}'.format(
                path, size, (image['width'], image['height'])))
        if orientation not in (None, 0, 1):
            # Orientation/coordinate consistency cannot be inferred just from
            # dimensions, especially for mirrors and square images.
            raise ValueError('Unexpected EXIF orientation {} in {}; audit its coordinate frame'.format(
                orientation, path))
        return dict(raw_width=decoded.width, raw_height=decoded.height,
                    exif_orientation=orientation)


def write_compact(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(document, stream, ensure_ascii=False,
                  allow_nan=False, separators=(',', ':'))
        stream.write('\n')
    temporary.replace(path)


def merge_pretraining(paths, output):
    """Merge only MLT + Syn1 + Syn2, retaining one source document at a time."""
    if [Path(path).stem for path in paths] != [name + '_train' for name in TRAIN_SOURCES]:
        raise ValueError('Pretraining must contain exactly MLT2017, SynText1 and SynText2 train')
    output = Path(output)
    temporary = output.with_suffix(output.suffix + '.partial')
    totals = dict(images=0, annotations=0, invalid_annotations=0)
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write('{"schema_version":"ocrdino.text.v1","categories":[{"id":1,"name":"text"}]')
        for key in ('images', 'annotations', 'invalid_annotations'):
            stream.write(',"' + key + '":[')
            first, image_offset, annotation_offset = True, 0, 0
            for path in paths:
                with Path(path).open(encoding='utf-8') as source_stream:
                    document = json.load(source_stream)
                for item in document.get(key, []):
                    if key == 'images':
                        item['id'] += image_offset
                    elif key == 'annotations':
                        item['id'] += annotation_offset
                        item['image_id'] += image_offset
                    elif 'image_id' in item:
                        item['image_id'] += image_offset
                    if not first:
                        stream.write(',')
                    json.dump(item, stream, ensure_ascii=False, allow_nan=False,
                              separators=(',', ':'))
                    first = False
                    totals[key] += 1
                image_offset += len(document['images'])
                annotation_offset += len(document['annotations'])
                del document
                gc.collect()
            stream.write(']')
        stream.write('}\n')
    temporary.replace(output)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, help='Directory containing the four original archives')
    parser.add_argument('--output-root', required=True, help='Separate output dataset root')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--extract-images', action='store_true')
    parser.add_argument('--temp-dir', help='Disk space for one nested ZIP at a time (up to about 10.4 GB)')
    parser.add_argument('--full-length-policy', choices=('rec-ignore', 'keep'), default='rec-ignore')
    parser.add_argument('--width-buckets', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--width-scale', type=float, default=16.)
    args = parser.parse_args()
    source_root, output_root = Path(args.source_root).resolve(), Path(args.output_root).resolve()
    if (source_root == output_root or source_root in output_root.parents or
            output_root in source_root.parents):
        parser.error('Source and output roots must be separate, non-nested directories')
    buckets = args.width_buckets
    if (not buckets or buckets != sorted(set(buckets)) or min(buckets) < 1 or
            not math.isfinite(args.width_scale) or args.width_scale <= 0):
        parser.error('Require positive increasing width buckets and positive scale')
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Datasets must not repeat')
    annotation_root = output_root / 'annotations'
    annotation_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in args.datasets:
        with DatasetSource(source_root, name, temp_dir=args.temp_dir) as archive:
            for split in (('train', 'test') if name == 'totaltext' else ('train',)):
                print('Reading {} {} annotations...'.format(name, split), flush=True)
                original = archive.load_annotations(split)
                document, report = convert_document(
                    original, name, split, args.full_length_policy,
                    tuple(buckets), args.width_scale)
                del original
                report_path = annotation_root / (name + '_' + split + '.audit.json')
                if not report['ok']:
                    prepare.write_json(report_path, report)
                    raise ValueError('Conversion cannot meet CTC/query capacity; inspect ' + str(report_path))
                if args.extract_images:
                    headers = Counter()
                    for index, item in enumerate(document['images'], 1):
                        destination = output_root / item['file_name']
                        resolved = destination.resolve()
                        if output_root not in resolved.parents:
                            raise ValueError('Image destination escaped output root')
                        archive.extract_image(
                            item['source_file_name'], split, destination,
                            strip_exif='image_repair' in item)
                        info = verify_image(destination, item)
                        headers[str(info['exif_orientation'])] += 1
                        if index % 10000 == 0:
                            print('{} {}: checked {:,} images'.format(name, split, index), flush=True)
                    report['image_headers_checked'] = True
                    report['image_header_orientations'] = dict(headers)
                output = annotation_root / (name + '_' + split + '.json')
                write_compact(output, document)
                prepare.write_json(report_path, report)
                reports.append(dict(dataset=name, split=split, **report['summary']))
                print(json.dumps(reports[-1], ensure_ascii=False), flush=True)
                del document
                gc.collect()
    if all(name in args.datasets for name in TRAIN_SOURCES):
        merged = merge_pretraining(
            [annotation_root / (name + '_train.json') for name in TRAIN_SOURCES],
            annotation_root / 'pretrain.json')
        print('Merged pretraining: ' + json.dumps(merged), flush=True)
    prepare.write_json(annotation_root / 'conversion_summary.json', dict(
        reports=reports, training_sources=list(TRAIN_SOURCES),
        totaltext_test_in_training=False, images_extracted=args.extract_images,
        full_length_policy=args.full_length_policy))
    print('Complete. JSON-only conversion does not prepare training images.'
          if not args.extract_images else 'Complete. Images extracted and headers checked.')


if __name__ == '__main__':
    main()
