"""Convert COCO-like or ReCTS text-region annotations to one audited JSON.

Requires only Python and Pillow (Pillow only when reading image dimensions).
No MMDetection/PyTorch installation is needed for dataset preparation.
"""
import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path


def _load_module(name):
    path = Path(__file__).resolve().parents[1] / 'ocrdino' / (name + '.py')
    spec = importlib.util.spec_from_file_location('_ocrdino_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


schema = _load_module('schema')
vocabulary = _load_module('vocabulary')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def read_json(path):
    with open(path, encoding='utf-8-sig') as handle:
        return json.load(handle)


def safe_json(value):
    """Preserve non-finite raw anomalies as explicit strings in valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_raw_value': str(value)}
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', errors='backslashreplace') as handle:
        json.dump(safe_json(data), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def guard_outputs(outputs, inputs=(), protected_roots=()):
    """Prevent command-line conversion/report paths from overwriting sources."""
    resolved_outputs = [Path(path).resolve() for path in outputs if path is not None]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError('Output JSON and audit/report paths must be different')
    resolved_inputs = {Path(path).resolve() for path in inputs if path is not None}
    roots = [Path(path).resolve() for path in protected_roots if path is not None]
    for output in resolved_outputs:
        if output in resolved_inputs:
            raise ValueError('Output must not overwrite an input/dictionary: ' + str(output))
        if any(output == root or root in output.parents for root in roots):
            raise ValueError('Output must be outside source image/annotation directories: ' + str(output))


def new_document():
    return dict(schema_version=schema.DATASET_VERSION, images=[], annotations=[],
                categories=[dict(id=1, name='text')], invalid_annotations=[])


def image_size(path):
    from PIL import Image
    with Image.open(path) as image:
        return image.size


def source_flag(value, name):
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError('{} must be boolean or 0/1, got {!r}'.format(name, value))


def points_to_bbox(points):
    if not isinstance(points, (list, tuple)) or len(points) < 8 or len(points) % 2:
        raise ValueError('points must contain at least four x/y vertices')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in points):
        raise ValueError('points must be finite numbers')
    pairs = list(zip(points[::2], points[1::2]))
    twice_area = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2)
                     in zip(pairs, pairs[1:] + pairs[:1]))
    if abs(twice_area) < 1e-8:
        raise ValueError('points describe a zero-area polygon')
    xs, ys = points[::2], points[1::2]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def _clip_bbox(bbox, image):
    # First check finite positive geometry, without requiring image bounds.
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError('bbox must contain [x, y, width, height]')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox):
        raise ValueError('bbox coordinates must be finite numbers')
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError('bbox must have positive width and height')
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(image['width'], x + w), min(image['height'], y + h)
    return [x1, y1, x2 - x1, y2 - y1]


def append_annotation(document, audit, image, raw, bbox, text,
                      dictionary=None, clip_boxes=False, source=None):
    """Keep valid boxes, mask bad text, quarantine invalid geometry with raw data."""
    try:
        original_bbox = bbox
        if clip_boxes:
            bbox = _clip_bbox(bbox, image)
        schema.validate_bbox(bbox, image['width'], image['height'])
    except (ValueError, TypeError) as error:
        image['annotation_complete'] = False
        document['invalid_annotations'].append(dict(
            image_id=image['id'], source=source, reason=str(error), raw=raw))
        audit['events'].append(dict(kind='invalid_geometry', image_id=image['id'],
                                    source=source, reason=str(error)))
        return
    ignore = source_flag(raw.get('ignore', raw.get('iscrowd', False)), 'ignore') or text == '###'
    rec_ignore = source_flag(raw.get('rec_ignore', False), 'rec_ignore') or ignore
    error = vocabulary.text_error(text)
    if not error and dictionary is not None and not rec_ignore:
        try:
            dictionary.encode(text)
        except ValueError as exception:
            raise ValueError('Image {}, annotation {}: {}. Convert without --dictionary '
                             'and build a dictionary from valid training transcripts, then '
                             're-audit with that dictionary.'.format(image['id'], source, exception)) from exception
    annotation = dict(id=len(document['annotations']) + 1, image_id=image['id'],
                      category_id=1, bbox=list(bbox), text=text if isinstance(text, str) else '',
                      ignore=ignore, rec_ignore=rec_ignore or bool(error))
    if error:
        annotation['raw_text'] = text
        annotation['rec_ignore_reason'] = error
        audit['events'].append(dict(kind='recognition_masked', image_id=image['id'],
                                    annotation_id=annotation['id'], source=source, reason=error,
                                    raw_text=text))
    if list(original_bbox) != list(bbox):
        annotation['raw_bbox'] = original_bbox
        audit['events'].append(dict(kind='bbox_clipped', image_id=image['id'],
                                    annotation_id=annotation['id'], before=original_bbox, after=bbox))
    document['annotations'].append(annotation)


def convert_coco(source, image_root=None, text_key='text', dictionary=None, clip_boxes=False):
    if not isinstance(source, dict) or not isinstance(source.get('images'), list):
        raise ValueError('COCO input must contain images and annotations lists')
    if not isinstance(source.get('annotations'), list):
        raise ValueError('COCO input must contain annotations list')
    document, audit = new_document(), dict(events=[])
    images, file_names = {}, set()
    for raw in source['images']:
        image = dict(id=raw['id'], file_name=raw['file_name'].replace('\\', '/'),
                     width=raw.get('width'), height=raw.get('height'),
                     annotation_complete=raw.get('annotation_complete', True))
        if image_root is not None:
            width, height = image_size(Path(image_root) / image['file_name'])
            if image['width'] is not None and image['height'] is not None and (width, height) != (image['width'], image['height']):
                raise ValueError('Image dimensions disagree with file: ' + image['file_name'])
            image.update(width=width, height=height)
        schema.validate_image(image)
        if image['id'] in images or image['file_name'] in file_names:
            raise ValueError('Duplicate image id or file_name in input')
        images[image['id']] = image
        file_names.add(image['file_name'])
        document['images'].append(image)
    for raw in source['annotations']:
        image = images.get(raw.get('image_id'))
        if image is None:
            document['invalid_annotations'].append(dict(
                image_id=raw.get('image_id'), reason='unknown image_id', raw=raw))
            audit['events'].append(dict(kind='orphan_annotation', raw=raw))
            continue
        append_annotation(document, audit, image, raw, raw.get('bbox'), raw.get(text_key),
                          dictionary, clip_boxes, source=raw.get('id'))
    # Preserve anomaly history when an already-normalized file is re-audited.
    document['invalid_annotations'].extend(source.get('invalid_annotations', []))
    return finish(document, audit)


def convert_rects(image_root, annotation_root, dictionary=None, clip_boxes=False):
    image_root, annotation_root = Path(image_root), Path(annotation_root)
    image_paths = sorted(path for path in image_root.rglob('*') if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise ValueError('No supported images found: ' + str(image_root))
    annotation_paths = sorted(annotation_root.rglob('*.json'))
    by_stem = {}
    for path in annotation_paths:
        if path.stem in by_stem:
            raise ValueError('Ambiguous ReCTS annotation basename: ' + path.stem)
        by_stem[path.stem] = path
    document, audit = new_document(), dict(events=[])
    used_stems = set()
    for index, path in enumerate(image_paths, 1):
        if path.stem in used_stems:
            raise ValueError('Ambiguous ReCTS image basename: ' + path.stem)
        used_stems.add(path.stem)
        width, height = image_size(path)
        image = dict(id=index, file_name=path.relative_to(image_root).as_posix(),
                     width=width, height=height, annotation_complete=True)
        document['images'].append(image)
        annotation_path = by_stem.get(path.stem)
        if annotation_path is None:
            image['annotation_complete'] = False
            audit['events'].append(dict(kind='missing_annotation_file', image_id=index,
                                        file_name=image['file_name']))
            continue
        raw_document = read_json(annotation_path)
        if not isinstance(raw_document, dict) or not isinstance(raw_document.get('lines'), list):
            raise ValueError('ReCTS annotation must contain lines list: ' + str(annotation_path))
        for line_index, raw in enumerate(raw_document['lines']):
            source = '{}:lines[{}]'.format(annotation_path.name, line_index)
            try:
                bbox = points_to_bbox(raw.get('points'))
            except (ValueError, TypeError) as error:
                image['annotation_complete'] = False
                document['invalid_annotations'].append(dict(
                    image_id=index, source=source, reason=str(error), raw=raw))
                audit['events'].append(dict(kind='invalid_geometry', image_id=index,
                                            source=source, reason=str(error)))
                continue
            append_annotation(document, audit, image, raw, bbox, raw.get('transcription'),
                              dictionary, clip_boxes, source)
    for stem in sorted(set(by_stem).difference(used_stems)):
        raw = read_json(by_stem[stem])
        document['invalid_annotations'].append(dict(
            source=str(by_stem[stem]), reason='annotation file has no matching image', raw=raw))
        audit['events'].append(dict(kind='orphan_annotation_file', source=str(by_stem[stem])))
    return finish(document, audit)


def finish(document, audit):
    schema.validate_document(document)
    audit['schema_version'] = schema.DATASET_VERSION
    audit['summary'] = dict(images=len(document['images']), annotations=len(document['annotations']),
                           invalid_annotations=len(document['invalid_annotations']),
                           incomplete_images=sum(not image.get('annotation_complete', True)
                                                 for image in document['images']),
                           ignored_regions=sum(item['ignore'] for item in document['annotations']),
                           recognition_ignored=sum(item['rec_ignore'] for item in document['annotations']),
                           events=dict(Counter(item['kind'] for item in audit['events'])))
    return document, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='format', required=True)
    coco = subparsers.add_parser('coco', help='Normalize one COCO-like bbox+text JSON')
    coco.add_argument('--input', required=True)
    coco.add_argument('--image-root', help='Read/verify dimensions from actual image files')
    coco.add_argument('--text-key', default='text')
    rects = subparsers.add_parser('rects', help='Aggregate ReCTS lines from per-image JSON files')
    rects.add_argument('--image-root', required=True)
    rects.add_argument('--annotation-root', required=True)
    for subparser in (coco, rects):
        subparser.add_argument('--output', required=True)
        subparser.add_argument('--audit-output', help='Default: output file with .audit.json suffix')
        subparser.add_argument('--dictionary', help='Optional strict character dictionary to audit unknown text')
        subparser.add_argument('--clip-boxes', action='store_true', help='Explicitly clip out-of-bounds boxes and audit every change')
    args = parser.parse_args()
    audit_output = args.audit_output or str(Path(args.output).with_suffix('.audit.json'))
    guard_outputs([args.output, audit_output],
                  [getattr(args, 'input', None), args.dictionary],
                  [args.image_root, getattr(args, 'annotation_root', None)])
    dictionary = vocabulary.CharacterDictionary(args.dictionary) if args.dictionary else None
    if args.format == 'coco':
        document, audit = convert_coco(read_json(args.input), args.image_root, args.text_key,
                                      dictionary, args.clip_boxes)
    else:
        document, audit = convert_rects(args.image_root, args.annotation_root, dictionary, args.clip_boxes)
    write_json(args.output, document)
    write_json(audit_output, audit)
    print(json.dumps(audit['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
