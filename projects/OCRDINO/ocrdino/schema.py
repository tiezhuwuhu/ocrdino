"""Dependency-free validation for the aggregated OCR training JSON format."""
import math
from numbers import Real
from pathlib import PurePosixPath


DATASET_VERSION = 'ocrdino.text.v1'


def positive_id(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_image(image):
    if not isinstance(image, dict) or not positive_id(image.get('id')):
        raise ValueError('Image id must be a positive integer')
    path = image.get('file_name')
    if not isinstance(path, str) or not path:
        raise ValueError('Image file_name must be a nonempty relative path')
    normalized = path.replace('\\', '/')
    if PurePosixPath(normalized).is_absolute() or '..' in normalized.split('/') or ':' in normalized:
        raise ValueError('Image file_name must stay inside image root: ' + path)
    for key in ('width', 'height'):
        if not positive_id(image.get(key)):
            raise ValueError('Image ' + key + ' must be a positive integer')
    if 'annotation_complete' in image and not isinstance(image['annotation_complete'], bool):
        raise ValueError('annotation_complete must be a JSON boolean')


def validate_bbox(bbox, width, height):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError('bbox must contain [x, y, width, height]')
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in bbox):
        raise ValueError('bbox coordinates must be finite numbers')
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError('bbox must have positive width and height')
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError('bbox extends outside image bounds')
    return [float(x), float(y), float(x + w), float(y + h)]


def validate_document(document):
    """Reject malformed training geometry instead of silently dropping it."""
    if not isinstance(document, dict) or document.get('schema_version') != DATASET_VERSION:
        raise ValueError('Dataset must declare schema_version=' + DATASET_VERSION)
    if document.get('categories') != [{'id': 1, 'name': 'text'}]:
        raise ValueError('categories must equal [{"id": 1, "name": "text"}]')
    if not isinstance(document.get('images'), list) or not document['images']:
        raise ValueError('images must be a nonempty list')
    if not isinstance(document.get('annotations'), list):
        raise ValueError('annotations must be a list')
    images = {}
    file_names = set()
    for image in document['images']:
        validate_image(image)
        if image['id'] in images or image['file_name'].replace('\\', '/') in file_names:
            raise ValueError('Duplicate image id or file_name')
        images[image['id']] = image
        file_names.add(image['file_name'].replace('\\', '/'))
    ids = set()
    for annotation in document['annotations']:
        if not isinstance(annotation, dict) or not positive_id(annotation.get('id')):
            raise ValueError('Annotation id must be a positive integer')
        if annotation['id'] in ids:
            raise ValueError('Duplicate annotation id: ' + str(annotation['id']))
        ids.add(annotation['id'])
        if annotation.get('image_id') not in images:
            raise ValueError('Annotation references an unknown image')
        if annotation.get('category_id') != 1:
            raise ValueError('Only category_id=1 (text) is supported')
        if not isinstance(annotation.get('text'), str):
            raise ValueError('Annotation text must be a string; use rec_ignore for damaged text')
        for key in ('ignore', 'rec_ignore'):
            if key in annotation and not isinstance(annotation[key], bool):
                raise ValueError(key + ' must be a JSON boolean')
        image = images[annotation['image_id']]
        try:
            validate_bbox(annotation.get('bbox'), image['width'], image['height'])
        except ValueError as error:
            raise ValueError('Annotation {}: {}'.format(annotation['id'], error)) from error
    return images
